# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
High-level API for fused matrix multiplication and all-reduce.

This module provides a torch-like interface for GEMM+All-Reduce operations,
automatically inferring dimensions, strides, and hardware parameters.
"""

from typing import Optional
import torch
import triton
import triton.language as tl

from tritonblas.kernels.stages import GemmContext, ScheduleContext, make_tensor_view

from .config import FusedConfig
from .workspace import FusedWorkspace
import iris
import iris.x


@triton.jit()
def _fused_matmul_all_reduce_kernel(
    A,
    B,
    C,
    aux_buffer,
    locks,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr,
    VARIANT: tl.constexpr,
    call_number,
):
    """
    Persistent fused GEMM + All-Reduce kernel.

    Each CTA iterates over multiple output tiles via ScheduleContext
    (persistent GEMM pattern). For each tile it computes GEMM, then
    dispatches to the chosen all-reduce variant. Non-responsible CTAs
    in two_shot immediately advance to the next tile while responsible
    CTAs perform the cross-rank reduce-scatter.

    Args:
        A, B, C: Input/output matrix pointers
        aux_buffer: Symmetric-heap buffer for one_shot/two_shot staging
        locks: Versioned lock array (one int32 per tile, on symmetric heap)
        M, N, K: Matrix dimensions
        stride_*: Tensor strides
        context_tensor: Iris DeviceContext tensor for RMA
        cur_rank, world_size: Rank info (constexpr)
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile dimensions
        GROUP_SIZE_M: Tile-swizzle group size for L2 locality
        NUM_SMS: Number of SMs (persistent grid size)
        NUM_XCDS: Number of chiplets for XCD-aware scheduling
        EVEN_K: Whether K is divisible by BLOCK_SIZE_K
        VARIANT: All-reduce algorithm
        call_number: Monotonic version for versioned locks (runtime int)
    """
    tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
    tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
    gemm_ctx = GemmContext(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_sms=NUM_SMS,
        num_xcds=NUM_XCDS,
        group_size_m=GROUP_SIZE_M,
        even_k=EVEN_K,
    )
    sched = ScheduleContext(M, N, K, gemm_ctx)

    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    dst_view = iris.x.make_tensor_view(C, M, N, stride_cm, stride_cn)

    start, total, stride = sched.persistent_tile_range()
    for tile_idx in range(start, total, stride):
        out_tile = sched.get_tile_from_idx(tile_idx)
        pid_m = out_tile.pid_m
        pid_n = out_tile.pid_n

        acc = gemm_ctx.reduce_axis(tensorA, tensorB, out_tile)
        rm, rn = out_tile.indices()
        c = acc.to(C.type.element_ty)

        tile_obj = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)

        if VARIANT == "atomic":
            iris.x.all_reduce_atomic(tile_obj, dst_view, ctx)
        elif VARIANT == "spinlock":
            iris.x.all_reduce_spinlock(tile_obj, dst_view, locks, ctx)
        elif VARIANT == "one_shot" or VARIANT == "two_shot":
            temp_ptr = aux_buffer + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            tl.store(temp_ptr, c, mask=mask, cache_modifier=".wt")
            tl.debug_barrier()

            num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
            tile_id = pid_m * num_tiles_n + pid_n
            lock_ptr = locks + tile_id
            tl.atomic_xchg(lock_ptr, call_number, sem="release", scope="sys")

            src_view = iris.x.make_tensor_view(aux_buffer, M, N, stride_cm, stride_cn)

            if VARIANT == "one_shot":
                iris.x.all_reduce_one_shot(tile_obj, src_view, dst_view, locks, ctx, call_number)
            elif VARIANT == "two_shot":
                iris.x.all_reduce_two_shot(tile_obj, src_view, dst_view, locks, ctx, call_number)


def _allocate_workspace(
    shmem,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    config: FusedConfig,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Allocate workspace buffers (locks, aux_buffer) without zeroing or barriers.

    Called once when workspace doesn't match the problem dimensions. Subsequent
    calls with matching shapes reuse the existing buffers -- versioned locks
    and overwrite semantics (two_shot) eliminate the need for re-zeroing.
    """
    world_size = shmem.get_num_ranks()
    config.validate(world_size=world_size)

    if workspace is None:
        workspace = FusedWorkspace()

    workspace.operation = "matmul_all_reduce"
    workspace.shape = (M, N, K)
    workspace.dtype = dtype
    workspace.world_size = world_size
    workspace.variant = config.all_reduce_variant
    workspace.call_counter = 0

    num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
    num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_pid_m * num_pid_n

    if config.all_reduce_variant in ["spinlock", "one_shot", "two_shot"]:
        if workspace.locks is None or workspace.locks.numel() != total_tiles:
            workspace.locks = shmem.zeros((total_tiles,), dtype=torch.int32)
    else:
        workspace.locks = None

    if config.all_reduce_variant in ["one_shot", "two_shot"]:
        if workspace.aux_buffer is None or workspace.aux_buffer.shape != (M, N):
            workspace.aux_buffer = shmem.zeros((M, N), dtype=dtype)
    else:
        workspace.aux_buffer = None

    if workspace._barrier_tensor is None:
        workspace._barrier_tensor = torch.zeros(1, dtype=torch.int32, device=shmem.get_device())

    workspace.prepared = True
    return workspace


def _pre_kernel_sync(shmem, C, config, workspace):
    """
    Variant-specific pre-kernel preparation (GPU-stream ops only, no host sync).

    - atomic: C must be zeroed + stream-level cross-rank barrier
    - spinlock: C must be zeroed + stream-level cross-rank barrier
    - one_shot/two_shot: no zeroing needed (C overwritten, versioned locks)
    """
    import torch.distributed as dist

    if config.all_reduce_variant in ["atomic", "spinlock"]:
        C.zero_()
        dist.all_reduce(workspace._barrier_tensor)
    # one_shot/two_shot: C is overwritten by tl.store/iris.store, locks are versioned


def matmul_all_reduce(
    shmem,
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Fused matrix multiplication and all-reduce.

    Computes: C = all_reduce(A @ B) across all ranks.

    For lock-based variants (one_shot, two_shot), uses versioned locks to
    eliminate inter-call zeroing and barriers. For atomic/spinlock, uses a
    lightweight stream-level barrier (dist.all_reduce on a 1-element tensor)
    instead of host-side torch.cuda.synchronize().

    Args:
        shmem: Iris shmem context
        C: Output tensor (M, N) on symmetric heap
        A: Input matrix A (M, K)
        B: Input matrix B (K, N)
        async_op: If False, performs stream-level barrier at end. Default: False.
        config: Optional FusedConfig for tuning. If None, uses defaults.
        workspace: Optional pre-allocated workspace. If None, creates new one.

    Returns:
        workspace: Updated workspace object (reusable for subsequent calls)
    """
    import torch.distributed as dist

    if config is None:
        config = FusedConfig()

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2D tensors, got shapes {A.shape} and {B.shape}")

    M, K = A.shape
    K_B, N = B.shape

    if K != K_B:
        raise ValueError(
            f"Incompatible matrix dimensions: A is ({M}, {K}), B is ({K_B}, {N}). "
            f"Inner dimensions must match (K={K} != K_B={K_B})"
        )

    if C.shape != (M, N):
        raise ValueError(f"Output tensor shape {C.shape} doesn't match expected ({M}, {N})")

    if A.dtype != B.dtype or A.dtype != C.dtype:
        raise ValueError(f"All tensors must have same dtype, got A:{A.dtype}, B:{B.dtype}, C:{C.dtype}")

    assert M >= config.block_size_m, f"M={M} too small for block_size_m={config.block_size_m}"
    assert K >= config.block_size_k, f"K={K} too small for block_size_k={config.block_size_k}"
    assert N >= config.block_size_n, f"N={N} too small for block_size_n={config.block_size_n}"

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Allocate workspace once; reuse on subsequent calls with same shape
    needs_alloc = workspace is None or not workspace.matches(
        "matmul_all_reduce", (M, N, K), A.dtype, world_size, config.all_reduce_variant
    )
    if needs_alloc:
        workspace = _allocate_workspace(shmem, M, N, K, A.dtype, config, workspace=workspace)

    # Verify lock array is large enough for current tile count.  Block sizes
    # may differ across calls even when shape/variant match.  We do NOT
    # allocate here (shmem.zeros is collective and can't be called in the hot
    # path).  Callers must pre-allocate or re-create the workspace.
    num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
    num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_pid_m * num_pid_n

    if config.all_reduce_variant in ["spinlock", "one_shot", "two_shot"]:
        if workspace.locks is not None and workspace.locks.numel() < total_tiles:
            raise ValueError(
                f"Lock array too small: have {workspace.locks.numel()} locks but need {total_tiles} "
                f"(block_size_m={config.block_size_m}, block_size_n={config.block_size_n}). "
                f"Pre-allocate workspace with the smallest block sizes you intend to use."
            )

    # Increment versioned lock counter
    workspace.call_counter += 1

    # Variant-specific pre-kernel work (no host sync)
    _pre_kernel_sync(shmem, C, config, workspace)

    device_context = shmem.get_device_context()

    even_k = K % config.block_size_k == 0

    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(A.device)
        num_sms = props.multi_processor_count

    grid = (num_sms,)

    _fused_matmul_all_reduce_kernel[grid](
        A,
        B,
        C,
        workspace.aux_buffer,
        workspace.locks,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        device_context,
        rank,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        config.group_size_m,
        num_sms,
        config.num_xcds,
        even_k,
        config.all_reduce_variant,
        workspace.call_counter,
    )

    # Stream-level post-kernel sync (no host-side torch.cuda.synchronize)
    if not async_op:
        dist.all_reduce(workspace._barrier_tensor)

    return workspace
