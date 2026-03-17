# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
High-level API for fused matrix multiplication and all-reduce.

This module provides a torch-like interface for GEMM+All-Reduce operations,
automatically inferring dimensions, strides, and hardware parameters.
"""

import logging
from typing import Optional
import torch
import triton
import triton.language as tl

from tritonblas.kernels.stages import GemmContext, ScheduleContext, make_tensor_view

from .config import FusedConfig
from .workspace import FusedWorkspace
import iris
from iris.host.tracing.kernel_artifacts import iris_launch


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
    Persistent fused GEMM + All-Reduce kernel with configurable all-reduce variant.

    Computes C = all_reduce(A @ B) across all ranks using the specified variant.
    This is useful for data-parallel distributed training where each rank computes
    a partial result over different data, and then reduces across all ranks.

    Each CTA iterates over multiple output tiles via ``ScheduleContext``
    (persistent GEMM pattern with XCD-aware swizzling). For each tile it computes
    the GEMM, then dispatches to the chosen all-reduce variant. In ``two_shot``,
    non-responsible CTAs immediately advance to the next tile while responsible
    CTAs perform the cross-rank reduce-scatter.

    Supported variants:
    - 'atomic': Fast, lock-free atomic accumulation
    - 'spinlock': Mutex-based serialized read-modify-write
    - 'one_shot': Each rank reduces all tiles (duplicated work, no remote stores)
    - 'two_shot': Work distribution with reduce-scatter then all-gather pattern

    Args:
        A: Pointer to input matrix A of shape (M, K) - local rank's data
        B: Pointer to input matrix B of shape (K, N) - replicated across ranks
        C: Pointer to output matrix C of shape (M, N) - will contain reduced result
        aux_buffer: Symmetric-heap staging buffer for one_shot/two_shot
        locks: Pointer to versioned lock array (one int32 lock per tile)
        M: Number of rows in A and C
        N: Number of columns in B and C
        K: Number of columns in A and rows in B
        stride_am, stride_ak: Strides for A tensor
        stride_bk, stride_bn: Strides for B tensor
        stride_cm, stride_cn: Strides for C tensor
        context_tensor: Device context tensor for RMA operations
        cur_rank: Current rank
        world_size: Total number of ranks
        BLOCK_SIZE_M: Block size for M dimension
        BLOCK_SIZE_N: Block size for N dimension
        BLOCK_SIZE_K: Block size for K dimension
        GROUP_SIZE_M: Tile-swizzle group size for L2 locality
        NUM_SMS: Number of SMs (persistent grid size)
        NUM_XCDS: Number of XCDs (chiplets) for XCD-aware scheduling
        EVEN_K: Whether K is evenly divisible by BLOCK_SIZE_K
        VARIANT: All-reduce algorithm variant
        call_number: Monotonic version for versioned locks (runtime int). Producers
            signal with this value, consumers spin until they observe it, so locks
            do not need to be zeroed between calls.
    """
    # ═══════════════════════════════════════════════════════════════════════
    # Create tritonblas views, context, and scheduler for GEMM
    # ═══════════════════════════════════════════════════════════════════════
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
    dst_view = iris.make_tensor_view(C, M, N, stride_cm, stride_cn)

    # Persistent loop over output tiles using the scheduler
    start, total, stride = sched.persistent_tile_range()
    for tile_idx in range(start, total, stride):
        out_tile = sched.get_tile_from_idx(tile_idx)
        pid_m = out_tile.pid_m
        pid_n = out_tile.pid_n

        # GEMM using tritonblas stages
        acc = gemm_ctx.reduce_axis(tensorA, tensorB, out_tile)

        # Get row and column indices from tile (needed for one_shot/two_shot variants)
        rm, rn = out_tile.indices()

        # Convert to output dtype
        c = acc.to(C.type.element_ty)

        # Create tile object once for all variants
        tile_obj = iris.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)

        # Dispatch to appropriate all-reduce variant
        if VARIANT == "atomic":
            ctx.all_reduce_atomic(tile_obj, dst_view)
        elif VARIANT == "spinlock":
            ctx.all_reduce_spinlock(tile_obj, dst_view, locks)
        elif VARIANT == "one_shot" or VARIANT == "two_shot":
            # For one_shot and two_shot: store tile to aux_buffer and signal ready with lock
            # Store GEMM result to aux_buffer (avoid race condition with final output)
            temp_ptr = aux_buffer + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            tl.store(temp_ptr, c, mask=(rm[:, None] < M) & (rn[None, :] < N), cache_modifier=".wt")
            tl.debug_barrier()  # Ensures all stores are visible before the atomic_xchg

            # Signal tile is ready by publishing the current call_number.
            # Release semantics ensure prior stores are visible to remote GPUs.
            num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
            tile_id = pid_m * num_tiles_n + pid_n
            lock_ptr = locks + tile_id
            tl.atomic_xchg(lock_ptr, call_number, sem="release", scope="sys")

            # Create source view only when needed (aux_buffer is not None)
            src_view = iris.make_tensor_view(aux_buffer, M, N, stride_cm, stride_cn)

            if VARIANT == "one_shot":
                ctx.all_reduce_one_shot(tile_obj, src_view, dst_view, locks, call_number)
            elif VARIANT == "two_shot":
                ctx.all_reduce_two_shot(tile_obj, src_view, dst_view, locks, call_number)


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

    ``shmem.zeros`` is collective, so this must only run when the workspace does
    not already match the problem. Subsequent calls with matching shapes reuse
    the existing buffers: versioned locks and overwrite semantics (one_shot /
    two_shot) remove the need for re-zeroing, which in turn keeps the hot path
    free of host-side synchronization and therefore CUDAGraph-capturable.

    Args:
        shmem: Iris shmem context
        M, N, K: Problem dimensions
        dtype: Data type of A/B/C
        config: FusedConfig (already defaulted by the caller)
        workspace: Optional existing workspace to reuse. If None, creates new one.

    Returns:
        FusedWorkspace instance with buffers allocated.
    """
    world_size = shmem.get_num_ranks()

    # Validate config
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

    # Allocate locks for spinlock, one_shot, and two_shot variants
    if config.all_reduce_variant in ["spinlock", "one_shot", "two_shot"]:
        if workspace.locks is None or workspace.locks.numel() != total_tiles:
            workspace.locks = shmem.zeros((total_tiles,), dtype=torch.int32)
    else:
        workspace.locks = None

    # Allocate auxiliary buffer for one_shot and two_shot to avoid race conditions
    # (GEMM results stored here, then reduced to final output)
    if config.all_reduce_variant in ["one_shot", "two_shot"]:
        if workspace.aux_buffer is None or workspace.aux_buffer.shape != (M, N):
            workspace.aux_buffer = shmem.zeros((M, N), dtype=dtype)
    else:
        workspace.aux_buffer = None

    # 1-element tensor used for stream-level cross-rank sync (see _pre_kernel_sync).
    if workspace._barrier_tensor is None:
        workspace._barrier_tensor = torch.zeros(1, dtype=torch.int32, device=shmem.get_device())

    workspace.prepared = True
    return workspace


def _pre_kernel_sync(shmem, C: torch.Tensor, config: FusedConfig, workspace: FusedWorkspace):
    """
    Variant-specific pre-kernel preparation using GPU-stream ops only (no host sync).

    - atomic / spinlock: C accumulates in place, so it must be zeroed, and all
      ranks must agree the zeroing happened before any remote atomic lands.
      A ``dist.all_reduce`` on a 1-element tensor gives that ordering on-stream
      instead of a host-side ``shmem.barrier()``.
    - one_shot / two_shot: nothing to do. C is fully overwritten by the kernel
      and the locks are versioned by ``workspace.call_counter``.
    """
    import torch.distributed as dist

    if config.all_reduce_variant in ["atomic", "spinlock"]:
        C.zero_()
        dist.all_reduce(workspace._barrier_tensor)


def matmul_all_reduce_preamble(
    shmem,
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Allocate and reset temporary buffers for matmul_all_reduce.

    This is the explicit, host-synchronizing entry point. ``matmul_all_reduce``
    allocates lazily on its own, so calling this is optional; it is useful to
    front-load the collective allocations (and the barrier) outside a hot loop
    or a CUDAGraph capture region.

    Args:
        shmem: Iris shmem context
        C: Output tensor (M, N)
        A: Input matrix A (M, K)
        B: Input matrix B (K, N)
        config: Optional FusedConfig. If None, uses defaults.
        workspace: Optional existing workspace to reuse. If None, creates new one.

    Returns:
        FusedWorkspace instance ready for kernel launch.
    """
    if config is None:
        config = FusedConfig()

    M, K = A.shape[:2]
    N = B.shape[1]

    workspace = _allocate_workspace(shmem, M, N, K, A.dtype, config, workspace=workspace)

    # Reset state and publish it to all ranks.
    if workspace.locks is not None:
        workspace.locks.zero_()
    if workspace.aux_buffer is not None:
        workspace.aux_buffer.zero_()
    C.zero_()
    shmem.barrier()

    return workspace


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

    Once the workspace has been allocated (first call, or via
    ``matmul_all_reduce_preamble``), the steady-state path issues no host-side
    synchronization: lock-based variants (one_shot, two_shot) use versioned
    locks instead of inter-call zeroing plus a barrier, and atomic/spinlock use
    a stream-level ``dist.all_reduce`` on a 1-element tensor. This makes repeat
    calls safe to record into a CUDAGraph.

    Args:
        shmem: Iris shmem context
        C: Output tensor (M, N) - will contain reduced result on all ranks
        A: Input matrix A (M, K) - each rank has different data (data-parallel)
        B: Input matrix B (K, N) - replicated across ranks
        async_op: If False, performs a stream-level barrier at end. Default: False.
        config: Optional FusedConfig for tuning. If None, uses defaults.
        workspace: Optional pre-allocated workspace. If None, creates new one.

    Returns:
        workspace: Updated workspace object (can be reused for subsequent calls)

    Example:
        >>> A = shmem.randn((1024, 512), dtype=torch.float16)
        >>> B = shmem.randn((512, 2048), dtype=torch.float16)
        >>> C = shmem.zeros((1024, 2048), dtype=torch.float16)
        >>> shmem.ops.matmul_all_reduce(C, A, B)
    """
    import torch.distributed as dist

    if config is None:
        config = FusedConfig()

    # Extract dimensions
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

    # Validate block sizes match problem dimensions
    assert M >= config.block_size_m, f"M={M} too small for block_size_m={config.block_size_m}"
    assert K >= config.block_size_k, f"K={K} too small for block_size_k={config.block_size_k}"
    assert N >= config.block_size_n, f"N={N} too small for block_size_n={config.block_size_n}"

    # Extract strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Get rank info
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    from iris.host.logging.logging import _log_rank

    _log_rank(
        logging.DEBUG,
        "matmul_all_reduce: shape=(%d,%d,%d) dtype=%s variant=%s rank=%d/%d",
        M,
        N,
        K,
        A.dtype,
        config.all_reduce_variant,
        rank,
        world_size,
        rank=rank,
        num_ranks=world_size,
    )

    # Allocate the workspace once; reuse it on subsequent calls with the same
    # shape/dtype/variant. The allocation path is collective, so it must not run
    # in the steady state (and cannot run during CUDAGraph capture).
    needs_alloc = workspace is None or not workspace.matches(
        "matmul_all_reduce", (M, N, K), A.dtype, world_size, config.all_reduce_variant
    )

    if needs_alloc:
        workspace = matmul_all_reduce_preamble(shmem, C, A, B, config=config, workspace=workspace)

    # Validate that the pre-allocated lock array is large enough for the current tile count.
    # This can occur when the workspace was prepared with larger block sizes (fewer tiles)
    # and is then reused with smaller block sizes (more tiles). We deliberately do not
    # reallocate here: shmem.zeros is collective and cannot be called on the hot path.
    num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
    num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_pid_m * num_pid_n

    if workspace.locks is not None and workspace.locks.numel() < total_tiles:
        raise ValueError(
            f"Lock array too small: have {workspace.locks.numel()} but need {total_tiles}. "
            f"Pre-allocate workspace with the smallest block sizes you intend to use."
        )

    # Bump the lock version for this call, then do variant-specific prep (no host sync).
    workspace.call_counter += 1
    _pre_kernel_sync(shmem, C, config, workspace)

    # Get device context for RMA
    device_context = shmem.get_device_context()

    even_k = K % config.block_size_k == 0

    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(A.device)
        num_sms = props.multi_processor_count

    # Persistent grid: one CTA per SM, each looping over many output tiles.
    grid = (num_sms,)

    iris_launch(
        _fused_matmul_all_reduce_kernel,
        grid,
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
        algorithm="matmul_all_reduce",
        rank=rank,
        dtype=A.dtype,
    )

    # The workspace stays "prepared": buffers are reusable as-is, and the lock
    # version bump above is what makes the next call safe without re-zeroing.

    # Stream-level post-kernel sync unless async (no host-side synchronize, so
    # this stays CUDAGraph-capturable).
    if not async_op:
        dist.all_reduce(workspace._barrier_tensor)

    return workspace
