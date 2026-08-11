# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fast GEMM + ReduceScatter: hipBLASLt GEMM + persistent one-shot pull RS.

The GEMM writes a full (M, N) partial product into a symmetric-heap staging
buffer; a persistent Triton kernel then has each rank pull the M-shard it owns
from every peer, reduce in fp32, and store to a local output.

Reported at 1.23-2.06x faster than ``torch.mm`` + RCCL RS on GPT-OSS-120B MoE
shapes. Those numbers come from an *unbarriered* measurement (see
``matmul_reduce_scatter_fast`` for why that is not a correct configuration);
budget roughly 0.07 ms/call for each cross-rank sync you add back.

Usage (steady state, e.g. inside a serving loop or a HIP graph):
    >>> shmem = iris.iris(heap_size)
    >>> output = torch.zeros(M_local, N, dtype=dtype, device=device)
    >>> # Allocate once, outside the hot loop / outside graph capture:
    >>> ws = matmul_reduce_scatter_fast_preamble(shmem, output, A, B)
    >>> # Hot path: no allocation, no host sync.
    >>> matmul_reduce_scatter_fast(shmem, output, A, B, workspace=ws)
"""

import logging
from typing import Optional
import torch
import triton
import triton.language as tl
import iris

from .workspace import FusedWorkspace


logger = logging.getLogger(__name__)


# Per-TP configs. NOTE: these constants are *unverified*. They shipped with the
# kernel but disagree with the tuning sweep that accompanied it (e.g. TP2:
# packaged block_m=128/num_sms=196 vs sweep block_m=256/num_sms=304), and no test
# or benchmark in-tree exercises them. Treat them as a starting point and re-tune
# per shape before trusting the performance numbers in this module's docstrings.
_AUTO_CONFIG = {
    2: dict(block_m=128, block_n=64, num_sms=196, num_warps=4),
    4: dict(block_m=64, block_n=64, num_sms=32, num_warps=4),
    8: dict(block_m=32, block_n=64, num_sms=32, num_warps=4),
}

_DEFAULT_CONFIG = dict(block_m=64, block_n=64, num_sms=64, num_warps=4)


def _get_config(world_size: int, M_local: int) -> dict:
    """
    Pick RS tile/grid parameters for ``world_size``, shrinking block_m to fit M_local.

    The halving loop is a pure performance heuristic: the kernel masks the M tail,
    so an oversized block_m is correct, just wasteful.
    """
    cfg = _AUTO_CONFIG.get(world_size, _DEFAULT_CONFIG).copy()
    while cfg["block_m"] > M_local and cfg["block_m"] > 4:
        cfg["block_m"] //= 2
    return cfg


@triton.jit
def _fast_reduce_scatter_kernel(
    input_ptr,
    output_ptr,
    N,
    M_local,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    M_ALIGNED: tl.constexpr,
    N_ALIGNED: tl.constexpr,
):
    """
    Persistent one-shot pull RS kernel.

    Stateless by construction: no locks, no flags, no atomics, no device barriers.
    Each CTA walks a strided slice of this rank's output tiles, and for each tile
    pulls the corresponding input tile from every peer via ``iris.load`` (starting at
    a ``pid``-dependent rank so pull traffic is spread across links) and accumulates
    in fp32. Only ``input_ptr`` must live in the symmetric heap.

    ``M_ALIGNED`` / ``N_ALIGNED`` are host-computed constexprs: True when ``M_local`` /
    ``N`` is an exact multiple of ``BLOCK_SIZE_M`` / ``BLOCK_SIZE_N``. Both index arrays
    are genuine contiguous aligned runs, so the contiguity hints are always legal; the
    flags only select whether the corresponding mask term can be dropped. Out-of-range
    lanes are masked on both the loads and the store, never wrapped -- a ``% N`` wrap
    would make the hint on ``iris.load`` a lie for the tail tile, since ``__translate``
    re-applies it to the translated pointer.
    """
    pid = tl.program_id(0)
    acc_dtype = tl.float32
    num_m_tiles = tl.cdiv(M_local, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles
    # First global row owned by this rank. This is not expressible as a whole number
    # of BLOCK_SIZE_M tiles when M_local is ragged, so the base is kept in rows.
    row_base = cur_rank * M_local

    for tile_id in range(pid, total_tiles, NUM_SMS):
        local_pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles

        out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        rm = row_base + out_rm

        base_ptr = input_ptr + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        out_ptrs = output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        start_rank = pid % world_size

        m_full = M_ALIGNED or (local_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M_local)
        if N_ALIGNED and m_full:
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            tl.store(out_ptrs, acc.to(output_ptr.type.element_ty))
        else:
            mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
            acc = iris.load(
                base_ptr, cur_rank, start_rank, heap_bases, mask=mask, other=0.0, hint=(1, BLOCK_SIZE_N)
            ).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases, mask=mask, other=0.0, hint=(1, BLOCK_SIZE_N)).to(
                    acc_dtype
                )
            tl.store(out_ptrs, acc.to(output_ptr.type.element_ty), mask=mask)


def fast_reduce_scatter(
    ctx,
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_sms: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> None:
    """
    Fast one-shot pull reduce-scatter over the M (row) dimension using ``iris.load``.

    Reported at 1.25-1.47x faster than RCCL RS at GPT-OSS-120B message sizes.

    This entry point is capture-safe on its own: the host side only reads shapes and
    strides, looks up a config dict, and launches. It performs no allocation and no
    host-side synchronization. It also performs **no cross-rank synchronization** --
    the caller must guarantee that every peer has finished writing ``input_tensor``
    before this launches, and that no peer overwrites its ``input_tensor`` until every
    peer's kernel has finished reading it.

    Args:
        ctx: Iris context. Only ``input_tensor`` must live in the symmetric heap;
            ``output_tensor`` may be an ordinary torch tensor.
        output_tensor: Output tensor (M_local, N) -- this rank's reduced shard.
        input_tensor: Input tensor (M, N) -- full partial sum, in the symmetric heap.
        block_m: Tile M dimension (auto-selected if None).
        block_n: Tile N dimension (auto-selected if None).
        num_sms: Number of persistent WGs (auto-selected if None).
        num_warps: Warps per WG (auto-selected if None).
    """
    M, N = input_tensor.shape
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M_local = M // world_size

    assert M % world_size == 0, f"M={M} must be divisible by world_size={world_size}"
    assert output_tensor.shape == (M_local, N), (
        f"expected output shape {(M_local, N)}, got {tuple(output_tensor.shape)}"
    )

    # Look the config up only if something is actually missing. `_get_config` does a
    # dict copy plus a halving loop, and a caller that has precomputed all four values
    # (e.g. aiter's fused wrapper, which resolves them once at cache-fill time) would
    # otherwise pay for it on every launch and get nothing back.
    if block_m and block_n and num_sms and num_warps:
        bm, bn, sms, warps = block_m, block_n, num_sms, num_warps
    else:
        cfg = _get_config(world_size, M_local)
        bm = block_m or cfg["block_m"]
        bn = block_n or cfg["block_n"]
        sms = num_sms or cfg["num_sms"]
        warps = num_warps or cfg["num_warps"]

    # Ragged M_local is handled by masking in the kernel; these flags only select
    # whether the per-axis contiguity hints are legal.
    m_aligned = (M_local % bm) == 0
    n_aligned = (N % bn) == 0

    heap_bases = ctx.get_heap_bases()

    _fast_reduce_scatter_kernel[(sms,)](
        input_tensor,
        output_tensor,
        N,
        M_local,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        heap_bases,
        rank,
        world_size,
        bm,
        bn,
        sms,
        m_aligned,
        n_aligned,
        num_warps=warps,
    )


def _validate(ctx, output_tensor: torch.Tensor, A: torch.Tensor, B: torch.Tensor):
    """Shape/dtype validation shared by the preamble and the main entry point."""
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2D tensors, got shapes {tuple(A.shape)} and {tuple(B.shape)}")

    M, K_local = A.shape
    K_B, N = B.shape
    if K_local != K_B:
        raise ValueError(
            f"Incompatible matrix dimensions: A is ({M}, {K_local}), B is ({K_B}, {N}). "
            f"B is laid out (K_local, N), not (N, K_local)."
        )

    world_size = ctx.get_num_ranks()
    if M % world_size != 0:
        raise ValueError(f"M={M} must be divisible by world_size={world_size}")
    M_local = M // world_size

    if tuple(output_tensor.shape) != (M_local, N):
        raise ValueError(f"Output tensor shape {tuple(output_tensor.shape)} doesn't match expected {(M_local, N)}")
    if A.dtype != B.dtype:
        raise ValueError(f"A and B must have the same dtype, got A:{A.dtype}, B:{B.dtype}")

    return M, N, K_local, world_size


def _allocate_workspace(
    ctx,
    M: int,
    N: int,
    K_local: int,
    dtype: torch.dtype,
    world_size: int,
    staging_buffer: Optional[torch.Tensor] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> tuple[FusedWorkspace, bool]:
    """
    Bind (or allocate) the (M, N) symmetric-heap staging buffer and the sync tensor.

    Returns ``(workspace, allocated)``, where ``allocated`` is True only if a collective
    heap allocation actually happened and therefore a barrier is required.

    ``ctx.zeros`` is collective -- it ends in ``refresh_peer_access()``, which does a
    host ``dist.barrier()`` plus a heap-base all-gather -- and Iris never frees, so this
    must only ever run off the hot path: never during graph capture, never per call.
    Everything lands in ``FusedWorkspace``, reusing the same slots (``aux_buffer``,
    ``_barrier_tensor``) that ``matmul_all_reduce`` uses. The RS case needs strictly
    less than the AR case: no locks, no versioned call counter.

    Args:
        ctx: Iris context.
        M, N, K_local: Problem dimensions.
        dtype: Data type of A/B and of the staging buffer.
        world_size: Number of ranks.
        staging_buffer: Optional caller-owned (M, N) symmetric-heap buffer. When given,
            nothing is allocated.
        workspace: Optional existing workspace to reuse/refresh.

    Returns:
        A prepared FusedWorkspace.
    """
    if workspace is None:
        workspace = FusedWorkspace()

    workspace.operation = "matmul_reduce_scatter_fast"
    workspace.shape = (M, N, K_local)
    workspace.dtype = dtype
    workspace.world_size = world_size
    workspace.variant = ""
    # `locks` is deliberately left alone: this op does not use it, and clearing it
    # would strand an all-reduce lock buffer if the caller recycles a workspace
    # (Iris never frees, so the dropped reference would leak).

    allocated = False
    if staging_buffer is not None:
        _validate_staging_buffer(ctx, staging_buffer, M, N, dtype)
        workspace.aux_buffer = staging_buffer
    elif (
        workspace.aux_buffer is None
        or tuple(workspace.aux_buffer.shape) != (M, N)
        or workspace.aux_buffer.dtype != dtype
    ):
        logger.warning(
            "matmul_reduce_scatter_fast: allocating a %s (%d, %d) staging buffer on the symmetric heap. "
            "This is COLLECTIVE (host barrier + heap-base all-gather) and Iris never frees, so it is NOT "
            "safe inside a HIP/CUDA graph capture and it leaks if it runs per call. Supply staging_buffer=, "
            "or hoist the allocation with matmul_reduce_scatter_fast_preamble().",
            dtype,
            M,
            N,
        )
        workspace.aux_buffer = ctx.zeros((M, N), dtype=dtype)
        allocated = True

    # 1-element tensor used for stream-level cross-rank sync (see _cross_rank_sync).
    if workspace._barrier_tensor is None:
        workspace._barrier_tensor = torch.zeros(1, dtype=torch.int32, device=ctx.get_device())

    workspace.prepared = True
    return workspace, allocated


def _validate_staging_buffer(ctx, staging_buffer: torch.Tensor, M: int, N: int, dtype) -> None:
    """
    Reject a staging buffer that the RS kernel cannot legally read from peers.

    The kernel translates addresses as ``peer_base + (ptr - my_base)``, so an off-heap
    pointer produces a meaningless address on every peer -- silently wrong numbers with
    no fault. Shape and dtype must match exactly too: ``torch.mm(..., out=)`` would
    otherwise *resize* the out tensor, replacing the heap buffer with a fresh off-heap
    allocation on the hot path.
    """
    if tuple(staging_buffer.shape) != (M, N):
        raise ValueError(f"staging_buffer shape {tuple(staging_buffer.shape)} doesn't match expected {(M, N)}")
    if staging_buffer.dtype != dtype:
        raise ValueError(f"staging_buffer dtype {staging_buffer.dtype} doesn't match A/B dtype {dtype}")
    heap = getattr(ctx, "heap", None)
    if heap is not None and hasattr(heap, "is_symmetric") and not heap.is_symmetric(staging_buffer):
        raise ValueError(
            "staging_buffer is not on the Iris symmetric heap. The reduce-scatter kernel "
            "reads it from every peer by offset, so an off-heap buffer yields garbage on "
            "all ranks without raising. Allocate it with ctx.zeros(...)/ctx.empty(...)."
        )


def _cross_rank_sync(workspace: FusedWorkspace) -> None:
    """
    Stream-level cross-rank sync: ``dist.all_reduce`` on a preallocated 1-element tensor.

    Same trick as ``matmul_all_reduce._pre_kernel_sync``. Unlike ``ctx.barrier()`` it
    does not synchronize the host, so it can be recorded into a graph; it only orders
    this rank's stream against every peer's stream.
    """
    import torch.distributed as dist

    dist.all_reduce(workspace._barrier_tensor)


def matmul_reduce_scatter_fast_preamble(
    ctx,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    staging_buffer: Optional[torch.Tensor] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Allocate/bind the staging buffer for ``matmul_reduce_scatter_fast``.

    This is the explicit, host-synchronizing entry point. Call it once, outside the hot
    loop and outside any graph capture region, then pass the returned workspace to every
    subsequent ``matmul_reduce_scatter_fast`` call.

    Args:
        ctx: Iris context.
        output_tensor: Output tensor (M_local, N).
        A: Input matrix (M, K_local) -- this rank's K-shard. An ordinary torch tensor is
            fine; only the staging buffer must be on the symmetric heap.
        B: Input matrix (K_local, N). An ordinary torch tensor is fine.
        staging_buffer: Optional caller-owned (M, N) symmetric-heap buffer. If supplied,
            nothing is allocated.
        workspace: Optional existing workspace to reuse.

    Returns:
        A prepared FusedWorkspace, ready for the hot path.
    """
    M, N, K_local, world_size = _validate(ctx, output_tensor, A, B)
    workspace, allocated = _allocate_workspace(
        ctx, M, N, K_local, A.dtype, world_size, staging_buffer=staging_buffer, workspace=workspace
    )
    # ctx.barrier() is a host-wide sync (torch.cuda.synchronize + dist.barrier). It is
    # only required to order the collective heap allocation above; when the caller
    # supplied the buffer, nothing collective happened and the barrier would be a
    # gratuitous host sync on what may be the hot path.
    if allocated:
        ctx.barrier()
    return workspace


def matmul_reduce_scatter_fast(
    ctx,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    staging_buffer: Optional[torch.Tensor] = None,
    workspace: Optional[FusedWorkspace] = None,
    sync: bool = True,
    async_op: bool = False,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_sms: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> FusedWorkspace:
    """
    Fast GEMM + ReduceScatter: hipBLASLt GEMM + one-shot pull RS.

    Computes ``output = reduce_scatter(A @ B)`` over the M dimension, each rank keeping
    ``M_local = M // world_size`` rows. The GEMM runs via ``torch.mm`` into an (M, N)
    symmetric-heap staging buffer; a persistent Triton kernel then pulls and reduces.

    Capture safety
    --------------
    Once the workspace exists (first call, or via ``matmul_reduce_scatter_fast_preamble``,
    or by passing ``staging_buffer=``), the steady-state path performs no allocation and
    no host-side synchronization, so it can be recorded into a HIP/CUDA graph. If no
    buffer is available it falls back to a collective ``ctx.zeros`` and logs a warning:
    that fallback is for eager/standalone use only. It is not capture-safe, and because
    Iris never frees, calling it per iteration leaks ``M*N*itemsize`` every time and
    moves the buffer, invalidating any previously captured pointer.

    Cross-rank ordering
    -------------------
    Same-stream ordering only guarantees that *this* rank's GEMM precedes *this* rank's
    RS; it says nothing about peers. Unsynchronized, the RS can read a peer's staging
    buffer before that peer's GEMM has written it, and -- with a reused buffer -- a
    peer's next GEMM can overwrite data this rank is still reading. Both guards are
    stream-level ``dist.all_reduce`` calls on a preallocated 1-element tensor, so they
    stay capture-safe:

    - ``sync=True`` (default): RAW guard between the GEMM and the RS launch.
    - ``async_op=False`` (default): WAR guard after the RS, so the staging buffer is
      safe to reuse on the next call.

    Each guard costs roughly 0.07 ms/call. ``sync=False, async_op=True`` is the
    unbarriered configuration in which the headline 1.23-2.06x speedups were measured;
    that path was never correctness-checked and is appropriate only for measuring the
    barrier delta, or when the caller supplies its own cross-rank ordering. The default
    is correct, not fastest.

    Args:
        ctx: Iris context. Only the staging buffer must live in the symmetric heap;
            A, B and output_tensor may be ordinary torch tensors.
        output_tensor: Output (M_local, N) -- this rank's reduced shard.
        A: Input matrix (M, K_local) -- this rank's K-shard.
        B: Input matrix (K_local, N). Note the (K, N) layout, not (N, K).
        staging_buffer: Optional caller-owned (M, N) symmetric-heap buffer. Supplying it
            avoids the allocation, but the *first* call still runs the preamble (which
            ends in a host ``ctx.barrier()``), so the returned workspace must be fed back
            in for the hot path to be host-sync-free and capture-safe.
        workspace: Optional workspace from a previous call or from the preamble.
        sync: Cross-rank sync between GEMM and RS. Default True (correct).
        async_op: If False (default), cross-rank sync after the RS.
        block_m: RS tile M (auto-selected if None).
        block_n: RS tile N (auto-selected if None).
        num_sms: RS persistent WGs (auto-selected if None).
        num_warps: RS warps per WG (auto-selected if None).

    Returns:
        The workspace, to be passed back in on subsequent calls.

    Example:
        >>> shmem = iris.iris(1 << 33)
        >>> A = torch.randn(M, K_local, dtype=torch.float16, device="cuda")
        >>> B = torch.randn(K_local, N, dtype=torch.float16, device="cuda")
        >>> output = torch.zeros(M_local, N, dtype=torch.float16, device="cuda")
        >>> ws = matmul_reduce_scatter_fast_preamble(shmem, output, A, B)
        >>> matmul_reduce_scatter_fast(shmem, output, A, B, workspace=ws)
    """
    M, N, K_local, world_size = _validate(ctx, output_tensor, A, B)

    needs_alloc = workspace is None or not workspace.matches(
        "matmul_reduce_scatter_fast", (M, N, K_local), A.dtype, world_size, ""
    )
    if needs_alloc:
        workspace = matmul_reduce_scatter_fast_preamble(
            ctx, output_tensor, A, B, staging_buffer=staging_buffer, workspace=workspace
        )
    elif staging_buffer is not None and staging_buffer.data_ptr() != workspace.aux_buffer.data_ptr():
        # `workspace.matches` only compares the workspace's own recorded metadata; it
        # never inspects staging_buffer, so this rebind must validate independently.
        _validate_staging_buffer(ctx, staging_buffer, M, N, A.dtype)
        workspace.aux_buffer = staging_buffer

    # GEMM: write this rank's full (M, N) partial into the symmetric-heap buffer.
    torch.mm(A, B, out=workspace.aux_buffer)

    # RAW: every peer's GEMM must land before anyone pulls.
    if sync:
        _cross_rank_sync(workspace)

    fast_reduce_scatter(
        ctx,
        output_tensor,
        workspace.aux_buffer,
        block_m=block_m,
        block_n=block_n,
        num_sms=num_sms,
        num_warps=num_warps,
    )

    # WAR: every peer's pull must finish before the buffer is reused on the next call.
    if not async_op:
        _cross_rank_sync(workspace)

    return workspace
