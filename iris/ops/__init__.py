# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
iris.ops: High-level API for fused GEMM+CCL operations.

This module provides torch-like interfaces for fused matrix multiplication
and collective communication operations. All operations automatically infer
dimensions, strides, and hardware parameters from input tensors.

Usage:
    >>> import iris
    >>> shmem = iris.iris(heap_size)
    >>>
    >>> # Via shmem.ops namespace (recommended)
    >>> A = shmem.randn((M, K), dtype=torch.float16)
    >>> B = shmem.randn((K, N), dtype=torch.float16)
    >>> output = shmem.zeros((M, N), dtype=torch.float16)
    >>> shmem.ops.matmul_all_reduce(output, A, B)
    >>>
    >>> # Or standalone (requires shmem as first parameter)
    >>> import iris.ops as ops
    >>> ops.matmul_all_reduce(shmem, output, A, B)

Available operations:
    - matmul_all_reduce: GEMM + All-Reduce
    - all_gather_matmul: All-Gather + GEMM
    - matmul_all_gather: GEMM + All-Gather
    - matmul_reduce_scatter: GEMM + Reduce-Scatter
    - matmul_reduce_scatter_fast: GEMM + one-shot pull Reduce-Scatter (over M)
    - fast_reduce_scatter: standalone one-shot pull Reduce-Scatter (over M)
"""

import logging as _logging

from .config import FusedConfig
from .workspace import FusedWorkspace

# Import operations
# from .matmul import matmul  # Simple single-GPU GEMM - TODO: implement

# matmul_reduce_scatter_fast needs only torch + triton, so it is imported first
# and unguarded: it must stay usable when the optional tritonblas dependency
# below is absent.
from .matmul_reduce_scatter_fast import (
    matmul_reduce_scatter_fast,
    matmul_reduce_scatter_fast_preamble,
    fast_reduce_scatter,
)

# These ops build their schedules with tritonblas (github.com/ROCm/tritonBLAS),
# which is not on PyPI and is absent from some ROCm containers. Importing them
# unguarded made a missing optional dependency fail `import iris` outright,
# which in turn broke every downstream package that imports iris at module
# scope. Degrade to unavailable instead.
_TRITONBLAS_OPS = (
    "matmul_all_reduce",
    "matmul_all_reduce_preamble",
    "all_gather_matmul",
    "all_gather_matmul_preamble",
    "all_gather_matmul_hbm_buffer",
    "all_gather_matmul_hbm_buffer_preamble",
    "matmul_all_gather",
    "matmul_reduce_scatter",
    "matmul_reduce_scatter_preamble",
)
TRITONBLAS_AVAILABLE = False
try:
    from .matmul_all_reduce import matmul_all_reduce, matmul_all_reduce_preamble
    from .all_gather_matmul import all_gather_matmul, all_gather_matmul_preamble
    from .all_gather_matmul_hbm_buffer import (
        all_gather_matmul_hbm_buffer,
        all_gather_matmul_hbm_buffer_preamble,
    )
    from .matmul_all_gather import matmul_all_gather
    from .matmul_reduce_scatter import matmul_reduce_scatter, matmul_reduce_scatter_preamble

    TRITONBLAS_AVAILABLE = True
except ImportError as _e:
    _logging.getLogger(__name__).warning(
        "iris.ops: %s unavailable (%s). Install tritonblas from "
        "https://github.com/ROCm/tritonBLAS to enable them; "
        "matmul_reduce_scatter_fast/fast_reduce_scatter are unaffected.",
        ", ".join(_TRITONBLAS_OPS),
        _e,
    )

    def _missing_tritonblas(name):
        def _raise(*args, **kwargs):
            raise ImportError(
                f"iris.ops.{name} requires tritonblas, which is not installed. See https://github.com/ROCm/tritonBLAS"
            )

        return _raise

    for _name in _TRITONBLAS_OPS:
        globals()[_name] = _missing_tritonblas(_name)


class OpsNamespace:
    """
    Namespace for fused GEMM+CCL operations.

    This class provides a convenient namespace for accessing fused operations
    through the shmem.ops property. It holds a reference to the shmem context
    so operations can access rank information and heap bases.

    Example:
        >>> shmem = iris.iris(heap_size)
        >>> A = shmem.randn((M, K), dtype=torch.float16)
        >>> B = shmem.randn((K, N), dtype=torch.float16)
        >>> output = shmem.zeros((M, N), dtype=torch.float16)
        >>> shmem.ops.matmul_all_reduce(output, A, B)
    """

    def __init__(self, shmem):
        """
        Initialize OpsNamespace with shmem context.

        Args:
            shmem: Iris shmem context
        """
        self._shmem = shmem

    def matmul_all_reduce(self, output_tensor, A, B, bias=None, async_op=False, config=None, workspace=None):
        """
        Fused matrix multiplication and all-reduce.

        Computes: output = all_reduce(A @ B + bias)

        Args:
            output_tensor: Output tensor (M, N)
            A: Input matrix A (M, K)
            B: Input matrix B (K, N)
            bias: Optional bias vector (M,) or (N,)
            async_op: If False, performs barrier at end
            config: Optional FusedConfig for tuning
            workspace: Optional pre-allocated workspace

        Returns:
            workspace: Updated workspace object

        Example:
            >>> output = shmem.zeros((M, N), dtype=torch.float16)
            >>> shmem.ops.matmul_all_reduce(output, A, B)
        """
        return matmul_all_reduce(self._shmem, output_tensor, A, B, async_op, config, workspace)

    def all_gather_matmul(self, output_tensor, A_sharded, B, bias=None, async_op=False, config=None, workspace=None):
        """
        Fused all-gather and matrix multiplication.

        Computes: output = all_gather(A_sharded) @ B + bias

        Args:
            output_tensor: Output tensor (M, N)
            A_sharded: Sharded input matrix (M, K_local)
            B: Input matrix B (K, N) where K = K_local * world_size
            bias: Optional bias vector (M,) or (N,)
            async_op: If False, performs barrier at end
            config: Optional FusedConfig for tuning
            workspace: Optional pre-allocated workspace

        Returns:
            workspace: Updated workspace object

        Example:
            >>> K_local = K // world_size
            >>> A_sharded = shmem.randn((M, K_local), dtype=torch.float16)
            >>> output = shmem.zeros((M, N), dtype=torch.float16)
            >>> shmem.ops.all_gather_matmul(output, A_sharded, B)
        """
        return all_gather_matmul(self._shmem, output_tensor, A_sharded, B, bias, async_op, config, workspace)

    def matmul_all_gather(self, output_tensor, A, B, bias=None, async_op=False, config=None, workspace=None):
        """
        Fused matrix multiplication and all-gather.

        Computes: output = all_gather(A @ B + bias) along M dimension

        Args:
            output_tensor: Output tensor (M*world_size, N)
            A: Input matrix A (M, K)
            B: Input matrix B (K, N)
            bias: Optional bias vector (M,) or (N,)
            async_op: If False, performs barrier at end
            config: Optional FusedConfig for tuning
            workspace: Optional pre-allocated workspace

        Returns:
            workspace: Updated workspace object

        Example:
            >>> M_local = M // world_size
            >>> A = shmem.randn((M_local, K), dtype=torch.float16)
            >>> output = shmem.zeros((M, N), dtype=torch.float16)
            >>> shmem.ops.matmul_all_gather(output, A, B)
        """
        return matmul_all_gather(self._shmem, output_tensor, A, B, bias, async_op, config, workspace)

    def matmul_reduce_scatter(self, output_tensor, A, B, bias=None, async_op=False, config=None, workspace=None):
        """
        Fused matrix multiplication and reduce-scatter.

        Computes: output = reduce_scatter(A @ B + bias) along N dimension

        Args:
            output_tensor: Output tensor (M, N_local) where N_local = N / world_size
            A: Input matrix A (M, K)
            B: Input matrix B (K, N)
            bias: Optional bias vector (M,) or (N,)
            async_op: If False, performs barrier at end
            config: Optional FusedConfig for tuning
            workspace: Optional pre-allocated workspace

        Returns:
            workspace: Updated workspace object

        Example:
            >>> N_local = N // world_size
            >>> output = shmem.zeros((M, N_local), dtype=torch.float16)
            >>> shmem.ops.matmul_reduce_scatter(output, A, B)
        """
        return matmul_reduce_scatter(self._shmem, output_tensor, A, B, bias, async_op, config, workspace)

    def matmul_reduce_scatter_fast(self, output_tensor, A, B, **kwargs):
        """
        Fast GEMM + ReduceScatter: hipBLASLt GEMM + one-shot pull RS, over the M dimension.

        Supply ``staging_buffer=`` or ``workspace=`` to keep the hot path allocation-free
        and CUDAGraph-capturable; otherwise the first call allocates on the symmetric heap
        (collective, not capture-safe) and warns.

        Args:
            output_tensor: Output (M_local, N) -- this rank's reduced shard
            A: Input matrix (M, K_local) -- this rank's K-shard; ordinary torch tensor is fine
            B: Input matrix (K_local, N)
            **kwargs: staging_buffer, workspace, sync, async_op, block_m, block_n,
                num_sms, num_warps

        Returns:
            workspace: FusedWorkspace to pass back on subsequent calls

        Example:
            >>> output = torch.zeros(M_local, N, dtype=torch.float16, device="cuda")
            >>> ws = shmem.ops.matmul_reduce_scatter_fast_preamble(output, A, B)
            >>> shmem.ops.matmul_reduce_scatter_fast(output, A, B, workspace=ws)
        """
        return matmul_reduce_scatter_fast(self._shmem, output_tensor, A, B, **kwargs)

    def matmul_reduce_scatter_fast_preamble(self, output_tensor, A, B, **kwargs):
        """
        Allocate/bind the staging buffer for ``matmul_reduce_scatter_fast``.

        Collective and host-synchronizing. Call once, outside the hot loop and outside
        any graph capture region.

        Args:
            output_tensor: Output (M_local, N)
            A: Input matrix (M, K_local)
            B: Input matrix (K_local, N)
            **kwargs: staging_buffer, workspace

        Returns:
            workspace: FusedWorkspace ready for the hot path
        """
        return matmul_reduce_scatter_fast_preamble(self._shmem, output_tensor, A, B, **kwargs)

    def fast_reduce_scatter(self, output_tensor, input_tensor, **kwargs):
        """
        Fast standalone reduce-scatter over M via one-shot pull.

        ``input_tensor`` must be in the symmetric heap; ``output_tensor`` need not be.
        Each rank reads all peers' partials via iris.load, reduces in fp32, and stores
        its shard. Capture-safe, and does no cross-rank synchronization -- the caller
        owns the ordering.

        Args:
            output_tensor: Output (M_local, N)
            input_tensor: Input (M, N) -- in symmetric heap
            **kwargs: block_m, block_n, num_sms, num_warps (auto-selected if omitted)

        Example:
            >>> C_partial = shmem.zeros((M, N), dtype=torch.float16)
            >>> torch.mm(A, B, out=C_partial)
            >>> output = torch.zeros(M_local, N, dtype=torch.float16, device="cuda")
            >>> shmem.ops.fast_reduce_scatter(output, C_partial)
        """
        return fast_reduce_scatter(self._shmem, output_tensor, input_tensor, **kwargs)


# Export public API
__all__ = [
    # Configuration
    "FusedConfig",
    "FusedWorkspace",
    # Namespace
    "OpsNamespace",
    # Operations
    "matmul",  # Simple single-GPU GEMM
    "matmul_all_reduce",
    "matmul_all_reduce_preamble",
    "all_gather_matmul",
    "all_gather_matmul_preamble",
    "all_gather_matmul_hbm_buffer",
    "all_gather_matmul_hbm_buffer_preamble",
    "matmul_all_gather",
    "matmul_reduce_scatter",
    "matmul_reduce_scatter_preamble",
    "matmul_reduce_scatter_fast",
    "matmul_reduce_scatter_fast_preamble",
    "fast_reduce_scatter",
]
