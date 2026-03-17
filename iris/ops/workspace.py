# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Workspace management for fused GEMM+CCL operations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class FusedWorkspace:
    """
    Workspace for fused GEMM+CCL operations.

    This class holds temporary buffers and metadata needed for fused operations.
    Buffers are allocated lazily and reused across calls when shapes match.

    Attributes:
        operation: Operation type ("matmul_all_reduce", "all_gather_matmul", etc.)
        shape: Problem dimensions (M, N, K)
        dtype: Data type of tensors
        world_size: Number of ranks in the communicator
        variant: Algorithm variant (for operations that support multiple variants)

        # Temporary buffers
        aux_buffer: Generic auxiliary buffer for intermediate results (gathered data, temp results, etc.)
        locks: Lock array for spinlock synchronization

        prepared: Whether workspace has been initialized for current operation
    """

    operation: str = ""
    shape: Tuple[int, int, int] = ()  # (M, N, K)
    dtype: Optional[torch.dtype] = None
    world_size: int = 1
    variant: str = ""

    # Temporary buffers (allocated as needed)
    aux_buffer: Optional[torch.Tensor] = None  # Generic buffer for intermediate results
    locks: Optional[torch.Tensor] = None  # Synchronization primitives

    # Versioned lock counter -- incremented each call so lock-based variants
    # (one_shot, two_shot) don't need lock zeroing between calls.
    call_counter: int = 0

    # Small tensor for stream-level cross-rank sync (replaces host barriers)
    _barrier_tensor: Optional[torch.Tensor] = None

    prepared: bool = False

    def matches(
        self,
        operation: str,
        shape: Tuple[int, int, int],
        dtype: torch.dtype,
        world_size: int,
        variant: str = "",
    ) -> bool:
        """
        Check if workspace can be reused for the given parameters.

        Returns True if buffers are allocated for the same problem configuration.
        Unlike before, this no longer checks ``prepared`` -- versioned locks and
        overwrite-based variants (two_shot) don't require re-preparation.
        """
        return (
            self.operation == operation
            and self.shape == shape
            and self.dtype == dtype
            and self.world_size == world_size
            and self.variant == variant
            and self.prepared
        )

    def reset(self):
        """Mark workspace as unprepared (buffers will be re-initialized next time)."""
        self.prepared = False

    def clear(self):
        """Free all allocated buffers."""
        self.aux_buffer = None
        self.locks = None
        self._barrier_tensor = None
        self.call_counter = 0
        self.prepared = False
