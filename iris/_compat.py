# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton version compatibility shims.

Centralizes API differences between Triton versions so each module
doesn't need its own inline getattr chain.

constexpr_function location by Triton version:
  - Triton 3.6.0+rocm7.2.0: triton.constexpr_function
  - Future versions:         triton.language.constexpr_function (expected)
"""

import triton
import triton.language as tl

# Resolve constexpr_function: prefer triton.language (future), fall back to
# triton top-level (Triton 3.6.0+rocm7.2.0).
_constexpr_function = getattr(
    tl, "constexpr_function", getattr(triton, "constexpr_function", None)
)

if _constexpr_function is None:
    raise ImportError(
        "Cannot find constexpr_function in triton or triton.language. "
        "Iris requires Triton >= 3.6.0. "
        f"Installed Triton version: {getattr(triton, '__version__', 'unknown')}"
    )

__all__ = ["_constexpr_function"]
