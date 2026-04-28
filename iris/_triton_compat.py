# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton version compatibility shim.

Provides ``constexpr_function`` that works across Triton versions:
- Triton ≥3.7 (or nightly builds with gluon): ``triton.language.constexpr_function`` exists natively.
- Triton 3.6.0+rocm7.2.0: The attribute is missing.  We fall back to a decorator that
  marks the function with ``__triton_builtin__ = True``, which is the same mechanism
  used by DeviceTracing (see iris/tracing/device.py) and accepted by Triton's
  dependency finder / kernel hasher.
"""

from __future__ import annotations

import triton.language as tl


def _constexpr_function_fallback(fn):
    """Fallback decorator: mark *fn* as a Triton built-in so the JIT accepts it."""
    fn.__triton_builtin__ = True
    return fn


# Prefer the real decorator when available; otherwise use the fallback.
constexpr_function: callable  # type: ignore[assignment]
if hasattr(tl, "constexpr_function"):
    constexpr_function = tl.constexpr_function
else:
    constexpr_function = _constexpr_function_fallback
