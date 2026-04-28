# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for Triton version compatibility shims."""

import pytest


def test_constexpr_function_resolved():
    """_constexpr_function must resolve to a callable decorator."""
    from iris._compat import _constexpr_function

    assert _constexpr_function is not None, (
        "_constexpr_function is None — constexpr_function not found in "
        "triton or triton.language"
    )
    assert callable(_constexpr_function), (
        f"_constexpr_function should be callable, got {type(_constexpr_function)}"
    )


def test_constexpr_function_importable_from_modules():
    """Each module that uses _constexpr_function should import it successfully."""
    from iris.tracing.events import _constexpr_function as events_cf
    from iris.iris import _constexpr_function as iris_cf
    from iris.x.core import _constexpr_function as core_cf

    # All should be the same object (imported from _compat)
    assert events_cf is iris_cf
    assert iris_cf is core_cf


def test_constexpr_function_is_decorator():
    """_constexpr_function should work as a decorator (basic sanity check)."""
    from iris._compat import _constexpr_function

    # Verify it can be used as a decorator without raising
    @_constexpr_function
    def dummy():
        pass
