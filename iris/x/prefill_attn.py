# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Prefill Attention primitives for Iris.

This module implements device functions and kernels for prefill-phase (multi-token query)
attention.

Phases implemented:
  Phase 1 — flash_prefill_step, flash_prefill_kernel   (single GPU, non-paged)
  Phase 2 — load_kv_tile_paged, paged_prefill_attn_kernel  (single GPU, paged)
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# ---------------------------------------------------------------------------
# Phase 1 — Reference single-GPU prefill attention (non-paged)
# ---------------------------------------------------------------------------


@triton.jit
def flash_prefill_step(
    q_tile,  # [BLOCK_Q, HEAD_DIM] fp16
    k_tile,  # [BLOCK_K, HEAD_DIM] fp16
    v_tile,  # [BLOCK_K, HEAD_DIM] fp16
    acc,  # [BLOCK_Q, HEAD_DIM] fp32 accumulated output
    e_max,  # [BLOCK_Q] fp32 running max logit
    e_sum,  # [BLOCK_Q] fp32 running denominator
    q_offset,  # int: global row index of q_tile[0] (for causal mask)
    kv_offset,  # int: global row index of k_tile[0] (for causal mask)
    scale,  # float: attention scale (1/sqrt(d))
    causal: tl.constexpr,  # bool constexpr: apply causal mask?
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Online-softmax flash attention inner-loop step.

    Computes QK^T for one (Q-tile, K-tile) pair, optionally applies causal mask,
    updates online softmax state (e_max, e_sum, acc), and returns the new state.

    Returns:
        (acc, e_max, e_sum): updated online softmax state
    """
    # QK^T: [BLOCK_Q, HEAD_DIM] x [HEAD_DIM, BLOCK_K] -> [BLOCK_Q, BLOCK_K]
    qk = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32)
    qk = qk * scale

    if causal:
        # Position j (key) can be attended by position i (query) only if j <= i
        q_idx = q_offset + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
        k_idx = kv_offset + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        causal_mask = q_idx[:, None] >= k_idx[None, :]  # [BLOCK_Q, BLOCK_K]
        qk = tl.where(causal_mask, qk, float("-inf"))

    # Online softmax update
    row_max = tl.max(qk, axis=1)  # [BLOCK_Q]
    n_e_max = tl.maximum(e_max, row_max)  # [BLOCK_Q]
    alpha = libdevice.fast_expf(e_max - n_e_max)  # [BLOCK_Q] rescale factor
    p = libdevice.fast_expf(qk - n_e_max[:, None])  # [BLOCK_Q, BLOCK_K] softmax numerator

    # Weighted value accumulation
    acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile).to(tl.float32)
    e_sum = e_sum * alpha + tl.sum(p, axis=1)
    e_max = n_e_max

    return acc, e_max, e_sum


@triton.jit
def flash_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    S,
    H,
    H_kv,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_os,
    stride_oh,
    stride_od,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    H_PER_KV: tl.constexpr,  # H // H_kv (GQA ratio)
    HEAD_DIM: tl.constexpr,
):
    """
    Single-GPU non-paged causal flash prefill attention kernel.

    Grid: (cdiv(S, BLOCK_Q), H)
    Each program handles one query tile × one query head.
    GQA: multiple query heads share one KV head (H_PER_KV query heads per KV head).
    """
    pid_q = tl.program_id(0)  # which BLOCK_Q slice of the sequence
    pid_h = tl.program_id(1)  # which query head

    # GQA: map query head to its KV head
    kv_h = pid_h // H_PER_KV

    # Query tile indices
    q_rows = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    d_cols = tl.arange(0, HEAD_DIM)  # [HEAD_DIM]
    q_mask = (q_rows < S)[:, None]  # [BLOCK_Q, 1]

    # Load Q tile: [BLOCK_Q, HEAD_DIM]
    q_off = q_rows[:, None] * stride_qs + pid_h * stride_qh + d_cols[None, :] * stride_qd
    q_tile = tl.load(q_ptr + q_off, mask=q_mask, other=0.0).to(tl.float16)

    # Online softmax state
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    e_max = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

    # Causal: only attend to positions <= current Q tile's last position
    # so num_k_tiles = ceil((pid_q+1)*BLOCK_Q / BLOCK_K)
    num_k_tiles = tl.cdiv((pid_q + 1) * BLOCK_Q, BLOCK_K)

    for kv_block in range(0, num_k_tiles):
        kv_start = kv_block * BLOCK_K
        kv_rows = kv_start + tl.arange(0, BLOCK_K)
        kv_mask = (kv_rows < S)[:, None]  # [BLOCK_K, 1]

        # Load K tile: [BLOCK_K, HEAD_DIM]
        k_off = kv_rows[:, None] * stride_ks + kv_h * stride_kh + d_cols[None, :] * stride_kd
        k_tile = tl.load(k_ptr + k_off, mask=kv_mask, other=0.0).to(tl.float16)

        # Load V tile: [BLOCK_K, HEAD_DIM]
        v_off = kv_rows[:, None] * stride_vs + kv_h * stride_vh + d_cols[None, :] * stride_vd
        v_tile = tl.load(v_ptr + v_off, mask=kv_mask, other=0.0).to(tl.float16)

        q_offset = pid_q * BLOCK_Q
        acc, e_max, e_sum = flash_prefill_step(
            q_tile,
            k_tile,
            v_tile,
            acc,
            e_max,
            e_sum,
            q_offset,
            kv_start,
            scale,
            causal=True,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            HEAD_DIM=HEAD_DIM,
        )

    # Normalize and write output
    denom = tl.where(e_sum[:, None] == 0.0, 1.0, e_sum[:, None])
    out = (acc / denom).to(tl.float16)
    out_off = q_rows[:, None] * stride_os + pid_h * stride_oh + d_cols[None, :] * stride_od
    tl.store(out_ptr + out_off, out, mask=q_mask)


def reference_prefill_attn(q, k, v, scale=None, causal=True):
    """
    Host wrapper for single-GPU non-paged flash prefill attention.

    Args:
        q: [S, H, D] query tensor (fp16, contiguous)
        k: [S, H_kv, D] key tensor (fp16, contiguous)
        v: [S, H_kv, D] value tensor (fp16, contiguous)
        scale: attention scale (default: D^{-0.5})
        causal: apply causal mask (default True; non-causal not tested)

    Returns:
        out: [S, H, D] attention output (fp16)
    """
    S, H, D = q.shape
    H_kv = k.shape[1]
    assert H % H_kv == 0, f"H={H} must be divisible by H_kv={H_kv}"
    H_PER_KV = H // H_kv

    if scale is None:
        scale = D**-0.5

    # Tile sizes
    BLOCK_Q = min(64, triton.next_power_of_2(S))
    BLOCK_K = min(64, triton.next_power_of_2(S))
    HEAD_DIM = triton.next_power_of_2(D)

    # Pad head dim to power of 2
    if HEAD_DIM != D:
        q = torch.nn.functional.pad(q, (0, HEAD_DIM - D))
        k = torch.nn.functional.pad(k, (0, HEAD_DIM - D))
        v = torch.nn.functional.pad(v, (0, HEAD_DIM - D))

    out = torch.empty(S, H, HEAD_DIM, dtype=torch.float16, device=q.device)

    grid = (triton.cdiv(S, BLOCK_Q), H)
    flash_prefill_kernel[grid](
        q,
        k,
        v,
        out,
        S,
        H,
        H_kv,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        scale,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        H_PER_KV=H_PER_KV,
        HEAD_DIM=HEAD_DIM,
    )

    if HEAD_DIM != D:
        out = out[:, :, :D]
    return out.contiguous()


# ---------------------------------------------------------------------------
# Phase 2 — Paged prefill attention (single GPU)
# ---------------------------------------------------------------------------


@triton.jit
def paged_prefill_attn_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    out_ptr,
    # Q strides: q shape [S, H, D]
    stride_qs,
    stride_qh,
    stride_qd,
    # KV cache strides: cache shape [num_phys_blocks, PAGE_SIZE, H_kv, D]
    stride_cb,
    stride_cs,
    stride_ch,
    stride_cd,
    # block_table strides: [batch, max_blocks]
    stride_tb,
    stride_tn,
    # output strides: [S, H, D]
    stride_os,
    stride_oh,
    stride_od,
    # metadata
    S,
    H,
    H_kv,
    scale,
    H_PER_KV: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Single-GPU paged prefill attention kernel.

    KV cache has shape [num_phys_blocks, PAGE_SIZE, H_kv, D].
    block_table has shape [batch, max_blocks]: block_table[b, i] = physical block index
    for logical KV block i of sequence b.

    Grid: (batch, cdiv(S, BLOCK_Q), H)
    """
    bid = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_h = tl.program_id(2)

    kv_h = pid_h // H_PER_KV

    kv_len = tl.load(seq_lens_ptr + bid)

    # Query tile
    q_rows = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_cols = tl.arange(0, HEAD_DIM)
    q_mask = (q_rows < S)[:, None]

    q_off = q_rows[:, None] * stride_qs + pid_h * stride_qh + d_cols[None, :] * stride_qd
    q_tile = tl.load(q_ptr + q_off, mask=q_mask, other=0.0).to(tl.float16)

    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    e_max = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

    # Causal: only attend up to q_tile's last position or kv_len, whichever is smaller
    max_kv = tl.minimum(kv_len, (pid_q + 1) * BLOCK_Q)
    num_k_tiles = tl.cdiv(max_kv, BLOCK_K)

    for kv_block in range(0, num_k_tiles):
        kv_start = kv_block * BLOCK_K
        kv_rows = kv_start + tl.arange(0, BLOCK_K)

        # Block table lookup: physical block for logical block (kv_rows // PAGE_SIZE)
        phys_block = tl.load(
            block_table_ptr + bid * stride_tb + (kv_rows // PAGE_SIZE) * stride_tn,
        )  # [BLOCK_K]
        slot = kv_rows % PAGE_SIZE  # [BLOCK_K]

        # Physical KV cache offset for this KV head
        # cache[phys_block, slot, kv_h, d] = cache_ptr + phys_block*stride_cb + slot*stride_cs + kv_h*stride_ch + d*stride_cd
        k_off = (
            phys_block[:, None] * stride_cb + slot[:, None] * stride_cs + kv_h * stride_ch + d_cols[None, :] * stride_cd
        )  # [BLOCK_K, HEAD_DIM]

        k_tile = tl.load(k_cache_ptr + k_off).to(tl.float16)
        v_tile = tl.load(v_cache_ptr + k_off).to(tl.float16)  # v_cache has same layout

        q_offset = pid_q * BLOCK_Q
        acc, e_max, e_sum = flash_prefill_step(
            q_tile,
            k_tile,
            v_tile,
            acc,
            e_max,
            e_sum,
            q_offset,
            kv_start,
            scale,
            causal=True,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            HEAD_DIM=HEAD_DIM,
        )

    denom = tl.where(e_sum[:, None] == 0.0, 1.0, e_sum[:, None])
    out = (acc / denom).to(tl.float16)
    out_off = q_rows[:, None] * stride_os + pid_h * stride_oh + d_cols[None, :] * stride_od
    tl.store(out_ptr + out_off, out, mask=q_mask)


def paged_prefill_attn(q, k_cache, v_cache, block_table, seq_lens, scale=None):
    """
    Host wrapper for single-GPU paged prefill attention.

    Args:
        q: [S, H, D] query tensor (fp16)
        k_cache: [num_blocks, PAGE_SIZE, H_kv, D] key cache (fp16)
        v_cache: [num_blocks, PAGE_SIZE, H_kv, D] value cache (fp16)
        block_table: [batch, max_blocks] int32 physical block indices
        seq_lens: [batch] int32 actual KV sequence lengths
        scale: attention scale (default 1/sqrt(D))

    Returns:
        out: [S, H, D] output tensor (fp16)
    """
    S, H, D = q.shape
    _, PAGE_SIZE, H_kv, _ = k_cache.shape
    assert H % H_kv == 0
    H_PER_KV = H // H_kv
    batch = block_table.shape[0]

    if scale is None:
        scale = D**-0.5

    BLOCK_Q = min(64, triton.next_power_of_2(S))
    BLOCK_K = PAGE_SIZE  # align BLOCK_K with page size for clean block table lookup
    HEAD_DIM = triton.next_power_of_2(D)

    if HEAD_DIM != D:
        q = torch.nn.functional.pad(q, (0, HEAD_DIM - D))

    out = torch.empty(S, H, HEAD_DIM, dtype=torch.float16, device=q.device)

    grid = (batch, triton.cdiv(S, BLOCK_Q), H)
    paged_prefill_attn_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        S,
        H,
        H_kv,
        scale,
        H_PER_KV=H_PER_KV,
        PAGE_SIZE=PAGE_SIZE,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        HEAD_DIM=HEAD_DIM,
    )

    if HEAD_DIM != D:
        out = out[:, :, :D]
    return out.contiguous()
