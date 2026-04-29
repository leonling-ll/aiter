from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_mla_sparse_fwd_kernel(
    Q_ptr,            # [B, S_q, H_q, D_qk] bf16
    KV_ptr,           # [B, S_kv, 1, D_qk]  bf16
    Indices_ptr,      # [B, S_q, 1, topk]   i32
    AttnSink_ptr,     # [H_q] f32 or 0
    TopkLen_ptr,      # [B, S_q] i32 or 0
    Out_ptr,          # [B, S_q, H_q, D_v]  bf16
    MaxLogits_ptr,    # [B, S_q, H_q] f32
    LSE_ptr,          # [B, S_q, H_q] f32
    sm_scale,
    s_kv,
    s_q: tl.constexpr,
    h_q: tl.constexpr,
    topk,
    stride_q_b, stride_q_s, stride_q_h,
    stride_kv_b, stride_kv_s,
    stride_idx_b, stride_idx_s,
    stride_o_b, stride_o_s, stride_o_h,
    stride_ml_b, stride_ml_s,
    stride_tl_b,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    D_LORA: tl.constexpr,
    D_ROPE: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # One program per (batch, q-token, head-block).
    H_BLOCKS: tl.constexpr = h_q // BLOCK_H
    QH: tl.constexpr = s_q * H_BLOCKS
    pid = tl.program_id(0)
    b_idx = pid // QH
    qh = pid % QH
    q_idx = qh // H_BLOCKS
    h_block = qh % H_BLOCKS
    h_off = h_block * BLOCK_H

    offs_h = h_off + tl.arange(0, BLOCK_H)
    offs_lora = tl.arange(0, D_LORA)
    offs_t = tl.arange(0, BLOCK_T)

    # --- Load Q ---
    q_base = (
        Q_ptr
        + b_idx * stride_q_b
        + q_idx * stride_q_s
        + offs_h[:, None] * stride_q_h
    )
    q_lora = tl.load(q_base + offs_lora[None, :])
    if D_ROPE > 0:
        offs_rope = tl.arange(0, D_ROPE)
        q_rope = tl.load(q_base + (D_LORA + offs_rope)[None, :])

    # --- Per-row topk_length ---
    if HAVE_TOPK_LENGTH:
        tk_len = tl.load(TopkLen_ptr + b_idx * stride_tl_b + q_idx)
    else:
        tk_len = topk

    m_i = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)

    indices_row = Indices_ptr + b_idx * stride_idx_b + q_idx * stride_idx_s
    kv_batch_base = KV_ptr + b_idx * stride_kv_b
    n_blocks = tl.cdiv(topk, BLOCK_T)

    for b in range(n_blocks):
        t_pos = b * BLOCK_T + offs_t
        in_topk = t_pos < topk
        idx = tl.load(indices_row + t_pos, mask=in_topk, other=-1)
        valid = in_topk & (idx >= 0) & (idx < s_kv)
        if HAVE_TOPK_LENGTH:
            valid = valid & (t_pos < tk_len)
        idx_safe = tl.where(valid, idx, 0)

        kT_lora = tl.load(
            kv_batch_base + idx_safe[None, :] * stride_kv_s + offs_lora[:, None],
            mask=valid[None, :],
            other=0.0,
        )  # [D_LORA, BLOCK_T]

        s = tl.dot(q_lora, kT_lora)  # [BLOCK_H, BLOCK_T]
        if D_ROPE > 0:
            kT_rope = tl.load(
                kv_batch_base + idx_safe[None, :] * stride_kv_s + (D_LORA + offs_rope)[:, None],
                mask=valid[None, :],
                other=0.0,
            )
            s += tl.dot(q_rope, kT_rope)
        s = s * sm_scale
        s = tl.where(valid[None, :], s, -float("inf"))

        # Online softmax update.
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

        v_lora = tl.load(
            kv_batch_base + idx_safe[:, None] * stride_kv_s + offs_lora[None, :],
            mask=valid[:, None],
            other=0.0,
        )  # [BLOCK_T, D_V]
        acc += tl.dot(p.to(v_lora.dtype), v_lora)

    # --- Epilogue ---
    no_valid = l_i == 0.0
    m_i_safe = tl.where(no_valid, 0.0, m_i)

    if HAVE_ATTN_SINK:
        attn_sink = tl.load(AttnSink_ptr + offs_h)
        denom = l_i + tl.exp(attn_sink - m_i_safe)
        scale = 1.0 / denom
    else:
        scale = tl.where(no_valid, 0.0, 1.0 / tl.where(no_valid, 1.0, l_i))

    out = acc * scale[:, None]

    offs_v = tl.arange(0, D_V)
    out_base = (
        Out_ptr
        + b_idx * stride_o_b
        + q_idx * stride_o_s
        + offs_h[:, None] * stride_o_h
    )
    tl.store(out_base + offs_v[None, :], out.to(Out_ptr.dtype.element_ty))

    max_logits = tl.where(no_valid, -float("inf"), m_i)
    lse_out = tl.where(no_valid, float("inf"), tl.log(l_i) + m_i)
    ml_base = MaxLogits_ptr + b_idx * stride_ml_b + q_idx * stride_ml_s
    lse_base = LSE_ptr + b_idx * stride_ml_b + q_idx * stride_ml_s
    tl.store(ml_base + offs_h, max_logits)
    tl.store(lse_base + offs_h, lse_out)


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    q:           [B, S_q, H_q, D_qk] bf16
    kv:          [B, S_kv, 1, D_qk]  bf16
    indices:     [B, S_q, 1, topk]   i32
    attn_sink:   [H_q] f32           (optional; per-head, no batch dim)
    topk_length: [B, S_q] i32        (optional)

    Returns (out, max_logits, lse) shaped
        [B, S_q, H_q, D_v], [B, S_q, H_q], [B, S_q, H_q].
    """
    assert q.dim() == 4 and kv.dim() == 4 and indices.dim() == 4, ()
    B, s_q, h_q, d_qk = q.shape
    B_kv, s_kv, h_kv, d_qk_kv = kv.shape
    assert B_kv == B and h_kv == 1 and d_qk_kv == d_qk
    assert d_v == 512 and d_qk in (512, 576)
    assert h_q in (16, 32, 64, 128)
    assert q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
    assert indices.dtype == torch.int32
    assert indices.shape[:3] == (B, s_q, h_kv)
    topk = indices.shape[-1]

    out = torch.empty((B, s_q, h_q, d_v), dtype=torch.bfloat16, device=q.device)
    max_logits = torch.empty((B, s_q, h_q), dtype=torch.float32, device=q.device)
    lse = torch.empty((B, s_q, h_q), dtype=torch.float32, device=q.device)

    have_sink = attn_sink is not None
    have_tk_len = topk_length is not None

    if have_tk_len:
        assert topk_length.shape == (B, s_q)
        tl_stride_b = topk_length.stride(0)
    else:
        tl_stride_b = 0

    BLOCK_H = 16
    BLOCK_T = 64
    D_LORA = 512
    D_ROPE = d_qk - 512  # 0 or 64

    grid = (B * s_q * (h_q // BLOCK_H),)
    _flash_mla_sparse_fwd_kernel[grid](
        q, kv, indices,
        attn_sink if have_sink else q,
        topk_length if have_tk_len else q,
        out, max_logits, lse,
        sm_scale,
        s_kv,
        s_q,
        h_q,
        topk,
        q.stride(0), q.stride(1), q.stride(2),
        kv.stride(0), kv.stride(1),
        indices.stride(0), indices.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        max_logits.stride(0), max_logits.stride(1),
        tl_stride_b,
        HAVE_ATTN_SINK=have_sink,
        HAVE_TOPK_LENGTH=have_tk_len,
        D_LORA=D_LORA,
        D_ROPE=D_ROPE,
        D_V=d_v,
        BLOCK_H=BLOCK_H,
        BLOCK_T=BLOCK_T,
        num_stages=1,
        num_warps=4,
    )
    return out, max_logits, lse
