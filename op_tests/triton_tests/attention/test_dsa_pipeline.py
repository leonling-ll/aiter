# End-to-end test for the DeepSeek Sparse Attention (DSA, V3.2) prefill pipeline.
#
# Pipeline (BSHD layout for the MLA stage):
#   1. Lightning Indexer  (aiter.ops.triton.attention.fp8_mqa_logits.fp8_mqa_logits)
#        Q_idx [s_q, H_idx, D_idx] fp8 + KV_idx [s_kv, D_idx] fp8 + weights/scales
#        → logits [s_q, s_kv] f32 (-inf outside cu_start/cu_end window)
#
#   2. Top-K selector     (aiter.ops.triton.topk.topk)
#        logits → indices [s_q, topk] i64, sorted desc
#
#   3. Sparse MLA prefill (aiter.ops.triton._triton_kernels.attention.sparse_flash_mla.flash_mla_sparse_fwd)
#        Q_mla [B, S_q, H_q, D_qk] bf16 + KV_mla [B, S_kv, 1, D_qk] bf16
#        + indices [B, S_q, 1, topk] i32  → out [B, S_q, H_q, D_v]
#
# Indexer + topk are run per batch element; we then stack the indices and
# run the BSHD sparse-MLA kernel once over the whole batch.

import pytest
import torch

from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.topk import topk as triton_topk
from aiter.ops.triton._triton_kernels.attention.sparse_flash_mla import (
    flash_mla_sparse_fwd,
)
from aiter.ops.triton.utils.types import get_fp8_dtypes

from test_dsa_sparse_fwd import (
    _ref_sparse_attn_fwd,
    _allclose_or_inf_match,
    _cos_diff,
    TOL_OUT,
    TOL_LSE,
)
from test_fp8_mqa_logits import (
    per_custom_dims_cast_to_fp8,
    ref_fp8_mqa_logits,
    calc_diff,
)


_, e4m3_type = get_fp8_dtypes()


# -----------------------------------------------------------------------------
# Per-batch indexer + topk (kernel and reference paths)
# -----------------------------------------------------------------------------

def _kernel_indexer_topk(q_idx_fp8, kv_idx_fp8, kv_idx_scales, weights,
                         cu_ks, cu_ke, topk_k):
    """Run the indexer kernel and topk kernel on one batch element."""
    s_q = q_idx_fp8.shape[0]
    s_kv = kv_idx_fp8.shape[0]
    logits = fp8_mqa_logits(q_idx_fp8, kv_idx_fp8, kv_idx_scales,
                            weights, cu_ks, cu_ke)
    k_eff = min(topk_k, s_kv)
    _, idx_eff = triton_topk(logits, k=k_eff)
    if k_eff < topk_k:
        pad = torch.full((s_q, topk_k - k_eff), -1,
                         dtype=idx_eff.dtype, device=idx_eff.device)
        idx = torch.cat([idx_eff, pad], dim=-1)
    else:
        idx = idx_eff
    finite = torch.isfinite(logits).sum(dim=-1).to(torch.int32)
    tk_len = torch.minimum(finite, torch.tensor(topk_k, device=logits.device))
    return logits, idx_eff, idx, tk_len


def _ref_indexer_topk(q_idx_fp8, kv_idx_fp8, kv_idx_scales, weights,
                      cu_ks, cu_ke, topk_k):
    s_q = q_idx_fp8.shape[0]
    s_kv = kv_idx_fp8.shape[0]
    kv_dq = (kv_idx_fp8.float() * kv_idx_scales[:, None]).to(torch.bfloat16)
    logits, _ = ref_fp8_mqa_logits(
        q=q_idx_fp8.to(torch.bfloat16), kv=kv_dq, weights=weights,
        cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )
    k_eff = min(topk_k, s_kv)
    _, idx = torch.topk(logits, k=k_eff, dim=-1)
    if k_eff < topk_k:
        pad = torch.full((s_q, topk_k - k_eff), -1,
                         dtype=idx.dtype, device=idx.device)
        idx = torch.cat([idx, pad], dim=-1)
    finite = torch.isfinite(logits).sum(dim=-1).to(torch.int32)
    tk_len = torch.minimum(finite, torch.tensor(topk_k, device=logits.device))
    return logits, idx, tk_len


# -----------------------------------------------------------------------------
# Pipeline driver
# -----------------------------------------------------------------------------

def _run_pipeline_case(B, s_q, s_kv, topk_k, h_q, d_qk,
                       num_idx_heads=32, idx_head_dim=64,
                       seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    d_v = 512
    sm_scale = d_qk ** -0.5

    # ---- MLA inputs (BSHD) ----
    q_mla = torch.randn(B, s_q, h_q, d_qk, dtype=torch.bfloat16, device=device)
    kv_mla = torch.randn(B, s_kv, 1, d_qk, dtype=torch.bfloat16, device=device)

    # Per-batch indexer/topk artifacts (indexer is single-sequence).
    kr_indices_list, kr_tklen_list = [], []
    ref_indices_list, ref_tklen_list = [], []
    indexer_diffs = []
    topk_set_ok = []

    for b in range(B):
        q_idx_bf = torch.randn(s_q, num_idx_heads, idx_head_dim,
                               dtype=torch.bfloat16, device=device)
        kv_idx_bf = torch.randn(s_kv, idx_head_dim,
                                dtype=torch.bfloat16, device=device)
        q_idx_fp8 = q_idx_bf.to(e4m3_type)
        kv_idx_fp8, kv_idx_scales = per_custom_dims_cast_to_fp8(
            kv_idx_bf, (0,), use_ue8m0=False
        )
        weights = torch.randn(s_q, num_idx_heads,
                              dtype=torch.float32, device=device)
        cu_ks = torch.zeros(s_q, dtype=torch.int32, device=device)
        cu_ke = torch.full((s_q,), s_kv, dtype=torch.int32, device=device)

        kr_logits, kr_idx_eff, kr_idx, kr_tk_len = _kernel_indexer_topk(
            q_idx_fp8, kv_idx_fp8, kv_idx_scales, weights, cu_ks, cu_ke, topk_k,
        )
        ref_logits, ref_idx, ref_tk_len = _ref_indexer_topk(
            q_idx_fp8, kv_idx_fp8, kv_idx_scales, weights, cu_ks, cu_ke, topk_k,
        )

        indexer_diffs.append(calc_diff(kr_logits, ref_logits).item())
        # Stage-2 sanity: kernel topk indices match torch.topk on same logits.
        k_eff = min(topk_k, s_kv)
        _, ref_idx_on_kr = torch.topk(kr_logits, k=k_eff, dim=-1)
        topk_set_ok.append(
            torch.equal(kr_idx_eff.sort(dim=-1).values,
                        ref_idx_on_kr.sort(dim=-1).values)
        )

        kr_indices_list.append(kr_idx)
        kr_tklen_list.append(kr_tk_len)
        ref_indices_list.append(ref_idx)
        ref_tklen_list.append(ref_tk_len)

    indices_BS1T = torch.stack(kr_indices_list, dim=0).to(torch.int32).unsqueeze(2)
    kr_topk_len = torch.stack(kr_tklen_list, dim=0)
    ref_indices_BS1T = torch.stack(ref_indices_list, dim=0).to(torch.int32).unsqueeze(2)
    ref_topk_len = torch.stack(ref_tklen_list, dim=0)

    # ============ Stage 3: Sparse MLA fwd (BSHD, batched) ============
    kr_out, kr_ml, kr_lse = flash_mla_sparse_fwd(
        q_mla, kv_mla, indices_BS1T, sm_scale, d_v,
        attn_sink=None, topk_length=kr_topk_len,
    )
    assert kr_out.shape == (B, s_q, h_q, d_v)
    assert kr_out.dtype == torch.bfloat16

    # Reference using the same kernel-derived indices (isolates stage-3 noise).
    ref_out_kr_idx, ref_ml_kr_idx, ref_lse_kr_idx = _ref_sparse_attn_fwd(
        q_mla, kv_mla, indices_BS1T, sm_scale, d_v,
        attn_sink=None, topk_length=kr_topk_len,
    )
    # Full-reference end-to-end (uses ref-derived indices).
    ref_out_e2e, _, _ = _ref_sparse_attn_fwd(
        q_mla, kv_mla, ref_indices_BS1T, sm_scale, d_v,
        attn_sink=None, topk_length=ref_topk_len,
    )

    prefix = (f"[B={B} s_q={s_q} s_kv={s_kv} topk={topk_k} h_q={h_q} d_qk={d_qk}]")

    # ---- Stage-1 check: indexer logits ----
    worst = max(indexer_diffs)
    assert worst < 1e-3, f"{prefix} indexer logits cos_diff={worst:.2e}"

    # ---- Stage-2 check: topk index sets ----
    assert all(topk_set_ok), f"{prefix} topk index sets differ"

    # ---- Stage-3 check ----
    cos = _cos_diff(kr_out.float(), ref_out_kr_idx)
    elem_ok = _allclose_or_inf_match(
        kr_out.float(), ref_out_kr_idx, TOL_OUT["abs_tol"], TOL_OUT["rel_tol"],
    )
    assert elem_ok or cos < TOL_OUT["cos_diff_tol"], (
        f"{prefix} sparse_fwd out cos_diff={cos:.2e}"
    )
    assert _allclose_or_inf_match(
        kr_ml, ref_ml_kr_idx, TOL_LSE["abs_tol"], TOL_LSE["rel_tol"]
    ), f"{prefix} max_logits mismatch"
    assert _allclose_or_inf_match(
        kr_lse, ref_lse_kr_idx, TOL_LSE["abs_tol"], TOL_LSE["rel_tol"]
    ), f"{prefix} lse mismatch"

    # End-to-end loose check.
    e2e_cos = _cos_diff(kr_out.float(), ref_out_e2e)
    assert e2e_cos < 5e-2, f"{prefix} end-to-end cos_diff={e2e_cos:.2e}"


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("B,s_q,s_kv,topk_k", [
    (1, 32, 256, 128),
    (2, 32, 256, 128),
    (1, 64, 512, 256),
    (3, 64, 512, 256),
    (1, 128, 1024, 256),
    (1, 16, 256, 256),     # topk == s_kv
])
@pytest.mark.parametrize("h_q", [64, 128])
@pytest.mark.parametrize("d_qk", [512, 576])
@torch.inference_mode()
def test_dsa_pipeline(B, s_q, s_kv, topk_k, h_q, d_qk):
    """Indexer → topk → BSHD sparse MLA fwd, validated stage-by-stage."""
    _run_pipeline_case(B, s_q, s_kv, topk_k, h_q, d_qk)


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("s_q,s_kv,topk_k", [
    (16, 64, 128),     # s_kv < topk → topk_length kicks in
    (32, 96, 256),
])
@pytest.mark.parametrize("h_q", [64])
@pytest.mark.parametrize("d_qk", [576])
@torch.inference_mode()
def test_dsa_pipeline_topk_oversubscribed(B, s_q, s_kv, topk_k, h_q, d_qk):
    _run_pipeline_case(B, s_q, s_kv, topk_k, h_q, d_qk)


@torch.inference_mode()
def test_dsa_pipeline_smoke():
    _run_pipeline_case(B=2, s_q=8, s_kv=128, topk_k=64, h_q=64, d_qk=512)
