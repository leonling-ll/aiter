# Triton port of FlashMLA's sparse-attention prefill forward
# (kernel at aiter/ops/triton/_triton_kernels/attention/sparse_flash_mla.py).
#
# BSHD layout:  q [B, S_q, H_q, D_qk], kv [B, S_kv, 1, D_qk],
#               indices [B, S_q, 1, topk], topk_length [B, S_q].
#
# Reference grid adapted from FlashMLA's correctness suite.

from typing import Optional, Tuple

import pytest
import torch

from aiter.ops.triton._triton_kernels.attention.sparse_flash_mla import (
    flash_mla_sparse_fwd,
)


# -----------------------------------------------------------------------------
# Reference implementation (PyTorch fp32, BSHD)
# -----------------------------------------------------------------------------

def _ref_sparse_attn_fwd(
    q: torch.Tensor,            # [B, S_q, H_q, D_qk] bf16
    kv: torch.Tensor,           # [B, S_kv, 1, D_qk]  bf16
    indices: torch.Tensor,      # [B, S_q, 1, topk]   i32
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,    # [H_q] f32
    topk_length: Optional[torch.Tensor] = None,  # [B, S_q] i32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (out_fp32 [B,S_q,H_q,D_v], max_logits [B,S_q,H_q], lse [B,S_q,H_q])."""
    B, s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[1]
    topk = indices.shape[-1]
    device = q.device

    idx = indices.clone().squeeze(2)  # [B, S_q, topk]
    if topk_length is not None:
        oob = (
            torch.arange(topk, device=device).view(1, 1, topk).expand(B, s_q, topk)
            >= topk_length.unsqueeze(-1)
        )
        idx[oob] = -1
    invalid = (idx < 0) | (idx >= s_kv)
    idx_safe = idx.clone()
    idx_safe[invalid] = 0

    # Gather kv per batch.
    kv_b = kv.squeeze(2).float()                                # [B, S_kv, D_qk]
    batch_offsets = (
        torch.arange(B, device=device).view(B, 1, 1) * s_kv
    ).expand(B, s_q, topk)
    flat_idx = (idx_safe.long() + batch_offsets).reshape(-1)
    gathered = kv_b.reshape(B * s_kv, d_qk).index_select(0, flat_idx).reshape(
        B, s_q, topk, d_qk
    )

    P = torch.einsum("bshd,bstd->bsht", q.float(), gathered) * sm_scale  # [B,S_q,H_q,topk]
    P[invalid.unsqueeze(2).expand_as(P)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)         # [B,S_q,H_q]
    max_logits = P.max(dim=-1).values             # [B,S_q,H_q]

    if attn_sink is not None:
        sink_b = attn_sink.float().view(1, 1, h_q).expand_as(orig_lse)
        lse_o = torch.logsumexp(torch.stack([orig_lse, sink_b], dim=0), dim=0)
    else:
        lse_o = orig_lse.clone()
    lse_o[lse_o == float("-inf")] = float("inf")
    s_o = torch.exp(P - lse_o.unsqueeze(-1))
    out_fp32 = torch.einsum("bsht,bstd->bshd", s_o, gathered[..., :d_v])

    orig_lse[orig_lse == float("-inf")] = float("inf")
    return out_fp32, max_logits, orig_lse


# -----------------------------------------------------------------------------
# Numerical tolerance
# -----------------------------------------------------------------------------

def _cos_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double().flatten()
    y = y.double().flatten()
    denom = (x * x + y * y).sum()
    return (1.0 - 2.0 * (x * y).sum() / denom).item()


def _allclose_or_inf_match(actual, ref, atol, rtol):
    inf_match = (
        torch.isinf(actual)
        & torch.isinf(ref)
        & ((actual < 0) == (ref < 0))
    )
    finite_close = torch.isclose(actual, ref, atol=atol, rtol=rtol)
    return bool((inf_match | finite_close).all().item())


TOL_OUT = dict(abs_tol=8e-4, rel_tol=3.01 / 128, cos_diff_tol=7e-6)
TOL_LSE = dict(abs_tol=1e-6, rel_tol=2.01 / 65536)


def _check(prefix, kernel_out, kernel_ml, kernel_lse, ref_out_fp32, ref_ml, ref_lse):
    cos = _cos_diff(kernel_out.float(), ref_out_fp32)
    out_ok_elem = _allclose_or_inf_match(
        kernel_out.float(), ref_out_fp32, TOL_OUT["abs_tol"], TOL_OUT["rel_tol"]
    )
    assert out_ok_elem or cos < TOL_OUT["cos_diff_tol"], (
        f"{prefix} out: cos_diff={cos:.2e}"
    )
    assert _allclose_or_inf_match(
        kernel_ml, ref_ml, TOL_LSE["abs_tol"], TOL_LSE["rel_tol"]
    ), f"{prefix} max_logits mismatch"
    assert _allclose_or_inf_match(
        kernel_lse, ref_lse, TOL_LSE["abs_tol"], TOL_LSE["rel_tol"]
    ), f"{prefix} lse mismatch"


def _run_case(
    B: int,
    s_q: int,
    s_kv: int,
    topk: int,
    h_q: int,
    d_qk: int,
    have_attn_sink: bool,
    have_topk_length: bool,
    is_all_indices_invalid: bool = False,
    seed: int = 0,
):
    torch.manual_seed(seed)
    device = "cuda"
    d_v = 512
    sm_scale = d_qk ** -0.5

    q = torch.randn(B, s_q, h_q, d_qk, dtype=torch.bfloat16, device=device)
    kv = torch.randn(B, s_kv, 1, d_qk, dtype=torch.bfloat16, device=device)

    if is_all_indices_invalid:
        indices = torch.full(
            (B, s_q, 1, topk), -1, dtype=torch.int32, device=device
        )
    else:
        indices = torch.randint(
            0, s_kv, (B, s_q, 1, topk), dtype=torch.int32, device=device
        )
        rand_mask = torch.rand((B, s_q, 1, topk), device=device)
        indices[rand_mask < 0.05] = -1
        indices[(rand_mask >= 0.95)] = s_kv + 7  # OOB

    attn_sink = (
        torch.randn(h_q, dtype=torch.float32, device=device)
        if have_attn_sink else None
    )
    topk_length = (
        torch.randint(0, topk + 1, (B, s_q), dtype=torch.int32, device=device)
        if have_topk_length else None
    )

    out, max_logits, lse = flash_mla_sparse_fwd(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length
    )
    ref_out_fp32, ref_ml, ref_lse = _ref_sparse_attn_fwd(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length
    )

    prefix = (
        f"[B={B} s_q={s_q} s_kv={s_kv} topk={topk} h_q={h_q} d_qk={d_qk} "
        f"sink={have_attn_sink} tklen={have_topk_length} "
        f"all_invalid={is_all_indices_invalid}]"
    )
    _check(prefix, out, max_logits, lse, ref_out_fp32, ref_ml, ref_lse)


# -----------------------------------------------------------------------------
# Parametrized tests
# -----------------------------------------------------------------------------

_SHAPES = [
    (128, 128),
    (256, 256),
    (512, 512),
    (592, 128),
    (1840, 256),
    (1521, 512),
    (95, 128),
    (153, 256),
]


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("s_q", [1, 62, 213])
@pytest.mark.parametrize("s_kv,topk", _SHAPES)
@pytest.mark.parametrize("h_q", [64, 128])
@pytest.mark.parametrize("d_qk", [512, 576])
@torch.inference_mode()
def test_sparse_fwd_basic(B, s_q, s_kv, topk, h_q, d_qk):
    _run_case(B, s_q, s_kv, topk, h_q, d_qk,
              have_attn_sink=False, have_topk_length=False)


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("s_q", [62, 213])
@pytest.mark.parametrize("s_kv,topk", [
    (1840, 256), (1521, 512), (95, 128),
])
@pytest.mark.parametrize("h_q", [64, 128])
@pytest.mark.parametrize("d_qk", [512, 576])
@pytest.mark.parametrize("have_attn_sink", [False, True])
@pytest.mark.parametrize("have_topk_length", [False, True])
@torch.inference_mode()
def test_sparse_fwd_with_features(
    B, s_q, s_kv, topk, h_q, d_qk, have_attn_sink, have_topk_length,
):
    _run_case(B, s_q, s_kv, topk, h_q, d_qk,
              have_attn_sink=have_attn_sink,
              have_topk_length=have_topk_length)


@pytest.mark.parametrize("B,s_q,s_kv,topk", [
    (1, 1, 128, 128),
    (2, 1, 256, 256),
    (1, 256, 1024, 1024),
])
@pytest.mark.parametrize("h_q", [64, 128])
@pytest.mark.parametrize("d_qk", [512, 576])
@torch.inference_mode()
def test_sparse_fwd_all_invalid_indices(B, s_q, s_kv, topk, h_q, d_qk):
    _run_case(B, s_q, s_kv, topk, h_q, d_qk,
              have_attn_sink=True, have_topk_length=True,
              is_all_indices_invalid=True)


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("s_q", [1, 64])
@pytest.mark.parametrize("s_kv,topk", [
    (32, 2048),
    (64, 1024),
])
@pytest.mark.parametrize("h_q", [64, 128])
@pytest.mark.parametrize("d_qk", [512, 576])
@torch.inference_mode()
def test_sparse_fwd_oversubscribed_topk(B, s_q, s_kv, topk, h_q, d_qk):
    _run_case(B, s_q, s_kv, topk, h_q, d_qk,
              have_attn_sink=True, have_topk_length=True)


# -----------------------------------------------------------------------------
# I/O contract checks
# -----------------------------------------------------------------------------

@torch.inference_mode()
def test_sparse_fwd_output_shapes_and_dtypes():
    B, s_q, s_kv, topk, h_q, d_qk, d_v = 2, 4, 256, 128, 128, 576, 512
    q = torch.randn(B, s_q, h_q, d_qk, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B, s_kv, 1, d_qk, dtype=torch.bfloat16, device="cuda")
    indices = torch.randint(0, s_kv, (B, s_q, 1, topk),
                            dtype=torch.int32, device="cuda")

    out, max_logits, lse = flash_mla_sparse_fwd(q, kv, indices, d_qk ** -0.5, d_v)

    assert out.shape == (B, s_q, h_q, d_v) and out.dtype == torch.bfloat16
    assert max_logits.shape == (B, s_q, h_q) and max_logits.dtype == torch.float32
    assert lse.shape == (B, s_q, h_q) and lse.dtype == torch.float32


@torch.inference_mode()
def test_sparse_fwd_lonely_q_writes_zero():
    B, s_q, s_kv, topk, h_q, d_qk, d_v = 2, 3, 64, 128, 64, 512, 512
    q = torch.randn(B, s_q, h_q, d_qk, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B, s_kv, 1, d_qk, dtype=torch.bfloat16, device="cuda")
    indices = torch.full((B, s_q, 1, topk), -1, dtype=torch.int32, device="cuda")
    sink = torch.randn(h_q, dtype=torch.float32, device="cuda")

    out, max_logits, lse = flash_mla_sparse_fwd(
        q, kv, indices, d_qk ** -0.5, d_v, attn_sink=sink
    )
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.all(max_logits == float("-inf"))
    assert torch.all(lse == float("+inf"))


@torch.inference_mode()
def test_sparse_fwd_topk_length_zero():
    B, s_q, s_kv, topk, h_q, d_qk, d_v = 2, 4, 128, 128, 64, 576, 512
    q = torch.randn(B, s_q, h_q, d_qk, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B, s_kv, 1, d_qk, dtype=torch.bfloat16, device="cuda")
    indices = torch.randint(0, s_kv, (B, s_q, 1, topk),
                            dtype=torch.int32, device="cuda")
    topk_length = torch.tensor(
        [[0, topk, 0, topk // 2], [topk, 0, topk // 2, 0]],
        dtype=torch.int32, device="cuda",
    )

    out, max_logits, lse = flash_mla_sparse_fwd(
        q, kv, indices, d_qk ** -0.5, d_v, topk_length=topk_length
    )

    # Rows where topk_length == 0 must be lonely.
    zero_mask = (topk_length == 0)
    nonzero_mask = ~zero_mask
    assert torch.equal(out[zero_mask], torch.zeros_like(out[zero_mask]))
    assert torch.all(max_logits[zero_mask] == float("-inf"))
    assert torch.all(lse[zero_mask] == float("+inf"))
    assert torch.isfinite(out[nonzero_mask]).all()
    assert torch.isfinite(lse[nonzero_mask]).all()
