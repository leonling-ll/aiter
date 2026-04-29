# Benchmark for the DeepSeek Sparse Attention (DSA, V3.2) prefill pipeline.
#
#   1. Lightning indexer  (fp8_mqa_logits)            — per batch element
#   2. Top-K selector     (triton topk)               — per batch element
#   3. Sparse MLA (flash_mla_sparse_fwd)              — once over the batch
#
import argparse
import os
import sys

import torch

from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.topk import topk as triton_topk
from aiter.ops.triton._triton_kernels.attention.sparse_flash_mla import (
    flash_mla_sparse_fwd,
)
from aiter.ops.triton.utils.types import get_fp8_dtypes

# Reuse the fp8 cast helper from the indexer test suite.
_TEST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "triton_tests", "attention")
)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
from test_fp8_mqa_logits import per_custom_dims_cast_to_fp8  # noqa: E402

_, e4m3_type = get_fp8_dtypes()


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

def _make_pipeline_inputs(B, s_q, s_kv, h_q, d_qk,
                          num_idx_heads, idx_head_dim, seed=0):
    torch.manual_seed(seed)
    device = "cuda"

    # MLA stage (BSHD). KV has 1 head — that is the MLA convention; the kernel
    # does not accept multi-head KV. The user-facing h_kv argument is checked
    # but ignored when constructing kv.
    q_mla = torch.randn(B, s_q, h_q, d_qk, dtype=torch.bfloat16, device=device)
    kv_mla = torch.randn(B, s_kv, 1, d_qk, dtype=torch.bfloat16, device=device)

    per_b = []
    for _ in range(B):
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
        per_b.append((q_idx_fp8, kv_idx_fp8, kv_idx_scales,
                      weights, cu_ks, cu_ke))
    return q_mla, kv_mla, per_b


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def _run_pipeline(q_mla, kv_mla, per_b_idx_inputs, topk_k, sm_scale, d_v):
    s_q = q_mla.shape[1]
    s_kv = kv_mla.shape[1]
    device = q_mla.device

    indices_list, tklen_list = [], []
    for (q_idx_fp8, kv_idx_fp8, kv_idx_scales,
         weights, cu_ks, cu_ke) in per_b_idx_inputs:
        with torch.profiler.record_function("indexer_fp8_mqa_logits"):
            logits = fp8_mqa_logits(q_idx_fp8, kv_idx_fp8, kv_idx_scales,
                                    weights, cu_ks, cu_ke)
        k_eff = min(topk_k, s_kv)
        with torch.profiler.record_function("topk"):
            _, idx_eff = triton_topk(logits, k=k_eff)
        if k_eff < topk_k:
            pad = torch.full((s_q, topk_k - k_eff), -1,
                             dtype=idx_eff.dtype, device=device)
            idx = torch.cat([idx_eff, pad], dim=-1)
        else:
            idx = idx_eff
        finite = torch.isfinite(logits).sum(dim=-1).to(torch.int32)
        tk_len = torch.minimum(finite, torch.tensor(topk_k, device=device))
        indices_list.append(idx)
        tklen_list.append(tk_len)

    indices_BS1T = torch.stack(indices_list, dim=0).to(torch.int32).unsqueeze(2)
    topk_len = torch.stack(tklen_list, dim=0)

    with torch.profiler.record_function("sparse_mla_fwd"):
        out, ml, lse = flash_mla_sparse_fwd(
            q_mla, kv_mla, indices_BS1T, sm_scale, d_v,
            attn_sink=None, topk_length=topk_len,
        )
    return out, ml, lse


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

_STAGE_KEYS = ("indexer_fp8_mqa_logits", "topk", "sparse_mla_fwd")


def run_benchmark(args):
    if args.h_kv != 1:
        print(f"[warn] sparse_flash_mla requires h_kv=1 (MLA); "
              f"ignoring --h_kv={args.h_kv} and using 1.")
    sm_scale = args.d_qk ** -0.5

    q_mla, kv_mla, per_b = _make_pipeline_inputs(
        args.B, args.s_q, args.s_kv, args.h_q, args.d_qk,
        args.num_idx_heads, args.idx_head_dim,
    )

    for _ in range(args.warmup):
        _run_pipeline(q_mla, kv_mla, per_b, args.topk, sm_scale, args.d_v)
    torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities,
                                record_shapes=False) as prof:
        for _ in range(args.iters):
            _run_pipeline(q_mla, kv_mla, per_b, args.topk, sm_scale, args.d_v)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

def main():
    parser = argparse.ArgumentParser(
        description="DSA prefill pipeline benchmark "
                    "(indexer + topk + sparse MLA fwd, BSHD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--B", type=int, default=16, help="batch size")
    parser.add_argument("--s_q", type=int, default=4096, help="num query tokens")
    parser.add_argument("--s_kv", type=int, default=4096, help="num KV tokens")
    parser.add_argument("--topk", type=int, default=2048, help="indices per query")
    parser.add_argument("--h_q", type=int, default=16, help="MLA query heads")
    parser.add_argument("--h_kv", type=int, default=1,
                        help="MLA KV heads (must be 1; flag accepted for table parity)")
    parser.add_argument("--d_qk", type=int, default=576, help="QK head dim")
    parser.add_argument("--d_v", type=int, default=512, help="V head dim")
    parser.add_argument("--num_idx_heads", type=int, default=32,
                        help="lightning-indexer query heads")
    parser.add_argument("--idx_head_dim", type=int, default=64,
                        help="lightning-indexer head dim")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("-o", action="store_true", help="write CSV")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
