# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Standalone tuner for the bf16 A16W16 GEMM (gluon on gfx1250 / MI455, triton
elsewhere).

Reads a shapes CSV (M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle -- the
ATOM ``bf16_untuned_collected_gemm.csv`` format), buckets M per (N,K) the way
production looks configs up, sweeps a candidate config space for each
(bucket, N, K) representative shape, correctness-gates every candidate against a
torch ``F.linear`` baseline, benchmarks the survivors, and writes the winners to
``{arch}-GEMM-A16W16-N={N}-K={K}.json`` in the standard M_LEQ/M_GEQ/any format.

Typical use on MI455:
    python -m op_tests.tuning_tests.gemm_a16w16.tune \
        --csv /home/leling/ATOM/model_collection/bf16_untuned_collected_gemm.csv

Results are checkpointed to a JSONL log so an interrupted run resumes with
``--resume`` and partial progress survives the push/pull workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Optional

import torch
import triton

from op_tests.tuning_tests.gemm_a16w16 import buckets as B
from op_tests.tuning_tests.gemm_a16w16.gemm_ref import (
    make_inputs,
    torch_gemm,
    triton_gemm,
    resolve_backend,
    get_arch,
)
from op_tests.tuning_tests.gemm_a16w16.search_space import candidates_for, clean_config

_DEFAULT_CSV = "/home/leling/ATOM/model_collection/bf16_untuned_collected_gemm.csv"
_DEFAULT_OUT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "aiter", "ops", "triton", "configs", "gemm"
)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def read_shapes(path: str) -> list[dict]:
    """Parse the untuned-GEMM CSV into a list of shape dicts (bf16 rows only)."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            dt = (r.get("dtype") or "").strip()
            if dt and "bfloat16" not in dt and "float16" not in dt:
                continue  # A16W16 is 16-bit only
            rows.append(
                {
                    "M": int(r["M"]),
                    "N": int(r["N"]),
                    "K": int(r["K"]),
                    "bias": (r.get("bias", "False").strip().lower() == "true"),
                }
            )
    return rows


def plan_points(rows: list[dict], all_m: bool) -> dict[tuple, dict]:
    """Group rows by (N,K) and decide which (bucket, M) points to tune.

    Returns {(N,K): {"bias": bool, "buckets": {bucket_key: [tune_M, ...]}}}.
    In representative mode each bucket has exactly one tune_M (its heaviest);
    in all-m mode every distinct M in the bucket is tuned.
    """
    by_nk_bias: dict[tuple, bool] = {}
    by_nk_ms: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        nk = (r["N"], r["K"])
        by_nk_ms[nk].append(r["M"])
        by_nk_bias[nk] = by_nk_bias.get(nk, False) or r["bias"]

    plan: dict[tuple, dict] = {}
    for nk, ms in by_nk_ms.items():
        grouped = B.group_by_bucket(ms)
        tune_buckets = {}
        for bk, bucket_ms in grouped.items():
            if all_m:
                tune_buckets[bk] = sorted(set(bucket_ms))
            else:
                tune_buckets[bk] = [B.representative_m(bucket_ms)]
        plan[nk] = {"bias": by_nk_bias[nk], "buckets": tune_buckets}
    return plan


# ---------------------------------------------------------------------------
# Checkpoint log
# ---------------------------------------------------------------------------


def _cfg_key(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)


class ResultLog:
    """Append-only JSONL of measurements, keyed by (N,K,M,backend,cfg)."""

    def __init__(self, path: str, resume: bool):
        self.path = path
        self.records: dict[tuple, dict] = {}
        if resume and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.records[self._key(rec)] = rec
            print(f"[resume] loaded {len(self.records)} prior measurements from {path}")
        self._fh = open(path, "a")

    @staticmethod
    def _key(rec: dict) -> tuple:
        return (rec["N"], rec["K"], rec["M"], rec["backend"], rec["cfg_key"])

    def get(self, N, K, M, backend, cfg_key) -> Optional[dict]:
        return self.records.get((N, K, M, backend, cfg_key))

    def add(self, rec: dict):
        self.records[self._key(rec)] = rec
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


# ---------------------------------------------------------------------------
# Benchmark one candidate
# ---------------------------------------------------------------------------


def bench_candidate(
    x, w, bias, dtype, cfg, backend, atol, rtol, warmup, rep
) -> tuple[Optional[float], str]:
    """Correctness-gate then time one candidate. Returns (ms, err).

    ms is None on any failure (compile error, validation mismatch, OOM); err
    carries a short reason for the log. The correctness gate runs first so we
    never report a latency for a wrong result.
    """
    kt = cfg.get("kernel_type", "bandwidth_bound")
    try:
        ref = torch_gemm(x, w, bias, dtype)
        out = triton_gemm(
            x, w, bias=bias, dtype=dtype, config=cfg, backend=backend, kernel_type=kt
        )
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
    except Exception as e:  # noqa: BLE001 - any failure disqualifies the candidate
        return None, f"{type(e).__name__}: {str(e)[:180]}"

    try:
        ms = triton.testing.do_bench(
            lambda: triton_gemm(
                x, w, bias=bias, dtype=dtype, config=cfg, backend=backend, kernel_type=kt
            ),
            warmup=warmup,
            rep=rep,
        )
    except Exception as e:  # noqa: BLE001
        return None, f"bench {type(e).__name__}: {str(e)[:180]}"
    return float(ms), ""


def tflops(M, N, K, ms) -> float:
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


# ---------------------------------------------------------------------------
# Tune one (N,K)
# ---------------------------------------------------------------------------


def tune_nk(
    N, K, spec, backend, args, log: ResultLog
) -> dict:
    """Tune every (bucket, tune_M) for one (N,K); return the per-bucket best.

    {bucket_key: {"config": cfg, "M": tune_M, "ms": ms, "tflops": t}}
    """
    dtype = torch.bfloat16
    bias_flag = spec["bias"]
    results: dict[str, dict] = {}

    # measured[M][cfg_key] = ms (successful only) -- used for all-m reduction
    measured: dict[int, dict[str, float]] = defaultdict(dict)
    cfg_by_key: dict[str, dict] = {}

    all_tune_ms = sorted({m for ms in spec["buckets"].values() for m in ms})
    for M in all_tune_ms:
        x, w, bias = make_inputs(M, N, K, dtype=dtype, bias=bias_flag)
        cands = candidates_for(
            backend, M, N, K, exhaustive=args.exhaustive, max_candidates=args.max_candidates
        )
        print(
            f"  [M={M:6d} N={N:6d} K={K:5d}] {len(cands)} candidates ({backend})",
            flush=True,
        )
        best_ms = None
        for i, cfg in enumerate(cands):
            cfg = clean_config(backend, cfg)
            ck = _cfg_key(cfg)
            cfg_by_key[ck] = cfg
            cached = log.get(N, K, M, backend, ck)
            if cached is not None:
                ms = cached["ms"]
            else:
                ms, err = bench_candidate(
                    x, w, bias, dtype, cfg, backend, args.atol, args.rtol,
                    args.warmup, args.rep,
                )
                log.add(
                    {
                        "N": N, "K": K, "M": M, "backend": backend,
                        "cfg_key": ck, "cfg": cfg, "ms": ms, "err": err,
                    }
                )
            if ms is not None:
                measured[M][ck] = ms
                if best_ms is None or ms < best_ms:
                    best_ms = ms
        # free operands before next M
        del x, w, bias
        torch.cuda.empty_cache()

    # Reduce measurements into a per-bucket winning config.
    for bk, bucket_ms in spec["buckets"].items():
        cfg, chosen_m, ms = _reduce_bucket(bucket_ms, measured, cfg_by_key)
        if cfg is None:
            print(f"    !! no valid config for bucket {bk} (N={N},K={K}) -- skipped")
            continue
        results[bk] = {
            "config": cfg,
            "M": chosen_m,
            "ms": ms,
            "tflops": tflops(chosen_m, N, K, ms),
        }
    return results


def _reduce_bucket(bucket_ms, measured, cfg_by_key):
    """Pick the winning config for a bucket.

    Representative mode: one M, argmin latency.
    All-m mode: config minimizing mean normalized latency across the bucket's M
    (only configs measured at every M in the bucket are comparable; fall back to
    the heaviest M's best if none are common).
    """
    bucket_ms = sorted(set(bucket_ms))
    if len(bucket_ms) == 1:
        M = bucket_ms[0]
        table = measured.get(M, {})
        if not table:
            return None, M, None
        ck = min(table, key=table.get)
        return cfg_by_key[ck], M, table[ck]

    # all-m: configs present for all Ms
    common = None
    for M in bucket_ms:
        keys = set(measured.get(M, {}).keys())
        common = keys if common is None else (common & keys)
    if common:
        best = min(
            common,
            key=lambda ck: sum(
                measured[M][ck] / min(measured[M].values()) for M in bucket_ms
            )
            / len(bucket_ms),
        )
        heaviest = bucket_ms[-1]
        return cfg_by_key[best], heaviest, measured[heaviest][best]

    # fall back to heaviest M alone
    heaviest = bucket_ms[-1]
    table = measured.get(heaviest, {})
    if not table:
        return None, heaviest, None
    ck = min(table, key=table.get)
    return cfg_by_key[ck], heaviest, table[ck]


# ---------------------------------------------------------------------------
# JSON emit (merge with existing)
# ---------------------------------------------------------------------------


def emit_json(out_dir, arch, N, K, results, dry_run) -> str:
    path = os.path.join(out_dir, f"{arch}-GEMM-A16W16-N={N}-K={K}.json")
    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:  # noqa: BLE001
            existing = {}

    for bk, r in results.items():
        existing[bk] = r["config"]

    # 'any' mirrors the heaviest populated bucket so out-of-range M is covered.
    if results:
        heaviest = max(results, key=B.bucket_sort_key)
        existing["any"] = results[heaviest]["config"]

    if dry_run:
        print(f"  [dry-run] would write {path}:\n{json.dumps(existing, indent=2)}")
        return path

    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=4)
        f.write("\n")
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Tune bf16 A16W16 GEMM configs.")
    p.add_argument("--csv", default=_DEFAULT_CSV, help="shapes CSV path")
    p.add_argument(
        "--backend", default="auto", choices=["auto", "gluon", "triton"],
        help="kernel backend (auto: gluon on gfx1250, triton elsewhere)",
    )
    p.add_argument("--out-dir", default=os.path.abspath(_DEFAULT_OUT))
    p.add_argument(
        "--all-m", action="store_true",
        help="tune every distinct M (not just the per-bucket representative)",
    )
    p.add_argument(
        "--exhaustive", action="store_true",
        help="sweep the full valid config grid instead of the curated set",
    )
    p.add_argument(
        "--max-candidates", type=int, default=None,
        help="cap candidates per shape (deterministic thinning)",
    )
    p.add_argument(
        "--only-nk", default=None,
        help="restrict to N,K pairs, e.g. '9216,6144;6144,8192'",
    )
    p.add_argument("--atol", type=float, default=1e-1)
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument(
        "--log", default=None,
        help="checkpoint JSONL path (default: <out-dir>/.tune_a16w16_<arch>.jsonl)",
    )
    p.add_argument("--resume", action="store_true", help="reuse prior measurements from --log")
    p.add_argument("--dry-run", action="store_true", help="don't write JSON configs")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA/ROCm device required", file=sys.stderr)
        return 1

    backend = resolve_backend(args.backend)
    arch = get_arch()
    print(f"arch={arch} backend={backend} exhaustive={args.exhaustive} all_m={args.all_m}")

    rows = read_shapes(args.csv)
    plan = plan_points(rows, args.all_m)

    if args.only_nk:
        want = set()
        for tok in args.only_nk.split(";"):
            n, k = tok.split(",")
            want.add((int(n), int(k)))
        plan = {nk: v for nk, v in plan.items() if nk in want}

    log_path = args.log or os.path.join(
        os.path.abspath(args.out_dir), f".tune_a16w16_{arch}.jsonl"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = ResultLog(log_path, args.resume)

    summary = []
    t0 = time.time()
    try:
        for (N, K), spec in sorted(plan.items()):
            print(f"\n=== N={N} K={K} bias={spec['bias']} "
                  f"buckets={sorted(spec['buckets'], key=B.bucket_sort_key)} ===")
            results = tune_nk(N, K, spec, backend, args, log)
            emit_json(args.out_dir, arch, N, K, results, args.dry_run)
            for bk, r in sorted(results.items(), key=lambda kv: B.bucket_sort_key(kv[0])):
                summary.append((N, K, bk, r["M"], r["ms"], r["tflops"], r["config"]))
    finally:
        log.close()

    print(f"\n===== summary ({time.time()-t0:.1f}s, backend={backend}, arch={arch}) =====")
    print(f"{'N':>7} {'K':>6} {'bucket':>12} {'M':>7} {'ms':>9} {'TFLOPs':>8}  config")
    for N, K, bk, M, ms, tf, cfg in summary:
        print(f"{N:>7} {K:>6} {bk:>12} {M:>7} {ms:>9.4f} {tf:>8.1f}  {json.dumps(cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
