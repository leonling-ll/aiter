# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Verify that every (M, N, K) shape in a CSV resolves to the config-file bucket the
tuner keys it under -- i.e. the tuner's emit layout and the runtime lookup agree.

Two modes:

* ``--live`` (default): drive the *real* ``get_gemm_config`` against whatever
  ``{arch}-GEMM-A16W16*.json`` files are installed. For each shape it reports
  whether a specialized N=K file exists, whether the lookup reports it as tuned,
  and which bucket key was selected. Use this on MI455 after tuning to confirm
  the shapes pick up the tuned buckets.

* ``--synthetic``: isolate the config path to a temp dir, synthesize specialized
  files whose buckets are exactly what the tuner would emit for this CSV, then
  drive the real ``get_gemm_config`` and assert every M lands in its own bucket.
  This proves the mapping end-to-end on any arch (no tuned files required).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from op_tests.tuning_tests.gemm_a16w16 import buckets as B
from op_tests.tuning_tests.gemm_a16w16.gemm_ref import get_arch
from op_tests.tuning_tests.gemm_a16w16.tune import read_shapes

_DEFAULT_CSV = os.path.join(_REPO_ROOT, "aiter", "configs", "bf16_untuned_gemm.csv")


def emitted_buckets(ms: list[int]) -> dict[str, int]:
    """The tuner's representative-mode buckets for a set of M: {bucket: rep_M}."""
    return {bk: B.representative_m(v) for bk, v in B.group_by_bucket(ms).items()}


def select_bucket(keys, M: int):
    """Replicate get_gemm_config's key selection given the available bucket keys.

    M_LEQ_x in increasing bound order, then M_GEQ_x in decreasing order, then
    'any'. Mirrors gemm_config_utils._get_gemm_config_cached exactly.
    """
    keys = set(keys)
    for b in B.STANDARD_M_BOUNDS:
        k = f"M_LEQ_{b}"
        if M <= b and k in keys:
            return k
    for b in reversed(B.STANDARD_M_BOUNDS):
        k = f"M_GEQ_{b}"
        if M >= b and k in keys:
            return k
    if "any" in keys:
        return "any"
    return None


# ---------------------------------------------------------------------------
# live mode
# ---------------------------------------------------------------------------


def run_live(rows, arch, cfg_dir):
    from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

    default_path = os.path.join(cfg_dir, f"{arch}-GEMM-A16W16.json")
    default_keys = []
    if os.path.exists(default_path):
        default_keys = list(json.load(open(default_path)).keys())

    hdr = f"{'M':>7} {'N':>7} {'K':>6} {'spec?':>6} {'is_tuned':>8} {'selected':>12} {'tuner_key':>12} {'match':>6}"
    print(hdr)
    print("-" * len(hdr))
    n_match = n_tuned = n_total = 0
    for r in sorted(rows, key=lambda r: (r["N"], r["K"], r["M"])):
        M, N, K = r["M"], r["N"], r["K"]
        spec_path = os.path.join(cfg_dir, f"{arch}-GEMM-A16W16-N={N}-K={K}.json")
        spec = os.path.exists(spec_path)
        keys = list(json.load(open(spec_path)).keys()) if spec else default_keys
        selected = select_bucket(keys, M)
        tuner_key = B.bucket_for_m(M)  # where the tuner would store this M
        _cfg, is_tuned = get_gemm_config("GEMM-A16W16", M, N, K)
        # When a specialized file exists, the selected key should equal the
        # tuner's key for this M (or fall to 'any' if that bucket wasn't tuned).
        match = (selected == tuner_key) if spec else (selected is not None)
        n_total += 1
        n_tuned += int(is_tuned)
        n_match += int(bool(match))
        print(
            f"{M:>7} {N:>7} {K:>6} {('yes' if spec else 'no'):>6} "
            f"{str(bool(is_tuned)):>8} {str(selected):>12} {tuner_key:>12} "
            f"{('OK' if match else 'x'):>6}"
        )
    print(f"\nshapes={n_total} tuned={n_tuned} default={n_total-n_tuned} "
          f"bucket-consistent={n_match}/{n_total}")
    return n_match == n_total


# ---------------------------------------------------------------------------
# synthetic mode
# ---------------------------------------------------------------------------


def run_synthetic(rows, arch):
    import aiter.ops.triton.utils.gemm_config_utils as gc

    # group CSV M by (N,K)
    by_nk: dict[tuple, list[int]] = {}
    for r in rows:
        by_nk.setdefault((r["N"], r["K"]), []).append(r["M"])

    tmp = tempfile.mkdtemp(prefix="a16w16_verify_")
    gem = os.path.join(tmp, "gemm")
    os.makedirs(gem)

    # required default file (tagged)
    with open(os.path.join(gem, f"{arch}-GEMM-A16W16.json"), "w") as f:
        json.dump({"any": {"bucket": "any", "_default": True}}, f)

    # specialized files: exactly the tuner's emitted buckets, each tagged
    for (N, K), ms in by_nk.items():
        eb = emitted_buckets(ms)
        cfgs = {bk: {"bucket": bk, "rep_M": rep} for bk, rep in eb.items()}
        heaviest = max(eb, key=B.bucket_sort_key)
        cfgs["any"] = {"bucket": heaviest, "rep_M": eb[heaviest]}  # 'any' mirrors heaviest
        with open(os.path.join(gem, f"{arch}-GEMM-A16W16-N={N}-K={K}.json"), "w") as f:
            json.dump(cfgs, f)

    saved = gc.AITER_TRITON_CONFIGS_PATH
    ok = True
    try:
        gc.AITER_TRITON_CONFIGS_PATH = tmp
        gc._get_gemm_config_cached.cache_clear()
        if hasattr(gc._get_gemm_config_cached, "_config_cache"):
            del gc._get_gemm_config_cached._config_cache

        hdr = f"{'M':>7} {'N':>7} {'K':>6} {'expected':>12} {'resolved':>12} {'is_tuned':>8} {'match':>6}"
        print(hdr)
        print("-" * len(hdr))
        n_match = 0
        rows_sorted = sorted(rows, key=lambda r: (r["N"], r["K"], r["M"]))
        for r in rows_sorted:
            M, N, K = r["M"], r["N"], r["K"]
            cfg, is_tuned = gc.get_gemm_config("GEMM-A16W16", M, N, K)
            resolved = cfg.get("bucket")
            expected = B.bucket_for_m(M)
            m = (resolved == expected) and bool(is_tuned)
            n_match += int(m)
            ok = ok and m
            print(
                f"{M:>7} {N:>7} {K:>6} {expected:>12} {str(resolved):>12} "
                f"{str(bool(is_tuned)):>8} {('OK' if m else 'x'):>6}"
            )
        print(f"\nshapes={len(rows_sorted)} bucket-consistent={n_match}/{len(rows_sorted)}")
    finally:
        gc.AITER_TRITON_CONFIGS_PATH = saved
        gc._get_gemm_config_cached.cache_clear()
        if hasattr(gc._get_gemm_config_cached, "_config_cache"):
            del gc._get_gemm_config_cached._config_cache
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=_DEFAULT_CSV)
    p.add_argument(
        "--synthetic", action="store_true",
        help="synthesize tuner-layout config files in a temp dir and assert each M "
        "resolves to its bucket (proves mapping on any arch)",
    )
    args = p.parse_args(argv)

    arch = get_arch()
    rows = read_shapes(args.csv)
    cfg_dir = os.path.join(
        _REPO_ROOT, "aiter", "ops", "triton", "configs", "gemm"
    )
    print(f"arch={arch} csv={args.csv} shapes={len(rows)} "
          f"mode={'synthetic' if args.synthetic else 'live'}\n")

    ok = run_synthetic(rows, arch) if args.synthetic else run_live(rows, arch, cfg_dir)
    print("\nRESULT:", "PASS" if ok else "MISMATCH")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
