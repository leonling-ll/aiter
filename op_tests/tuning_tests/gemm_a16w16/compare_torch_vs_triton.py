# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Compare torch (F.linear) vs the triton/gluon A16W16 kernel over a shapes CSV.

For every (M, N, K) row the kernel runs with ``config=None`` -- i.e. exactly the
production config policy: the specialized ``{arch}-GEMM-A16W16-N={N}-K={K}.json``
is used when it exists, otherwise the arch default ``{arch}-GEMM-A16W16.json``.
Each shape is labeled ``tuned`` or ``default`` accordingly.

Both paths are correctness-checked against each other and benchmarked; the script
prints per-shape latency / TFLOPs / speedup and a summary (geomean speedup,
tuned-vs-default counts, mismatches).

Usage:
    python -m op_tests.tuning_tests.gemm_a16w16.compare_torch_vs_triton
    python -m op_tests.tuning_tests.gemm_a16w16.compare_torch_vs_triton \
        --csv aiter/configs/bf16_untuned_gemm.csv --backend auto -o results.csv
"""

from __future__ import annotations

import argparse
import math
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import triton

from op_tests.tuning_tests.gemm_a16w16.gemm_ref import (
    make_inputs,
    torch_gemm,
    triton_gemm,
    resolve_backend,
    get_arch,
)

_DEFAULT_CSV = os.path.join(_REPO_ROOT, "aiter", "configs", "bf16_untuned_gemm.csv")

_DTYPE_MAP = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _parse_dtype(s: str, default: torch.dtype) -> torch.dtype:
    return _DTYPE_MAP.get((s or "").strip(), default)


def read_rows(path: str) -> list[dict]:
    import csv

    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            in_dtype = _parse_dtype(r.get("dtype"), torch.bfloat16)
            if in_dtype not in (torch.bfloat16, torch.float16):
                continue  # A16W16 is 16-bit input only
            rows.append(
                {
                    "M": int(r["M"]),
                    "N": int(r["N"]),
                    "K": int(r["K"]),
                    "bias": (r.get("bias", "False").strip().lower() == "true"),
                    "in_dtype": in_dtype,
                    "out_dtype": _parse_dtype(r.get("outdtype"), in_dtype),
                }
            )
    return rows


def config_source(N: int, K: int) -> str:
    """'tuned' if a specialized {arch}-...-N=N-K=K.json exists, else 'default'.

    Mirrors what ``config=None`` dispatch will actually load (get_gemm_config's
    is_tuned flag is True only when the specialized N=K file was found).
    """
    try:
        from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

        _cfg, is_tuned = get_gemm_config("GEMM-A16W16", max(N, K), N, K)
        return "tuned" if is_tuned else "default"
    except Exception:  # noqa: BLE001
        return "default"


def forced_default_config(backend: str, M: int, N: int, K: int) -> dict:
    """The arch **default** config for (M,N,K), ignoring any specialized N=K file.

    Passing this explicitly to the kernel reproduces exactly what would run if no
    ``{arch}-GEMM-A16W16-N={N}-K={K}.json`` existed -- get_gemm_config with
    N=K=None only reads the default ``{arch}-GEMM-A16W16.json``. For the triton
    backend the split-K fields are filled in the same way production does on load.
    """
    from aiter.ops.triton.utils.gemm_config_utils import (
        get_gemm_config,
        compute_splitk_params,
    )

    cfg, _ = get_gemm_config("GEMM-A16W16", M)  # N=K=None -> default file only
    if "BLOCK_SIZE_M" in cfg and "SPLITK_BLOCK_SIZE" not in cfg:
        cfg = compute_splitk_params(cfg, K)  # triton schema needs this to launch
    return cfg


def bench(fn, warmup, rep) -> float:
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def tflops(M, N, K, ms) -> float:
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=_DEFAULT_CSV)
    p.add_argument("--backend", default="auto", choices=["auto", "gluon", "triton"])
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("--atol", type=float, default=1e-1)
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--no-check", action="store_true", help="skip correctness check")
    p.add_argument(
        "--force-default", action="store_true",
        help="force the arch default config for the kernel (ignore any tuned "
        "N=K json) -- for A/B comparing perf with vs without tuned configs",
    )
    p.add_argument("-o", "--out", default=None, help="write per-shape results to CSV")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA/ROCm device required", file=sys.stderr)
        return 1

    backend = resolve_backend(args.backend)
    arch = get_arch()
    rows = read_rows(args.csv)
    print(f"arch={arch} backend={backend} shapes={len(rows)} csv={args.csv}\n")

    hdr = (
        f"{'M':>6} {'N':>7} {'K':>6} {'src':>7} {'odt':>7} "
        f"{'torch_ms':>9} {'trit_ms':>9} {'speedup':>8} "
        f"{'torch_TF':>8} {'trit_TF':>8} {'chk':>5}"
    )
    print(hdr)
    print("-" * len(hdr))

    results = []
    speedups = []
    n_ok = n_bad = n_err = 0
    for r in rows:
        M, N, K = r["M"], r["N"], r["K"]
        src = "default*" if args.force_default else config_source(N, K)
        odt = str(r["out_dtype"]).replace("torch.", "")
        try:
            x, w, bias = make_inputs(
                M, N, K, dtype=r["in_dtype"], bias=r["bias"]
            )
            otype = r["out_dtype"]
            # config=None -> production policy (tuned-if-present-else-default);
            # --force-default -> always the arch default, ignoring tuned json.
            use_cfg = forced_default_config(backend, M, N, K) if args.force_default else None
            use_kt = (
                use_cfg.get("kernel_type", "bandwidth_bound") if use_cfg else "bandwidth_bound"
            )

            def run_torch():
                return torch_gemm(x, w, bias, otype)

            def run_triton():
                return triton_gemm(
                    x, w, bias=bias, dtype=otype, config=use_cfg,
                    backend=backend, kernel_type=use_kt,
                )

            chk = "skip"
            if not args.no_check:
                ref = run_torch()
                out = run_triton()
                try:
                    torch.testing.assert_close(
                        out, ref, atol=args.atol, rtol=args.rtol
                    )
                    chk = "ok"
                    n_ok += 1
                except AssertionError:
                    chk = "FAIL"
                    n_bad += 1

            t_ms = bench(run_torch, args.warmup, args.rep)
            k_ms = bench(run_triton, args.warmup, args.rep)
            spd = t_ms / k_ms if k_ms > 0 else float("nan")
            speedups.append(spd)
            print(
                f"{M:>6} {N:>7} {K:>6} {src:>7} {odt:>7} "
                f"{t_ms:>9.4f} {k_ms:>9.4f} {spd:>8.2f} "
                f"{tflops(M,N,K,t_ms):>8.1f} {tflops(M,N,K,k_ms):>8.1f} {chk:>5}"
            )
            results.append(
                {
                    "M": M, "N": N, "K": K, "src": src, "out_dtype": odt,
                    "torch_ms": t_ms, "triton_ms": k_ms, "speedup": spd,
                    "torch_tflops": tflops(M, N, K, t_ms),
                    "triton_tflops": tflops(M, N, K, k_ms),
                    "check": chk,
                }
            )
            del x, w, bias
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"{M:>6} {N:>7} {K:>6} {src:>7} {odt:>7}  ERROR: {type(e).__name__}: {str(e)[:80]}")

    # summary
    print("\n===== summary =====")
    if speedups:
        geo = math.exp(sum(math.log(s) for s in speedups if s > 0) / len(speedups))
        wins = sum(1 for s in speedups if s > 1.0)
        print(f"shapes benchmarked : {len(speedups)}")
        print(f"geomean speedup    : {geo:.3f}x  (triton/gluon vs torch)")
        print(f"triton faster on   : {wins}/{len(speedups)} shapes")
        print(f"min / max speedup  : {min(speedups):.2f}x / {max(speedups):.2f}x")
    n_tuned = sum(1 for r in results if r["src"] == "tuned")
    print(f"config source      : {n_tuned} tuned, {len(results)-n_tuned} default")
    if not args.no_check:
        print(f"correctness        : {n_ok} ok, {n_bad} FAIL")
    if n_err:
        print(f"errors             : {n_err}")

    if args.out and results:
        import csv as _csv

        with open(args.out, "w", newline="") as f:
            wtr = _csv.DictWriter(f, fieldnames=list(results[0].keys()))
            wtr.writeheader()
            wtr.writerows(results)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
