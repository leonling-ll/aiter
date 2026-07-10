# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Standalone UT for the bf16 A16W16 GEMM tuning suite.

Two things are checked:

1. ``test_production_dispatch`` -- for the CSV shapes, the production entry point
   (``config=None`` -> config resolved from the JSON files by shape, gluon on
   gfx1250 / triton elsewhere) produces numerically correct output vs a torch
   ``F.linear`` baseline. This is the end-to-end correctness of whatever configs
   are currently installed (tuned or default).

2. ``test_tuned_configs`` -- if a tuned ``{arch}-GEMM-A16W16-N=..-K=..json``
   exists, every stored bucket config is explicitly forced and must compile, run,
   and match the baseline at that bucket's representative M. This catches a tuned
   JSON that was written for the wrong arch or with a config the kernel rejects.

Run everything under pytest, or ``python test_gemm_a16w16_tuned.py --smoke`` for a
quick subset without pytest.
"""

from __future__ import annotations

import json
import os
import sys

# Allow running this file directly (python test_...py --smoke) as well as under
# pytest / -m: put the repo root (4 levels up) on sys.path.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

from op_tests.tuning_tests.gemm_a16w16 import buckets as B
from op_tests.tuning_tests.gemm_a16w16.gemm_ref import (
    make_inputs,
    torch_gemm,
    triton_gemm,
    resolve_backend,
    get_arch,
)
from op_tests.tuning_tests.gemm_a16w16.tune import read_shapes, plan_points

_CSV = os.environ.get(
    "AITER_A16W16_CSV",
    "/home/leling/ATOM/model_collection/bf16_untuned_collected_gemm.csv",
)
_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "aiter", "ops", "triton", "configs", "gemm"
)

_ATOL = 1e-1
_RTOL = 1e-2


def _rep_shapes() -> list[tuple[int, int, int]]:
    """One (M,N,K) per (bucket, N,K): the representative-M plan points."""
    if not os.path.exists(_CSV):
        return []
    plan = plan_points(read_shapes(_CSV), all_m=False)
    shapes = []
    for (N, K), spec in plan.items():
        for _bk, ms in spec["buckets"].items():
            shapes.append((ms[0], N, K))
    return sorted(set(shapes))


_REP_SHAPES = _rep_shapes()


@pytest.mark.skipif(not _REP_SHAPES, reason=f"shapes CSV not found: {_CSV}")
@pytest.mark.parametrize("M,N,K", _REP_SHAPES)
def test_production_dispatch(M, N, K):
    """config=None -> production dispatch must match the torch baseline."""
    dtype = torch.bfloat16
    x, w, bias = make_inputs(M, N, K, dtype=dtype, bias=False)
    ref = torch_gemm(x, w, bias, dtype)
    out = triton_gemm(x, w, bias=bias, dtype=dtype, config=None)
    torch.testing.assert_close(out, ref, atol=_ATOL, rtol=_RTOL)


def _tuned_files() -> list[str]:
    arch = get_arch()
    out = []
    if not os.path.isdir(_CONFIG_DIR):
        return out
    for fn in os.listdir(_CONFIG_DIR):
        if fn.startswith(f"{arch}-GEMM-A16W16-N=") and fn.endswith(".json"):
            out.append(os.path.join(_CONFIG_DIR, fn))
    return sorted(out)


def _tuned_cases() -> list[tuple]:
    """(path, N, K, bucket, M, config) for every stored bucket in tuned files."""
    cases = []
    for path in _tuned_files():
        base = os.path.basename(path)
        try:
            nk = base.split("-N=")[1].rsplit(".json", 1)[0]
            n_s, k_s = nk.split("-K=")
            N, K = int(n_s), int(k_s)
        except (IndexError, ValueError):
            continue
        with open(path) as f:
            cfgs = json.load(f)
        for bucket, cfg in cfgs.items():
            if not isinstance(cfg, dict):
                continue
            M = _bucket_probe_m(bucket)
            cases.append((base, N, K, bucket, M, cfg))
    return cases


def _bucket_probe_m(bucket: str) -> int:
    """A representative M that falls inside the bucket, for probing its config."""
    if bucket.startswith("M_LEQ_"):
        return int(bucket[len("M_LEQ_") :])
    if bucket.startswith("M_GEQ_"):
        return int(bucket[len("M_GEQ_") :]) * 2
    return 8192  # 'any'


_TUNED_CASES = _tuned_cases()


def _prepare_launch_config(cfg: dict, K: int) -> dict:
    """Augment a stored config exactly as production does before launch.

    Triton-schema configs (BLOCK_SIZE_*) get SPLITK_BLOCK_SIZE / defaults filled
    in by compute_splitk_params (production does this in ``_get_config``); gluon
    configs (BLOCK_*) are launched as-is.
    """
    cfg = dict(cfg)
    if "BLOCK_SIZE_M" in cfg and "SPLITK_BLOCK_SIZE" not in cfg:
        from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params

        cfg = compute_splitk_params(cfg, K)
    return cfg


@pytest.mark.skipif(not _TUNED_CASES, reason="no tuned A16W16 config files for this arch")
@pytest.mark.parametrize(
    "fname,N,K,bucket,M,cfg",
    _TUNED_CASES,
    ids=[f"{c[0]}:{c[3]}" for c in _TUNED_CASES],
)
def test_tuned_configs(fname, N, K, bucket, M, cfg):
    """Every stored bucket config must compile, run, and be correct."""
    dtype = torch.bfloat16
    x, w, bias = make_inputs(M, N, K, dtype=dtype, bias=False)
    ref = torch_gemm(x, w, bias, dtype)
    out = triton_gemm(
        x, w, bias=bias, dtype=dtype, config=_prepare_launch_config(cfg, K),
        kernel_type=cfg.get("kernel_type", "bandwidth_bound"),
    )
    torch.testing.assert_close(out, ref, atol=_ATOL, rtol=_RTOL)


def _smoke():
    backend = resolve_backend(None)
    print(f"smoke: arch={get_arch()} backend={backend}")
    shapes = _REP_SHAPES[:6] or [(16, 9216, 6144), (8183, 6144, 8192)]
    ok = 0
    for M, N, K in shapes:
        dtype = torch.bfloat16
        x, w, bias = make_inputs(M, N, K, dtype=dtype, bias=False)
        ref = torch_gemm(x, w, bias, dtype)
        out = triton_gemm(x, w, bias=bias, dtype=dtype, config=None)
        torch.testing.assert_close(out, ref, atol=_ATOL, rtol=_RTOL)
        print(f"  ok M={M} N={N} K={K}")
        ok += 1
    print(f"smoke passed: {ok}/{len(shapes)}")


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        _smoke()
    else:
        raise SystemExit(pytest.main([__file__, "-v"]))
