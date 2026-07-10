# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Candidate config generation for the A16W16 tuner.

Two schemas, one per backend:

* gluon (gfx1250 / MI400 / MI455) -- the primary target. Keys consumed by
  ``gemm_a16w16_``'s gluon branch: BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS,
  num_warps, kernel_type. ``num_warps`` must be a power of two (the kernel takes
  ``log2(num_warps)`` in ``create_wmma_layouts``). The WMMA instr shape is
  16x16x32, so block dims are multiples of 16 and BLOCK_K a multiple of 32.

* triton (gfx942 / gfx950 / ...) -- for local development/smoke only. Reuses the
  known-valid arch default entries and varies a couple of fields, so every emitted
  dict is guaranteed to be a legal kernel launch.

Validity pruning mirrors the pipeline-depth rules the kernel enforces at launch
(see ``gemm_a16w16.py``): bandwidth_bound needs ``num_k_tiles >= NUM_BUFFERS``;
compute_bound needs ``num_k_tiles >= NUM_BUFFERS + 2``. Candidates that can't
satisfy their depth are dropped here rather than silently downgraded at runtime.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import triton

# ---------------------------------------------------------------------------
# gluon (gfx1250) search space
# ---------------------------------------------------------------------------

_GLUON_BLOCK_M = (16, 32, 64, 128, 256)
_GLUON_BLOCK_N = (16, 32, 64, 128, 256)
_GLUON_BLOCK_K = (64, 128, 256, 512)
_GLUON_NUM_BUFFERS = (2, 3, 4, 5)
_GLUON_NUM_WARPS = (2, 4, 8)
_GLUON_KERNEL_TYPES = ("bandwidth_bound", "compute_bound")

# Curated search is a per-M-regime sub-grid rather than a fixed tuple list: the
# knobs that matter differ by M. Skinny M (decode) wants BLOCK_M=16 and explores
# BLOCK_N/BLOCK_K/pipeline depth under a bandwidth_bound kernel; large M
# (prefill) wants big square tiles and compute_bound. Each regime is a small
# Cartesian grid, then filtered per-shape for validity (depth/warp/LDS).
#
# Regime -> dict of value lists for (BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS,
# num_warps, kernel_type).
_GLUON_REGIMES = {
    "skinny": dict(  # M <= 16
        BLOCK_M=(16,),
        BLOCK_N=(32, 64, 128),
        BLOCK_K=(128, 256, 512),
        NUM_BUFFERS=(2, 3, 4),
        num_warps=(2, 4),
        kernel_type=("bandwidth_bound",),
    ),
    "small": dict(  # 16 < M <= 128
        BLOCK_M=(16, 32, 64),
        BLOCK_N=(32, 64, 128),
        BLOCK_K=(128, 256, 512),
        NUM_BUFFERS=(2, 3, 4),
        num_warps=(2, 4),
        kernel_type=("bandwidth_bound", "compute_bound"),
    ),
    "large": dict(  # M > 128
        BLOCK_M=(64, 128, 256),
        BLOCK_N=(64, 128, 256),
        BLOCK_K=(64, 128, 256),
        NUM_BUFFERS=(2, 3, 4),
        num_warps=(4, 8),
        kernel_type=("bandwidth_bound", "compute_bound"),
    ),
}


def _gluon_regime(M: int) -> str:
    if M <= 16:
        return "skinny"
    if M <= 128:
        return "small"
    return "large"

# Depth requirements, keyed by kernel_type (from gemm_a16w16.py:170-171).
_MIN_BUFFERS = {"bandwidth_bound": 1, "compute_bound": 2}
_DEPTH_SLACK = {"bandwidth_bound": 0, "compute_bound": 2}


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1)).bit_length()


def _gluon_depth_ok(K: int, block_k: int, num_buffers: int, kernel_type: str) -> bool:
    num_k_tiles = triton.cdiv(K, block_k)
    depth_cap = num_k_tiles - _DEPTH_SLACK[kernel_type]
    return depth_cap >= _MIN_BUFFERS[kernel_type] and num_buffers <= depth_cap


def _lds_bytes(block_m, block_n, block_k, num_buffers, elem_bytes=2) -> int:
    """Rough shared-memory footprint: NUM_BUFFERS deep A and B tiles.

    Used only to prune obviously-too-large tiles before we ever hit the
    compiler. Padding/atom overheads make this approximate, so keep the bound
    loose (the compiler is the real gate; failures are caught + skipped anyway).
    """
    return num_buffers * (block_m * block_k + block_k * block_n) * elem_bytes


# Loose upper bound on the LDS footprint used only to drop absurd tiles before
# the compiler sees them. Deliberately generous: the estimate ignores padding
# and the exact multi-buffer layout, and the real gate is the compiler (invalid
# launches are caught + skipped at bench time). Under-pruning is the safe error
# -- over-pruning would silently hide good candidates from the MI455 sweep.
_LDS_BUDGET = 160 * 1024


def gluon_candidates(
    M: int,
    N: int,
    K: int,
    exhaustive: bool = False,
    max_candidates: Optional[int] = None,
) -> list[dict]:
    """Valid gluon candidate configs for one (M, N, K).

    Curated mode prunes by shape (skip block dims far larger than the problem,
    keep the search focused); exhaustive mode keeps every depth/LDS-valid point
    in the grid.
    """
    m_cap = _next_pow2(max(M, 1))
    n_cap = _next_pow2(max(N, 1))

    def _valid(bm, bn, bk, nb, nw, kt) -> bool:
        if not _gluon_depth_ok(K, bk, nb, kt):
            return False
        if _lds_bytes(bm, bn, bk, nb) > _LDS_BUDGET:
            return False
        # need at least one 16x16 WMMA block per warp in the tile
        if (bm // 16) * (bn // 16) < nw:
            return False
        return True

    def _shape_ok(bm, bn) -> bool:
        # Don't tile far past the problem (wasted lanes); allow one step over.
        if bm > 16 and bm > 2 * m_cap:
            return False
        if bn > max(16, n_cap):
            return False
        return True

    out: list[dict] = []
    seen = set()

    def _add(bm, bn, bk, nb, nw, kt):
        cfg = {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_K": bk,
            "NUM_BUFFERS": nb,
            "num_warps": nw,
            "kernel_type": kt,
        }
        key = tuple(sorted(cfg.items()))
        if key in seen:
            return
        seen.add(key)
        out.append(cfg)

    if exhaustive:
        grids = [
            (
                _GLUON_BLOCK_M,
                _GLUON_BLOCK_N,
                _GLUON_BLOCK_K,
                _GLUON_NUM_BUFFERS,
                _GLUON_NUM_WARPS,
                _GLUON_KERNEL_TYPES,
            )
        ]
    else:
        r = _GLUON_REGIMES[_gluon_regime(M)]
        grids = [
            (
                r["BLOCK_M"],
                r["BLOCK_N"],
                r["BLOCK_K"],
                r["NUM_BUFFERS"],
                r["num_warps"],
                r["kernel_type"],
            )
        ]

    for BMs, BNs, BKs, NBs, NWs, KTs in grids:
        for kt in KTs:
            for bm in BMs:
                for bn in BNs:
                    if not exhaustive and not _shape_ok(bm, bn):
                        continue
                    for bk in BKs:
                        for nb in NBs:
                            for nw in NWs:
                                if _valid(bm, bn, bk, nb, nw, kt):
                                    _add(bm, bn, bk, nb, nw, kt)

    if not out:
        # Guaranteed-legal fallback: smallest tile, shallow pipeline.
        kt = "bandwidth_bound"
        nb = min(
            _GLUON_NUM_BUFFERS,
            key=lambda b: (not _gluon_depth_ok(K, 64, b, kt), b),
        )
        out.append(
            {
                "BLOCK_M": 16,
                "BLOCK_N": 16,
                "BLOCK_K": 64,
                "NUM_BUFFERS": nb,
                "num_warps": 2,
                "kernel_type": kt,
            }
        )

    if max_candidates is not None and len(out) > max_candidates:
        # Deterministic thinning: keep an evenly-spaced subset.
        step = len(out) / max_candidates
        out = [out[int(i * step)] for i in range(max_candidates)]
    return out


# ---------------------------------------------------------------------------
# triton search space (local dev / smoke on gfx942, gfx950, ...)
# ---------------------------------------------------------------------------


def _triton_base_configs() -> list[dict]:
    """A small hand-picked set of legal triton entries (schema-complete).

    We keep these minimal and known-valid; the tuner still correctness-gates
    and benchmarks each, and invalid launches are skipped. NUM_KSPLIT/
    SPLITK_BLOCK_SIZE are filled in per-K by ``compute_splitk_params``.
    """
    base = []
    for bm in (16, 32, 64, 128):
        for bn in (32, 64, 128, 256):
            for bk in (64, 128, 256):
                base.append(
                    {
                        "BLOCK_SIZE_M": bm,
                        "BLOCK_SIZE_N": bn,
                        "BLOCK_SIZE_K": bk,
                        "GROUP_SIZE_M": 1,
                        "num_warps": 4,
                        "num_stages": 2,
                        "waves_per_eu": 2,
                        "matrix_instr_nonkdim": 16,
                        "cache_modifier": None,
                        "kpack": 1,
                        "NUM_KSPLIT": 1,
                    }
                )
    return base


def triton_candidates(
    M: int,
    N: int,
    K: int,
    exhaustive: bool = False,
    max_candidates: Optional[int] = None,
) -> list[dict]:
    from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params

    m_cap = _next_pow2(max(M, 1))
    out: list[dict] = []
    for cfg in _triton_base_configs():
        if not exhaustive and cfg["BLOCK_SIZE_M"] > 2 * m_cap and cfg["BLOCK_SIZE_M"] > 16:
            continue
        cfg = compute_splitk_params(copy.deepcopy(cfg), K)
        out.append(cfg)
    if max_candidates is not None and len(out) > max_candidates:
        step = len(out) / max_candidates
        out = [out[int(i * step)] for i in range(max_candidates)]
    return out


def candidates_for(
    backend: str,
    M: int,
    N: int,
    K: int,
    exhaustive: bool = False,
    max_candidates: Optional[int] = None,
) -> list[dict]:
    if backend == "gluon":
        return gluon_candidates(M, N, K, exhaustive, max_candidates)
    return triton_candidates(M, N, K, exhaustive, max_candidates)


# Keys the runtime actually stores/reads per backend (drop tuner-internal extras
# before writing JSON).
GLUON_CONFIG_KEYS = (
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "NUM_BUFFERS",
    "num_warps",
    "kernel_type",
)
TRITON_CONFIG_KEYS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "cache_modifier",
    "kpack",
    "NUM_KSPLIT",
    # Needed when a config is forced directly (production recomputes it from K
    # via compute_splitk_params on load, so keeping it is harmless).
    "SPLITK_BLOCK_SIZE",
)


def clean_config(backend: str, cfg: dict) -> dict:
    keys = GLUON_CONFIG_KEYS if backend == "gluon" else TRITON_CONFIG_KEYS
    return {k: cfg[k] for k in keys if k in cfg}
