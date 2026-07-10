# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
M -> config-file bucket mapping.

The runtime lookup (``aiter/ops/triton/utils/gemm_config_utils.py``) only
searches ``M_LEQ_x`` / ``M_GEQ_x`` for ``x`` in ``STANDARD_M_BOUNDS`` and then
falls back to ``any``. So those are the only keys a tuned JSON can expose that
production will actually reach. We mirror that here exactly.

Bucketing rule (agreed for the ATOM bf16 shape set):
  * every distinct M lands in ``M_LEQ_<smallest bound >= M>``;
  * M larger than the biggest bound (8192) lands in the open ``M_GEQ_8192``
    bucket (16384/32768 aren't reachable bounds, so they cannot get their own
    key);
  * the representative M tuned for a bucket is the **largest** M observed in it
    (most tiles / most config-sensitive; a safe default for smaller M in the
    same bucket);
  * ``any`` mirrors the heaviest populated bucket so out-of-range M still gets a
    compute-oriented config.
"""

from __future__ import annotations

# Must match STANDARD_M_BOUNDS in gemm_config_utils.py.
STANDARD_M_BOUNDS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
_MAX_BOUND = STANDARD_M_BOUNDS[-1]
LARGE_BUCKET = f"M_GEQ_{_MAX_BOUND}"


def bucket_for_m(m: int) -> str:
    """Return the config-file key that production will look up for this M."""
    for b in STANDARD_M_BOUNDS:
        if m <= b:
            return f"M_LEQ_{b}"
    return LARGE_BUCKET


def bucket_sort_key(bucket: str) -> int:
    """Ordering weight: bigger => heavier M. Used to pick the ``any`` fallback."""
    if bucket == LARGE_BUCKET:
        return _MAX_BOUND * 2
    if bucket.startswith("M_LEQ_"):
        return int(bucket[len("M_LEQ_") :])
    if bucket.startswith("M_GEQ_"):
        return int(bucket[len("M_GEQ_") :]) * 2
    return 0


def group_by_bucket(ms: list[int]) -> dict[str, list[int]]:
    """distinct M list -> {bucket_key: [M, ...]} (sorted M within each bucket)."""
    out: dict[str, list[int]] = {}
    for m in sorted(set(ms)):
        out.setdefault(bucket_for_m(m), []).append(m)
    return out


def representative_m(bucket_ms: list[int]) -> int:
    """Heaviest M in a bucket."""
    return max(bucket_ms)
