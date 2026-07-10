# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Standalone reference + candidate runner for the bf16 A16W16 GEMM.

This extracts the two code paths that ``aiter/tuned_gemm.py`` dispatches to for
plain bf16/fp16 ``Y = X @ W^T`` -- the ``triton_gemm`` path (which itself picks
the gluon kernel on gfx1250 and the triton kernel elsewhere) and the
``torch_gemm`` baseline -- into a small self-contained module the tuner and the
unit test can both call.

Nothing here owns a kernel: ``triton_gemm`` forwards to the real
``aiter.ops.triton.gemm.basic.gemm_a16w16.gemm_a16w16`` so we tune exactly the
kernel that production dispatches to. The only addition over the production
wrapper is a ``config`` override so the tuner can force a specific candidate.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.basic.gemm_a16w16 import (
    gemm_a16w16 as _gemm_a16w16,
    _is_gluon_available,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch


def resolve_backend(backend: Optional[str] = None) -> str:
    """auto -> gluon on gfx1250 (MI400/MI455), triton everywhere else."""
    if backend in (None, "auto"):
        return "gluon" if _is_gluon_available() else "triton"
    backend = backend.lower()
    assert backend in ("triton", "gluon"), f"unknown backend {backend!r}"
    return backend


def make_inputs(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.bfloat16,
    bias: bool = False,
    device: str = "cuda",
    seed: int = 0,
):
    """TN-layout operands: x=(M,K) row-major, w=(N,K) row-major (W^T internally).

    Matches ``op_tests/triton_tests/.../test_gemm_a16w16.py`` so tuned configs
    are validated on the same operand layout production uses.
    """
    torch.manual_seed(seed)
    x = torch.randn((M, K), dtype=dtype, device=device)
    w = torch.randn((N, K), dtype=dtype, device=device)
    bias_t = torch.randn((N,), dtype=dtype, device=device) if bias else None
    return x, w, bias_t


def torch_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Baseline ``Y = X @ W^T (+ bias)``.

    Extracted from ``tuned_gemm.torch_gemm`` reduced to the bf16/fp16 path (the
    fp8/scaled-mm branches don't apply to A16W16). This is the correctness
    oracle the tuner gates every candidate against.
    """
    out = F.linear(x, w, bias)
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype)
    return out


def triton_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    config: Optional[dict] = None,
    backend: Optional[str] = None,
    kernel_type: str = "bandwidth_bound",
    y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the production A16W16 kernel, optionally forcing ``config``.

    ``config=None`` reproduces production dispatch (config resolved from the
    JSON files by shape). ``config=<dict>`` forces a candidate -- for the gluon
    backend the dict carries ``kernel_type`` and it is honored over the arg.
    """
    backend = resolve_backend(backend)
    return _gemm_a16w16(
        x,
        w,
        bias=bias,
        dtype=dtype,
        y=y,
        config=config,
        kernel_type=kernel_type,
        backend=backend,
    )


__all__ = [
    "resolve_backend",
    "make_inputs",
    "torch_gemm",
    "triton_gemm",
    "get_arch",
    "_is_gluon_available",
]
