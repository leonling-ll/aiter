# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""TEMPORARY one-shot instrumentation.

Dumps untuned Triton GEMM / batched-GEMM shapes seen while running the
DeepSeek-R1-0528-MXFP4-MTP-MoEFP4 workload on gfx1250, into
aiter/configs/model_configs/dsr1_*_untuned_gemm.csv so they can be fed to the
tuner later. Disabled unless AITER_DUMP_DSR1_UNTUNED_GEMM_SHAPES=1 is set, so
it is a no-op in normal operation.

Revert: delete this file and the two call sites in
aiter/ops/triton/gemm/basic/gemm_a8w8.py and
aiter/ops/triton/gemm/batched/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py
(``git checkout`` both files, then remove this one).
"""

import csv
import os
import threading
from pathlib import Path

_ENABLED = os.environ.get("AITER_DUMP_DSR1_UNTUNED_GEMM_SHAPES", "0") == "1"
_MODEL_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs" / "model_configs"
_seen = set()
_lock = threading.Lock()


def dump_untuned_shape(csv_name: str, header: list, row: tuple) -> None:
    """Append `row` to aiter/configs/model_configs/<csv_name>, once per
    distinct (csv_name, row) seen in this process. Multiple ranks/processes
    may race and duplicate rows across processes -- dedupe the file
    afterwards (e.g. drop_duplicates in pandas) before using it."""
    if not _ENABLED:
        return
    key = (csv_name, row)
    with _lock:
        if key in _seen:
            return
        _seen.add(key)
        path = _MODEL_CONFIGS_DIR / csv_name
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)
