# A16W16 (bf16) GEMM tuning suite

Standalone tuner + unit test for the plain 16-bit `Y = X @ W^T` GEMM that
`aiter/tuned_gemm.py` dispatches to. It sweeps kernel configs for a set of
`(M, N, K)` shapes, keeps the fastest correct config per M-bucket, and writes the
standard `{arch}-GEMM-A16W16-N={N}-K={K}.json` files that the runtime loads via
`aiter/ops/triton/utils/gemm_config_utils.py`.

The kernel tuned is the real production one
(`aiter.ops.triton.gemm.basic.gemm_a16w16.gemm_a16w16`), which auto-selects the
**gluon** backend on `gfx1250` (MI400 / MI455) and the **triton** backend
elsewhere. The baseline / correctness oracle is torch `F.linear`.

## Quick start

Run everything from the repo root (`cd /home/leling/aiter`). Backend/arch are
auto-detected (gluon on gfx1250, triton elsewhere).

```bash
# 0. shapes CSV to work on (default here uses aiter/configs/bf16_untuned_gemm.csv)
CSV=aiter/configs/bf16_untuned_gemm.csv

# 1. baseline BEFORE tuning: torch vs kernel with arch-default configs only
python -m op_tests.tuning_tests.gemm_a16w16.compare_torch_vs_triton --csv $CSV --force-default -o before.csv

# 2. tune the shapes -> writes {arch}-GEMM-A16W16-N=*-K=*.json into configs/gemm/
python -m op_tests.tuning_tests.gemm_a16w16.tune --csv $CSV

# 3. check every shape resolves to the bucket it was tuned under
python -m op_tests.tuning_tests.gemm_a16w16.verify_buckets --csv $CSV

# 4. validate the emitted configs compile/run and match torch
python -m pytest op_tests/tuning_tests/gemm_a16w16/test_gemm_a16w16_tuned.py -q

# 5. perf AFTER tuning (tuned json where present) — diff against before.csv
python -m op_tests.tuning_tests.gemm_a16w16.compare_torch_vs_triton --csv $CSV -o after.csv
```

Each step is described in detail in its section below.

## Files

| file | purpose |
|---|---|
| `gemm_ref.py` | extracted `triton_gemm` (gluon+triton) and `torch_gemm` baseline; `make_inputs` |
| `search_space.py` | candidate config generation per backend (curated regime-grid + `--exhaustive`) |
| `buckets.py` | M → `M_LEQ_/M_GEQ_/any` bucket mapping (mirrors the runtime lookup) |
| `tune.py` | the tuner CLI (parse CSV → bucket → gate → bench → emit JSON) |
| `test_gemm_a16w16_tuned.py` | UT: production dispatch + every stored bucket config is correct |
| `compare_torch_vs_triton.py` | torch vs triton/gluon perf comparison over a shapes CSV (tuned-if-present-else-default) |
| `verify_buckets.py` | verify each CSV shape resolves to the bucket the tuner keys it under |

## Tune on MI455 (gfx1250, gluon)

```bash
cd /home/leling/aiter
python -m op_tests.tuning_tests.gemm_a16w16.tune \
    --csv /home/leling/ATOM/model_collection/bf16_untuned_collected_gemm.csv
```

This writes `gfx1250-GEMM-A16W16-N=*-K=*.json` into
`aiter/ops/triton/configs/gemm/`. Backend and arch are auto-detected; pass
`--backend gluon` to force it.

Useful flags:

- `--dry-run` — print the JSON that would be written, don't touch disk.
- `--only-nk "9216,6144;6144,8192"` — restrict to specific N,K pairs.
- `--all-m` — tune **every** distinct M (not just the per-bucket representative)
  and pick, per bucket, the config with the best mean-normalized latency across
  the bucket's M. Much longer; use for a deep pass.
- `--exhaustive` — sweep the full valid config grid instead of the curated set.
- `--max-candidates N` — cap candidates/shape (deterministic thinning).
- `--resume` — reuse measurements from the checkpoint log (see below).
- `--warmup/--rep`, `--atol/--rtol` — bench and correctness knobs.

### Buckets

Per `(N, K)`, distinct M are grouped into the only keys the runtime can reach
(`M_LEQ_x`/`M_GEQ_x` for `x` in `STANDARD_M_BOUNDS`, then `any`). Each bucket is
tuned at its **heaviest** M; `any` mirrors the heaviest populated bucket. For the
ATOM bf16 set this yields, per weight shape: `M_LEQ_4/8/16` (tiny/decode),
`M_LEQ_8192` (the 6.5k–8.2k prefill cluster), `M_GEQ_8192` (13k–32k), and `any`.
The vocab shape `(200064, 6144)` only has tiny M, so just `M_LEQ_4/8/16` + `any`.

### Checkpoint / resume

Every measurement is appended to `<out-dir>/.tune_a16w16_<arch>.jsonl`. Re-run
with `--resume` to skip already-measured `(N, K, M, backend, config)` points —
an interrupted run continues where it stopped, and partial progress survives a
push/pull between machines (copy the `.jsonl` along with the repo).

## Validate

```bash
# full UT (production dispatch + every stored bucket config)
python -m pytest op_tests/tuning_tests/gemm_a16w16/test_gemm_a16w16_tuned.py -q

# quick, no pytest
python op_tests/tuning_tests/gemm_a16w16/test_gemm_a16w16_tuned.py --smoke
```

`test_production_dispatch` checks `config=None` (real dispatch) is numerically
correct for the CSV shapes; `test_tuned_configs` forces every stored bucket
config and asserts it compiles, runs, and matches the baseline.

## Compare torch vs triton/gluon

Benchmark the kernel against the torch `F.linear` baseline over a shapes CSV:

```bash
python -m op_tests.tuning_tests.gemm_a16w16.compare_torch_vs_triton
python -m op_tests.tuning_tests.gemm_a16w16.compare_torch_vs_triton \
    --csv aiter/configs/bf16_untuned_gemm.csv --backend auto -o results.csv
```

Every `(M, N, K)` row runs the kernel with `config=None` — i.e. exactly the
production config policy: the specialized `{arch}-GEMM-A16W16-N={N}-K={K}.json`
is used when it exists, otherwise the arch default `{arch}-GEMM-A16W16.json`.
Each shape is labeled **`tuned`** or **`default`** in the `src` column so you can
see which shapes actually pick up your tuned files.

Per shape it prints torch vs triton latency, TFLOPs, `speedup = torch_ms /
triton_ms`, and a correctness check (torch vs kernel); then a summary (geomean
speedup, win count, tuned/default counts, mismatches). `-o` dumps the per-shape
table to CSV.

Flags: `--backend {auto,gluon,triton}`, `--csv PATH`, `--warmup/--rep`,
`--atol/--rtol`, `--no-check`, `-o/--out`, `--force-default`.

**`--force-default`** forces the arch default config for the kernel, ignoring any
tuned `N=K` json. This is the easy A/B for "how much did tuning buy me":

```bash
# without tuned configs (arch default only)
python -m op_tests.tuning_tests.gemm_a16w16.compare_torch_vs_triton --force-default -o before.csv
# with tuned configs (specialized json where present)
python -m op_tests.tuning_tests.gemm_a16w16.compare_torch_vs_triton              -o after.csv
```

Rows run under `--force-default` are labeled `default*` in the `src` column.
(Equivalently, a plain run shows `tuned`/`default` per shape depending on whether
a specialized json exists, so a single default-policy run already tells you which
shapes are picking up tuned configs.)

Note the default CSV here (`aiter/configs/bf16_untuned_gemm.csv`) has a different
N,K set than the ATOM CSV `tune.py` defaults to, so tune against the same CSV you
compare on for the rows to line up.

## Verify shape → bucket → json mapping

Confirm every `(M, N, K)` in a CSV resolves, through the real `get_gemm_config`,
to the bucket key the tuner stores it under:

```bash
# against currently-installed configs (use on MI455 after tuning)
python -m op_tests.tuning_tests.gemm_a16w16.verify_buckets

# arch-independent proof: synthesize tuner-layout json in a temp dir and assert
# each M lands in its own bucket (no tuned files required)
python -m op_tests.tuning_tests.gemm_a16w16.verify_buckets --synthetic
```

`--synthetic` writes specialized files whose buckets are exactly what the tuner
emits for the CSV, points the config loader at them, and checks that each M is
resolved to its bucket (`M_LEQ_x` / `M_GEQ_8192`) with `is_tuned=True`. The
`--live` (default) mode reports, per shape, whether a specialized file exists,
whether the lookup treats it as tuned, and which bucket was selected.

## Notes on developing off-target (e.g. MI350 / gfx950)

Gluon is `gfx1250`-only and import-guarded, so on other archs the suite
auto-falls back to the **triton** backend and emits `gfx950-…`/`gfx942-…` files.
That path is only for exercising the harness end-to-end; the MI455 gluon run is
what produces the `gfx1250-…` configs. Point `--out-dir` at a scratch directory
when smoke-testing so you don't overwrite installed configs.
