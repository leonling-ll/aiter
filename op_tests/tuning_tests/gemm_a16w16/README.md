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

## Files

| file | purpose |
|---|---|
| `gemm_ref.py` | extracted `triton_gemm` (gluon+triton) and `torch_gemm` baseline; `make_inputs` |
| `search_space.py` | candidate config generation per backend (curated regime-grid + `--exhaustive`) |
| `buckets.py` | M → `M_LEQ_/M_GEQ_/any` bucket mapping (mirrors the runtime lookup) |
| `tune.py` | the tuner CLI (parse CSV → bucket → gate → bench → emit JSON) |
| `test_gemm_a16w16_tuned.py` | UT: production dispatch + every stored bucket config is correct |

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

## Notes on developing off-target (e.g. MI350 / gfx950)

Gluon is `gfx1250`-only and import-guarded, so on other archs the suite
auto-falls back to the **triton** backend and emits `gfx950-…`/`gfx942-…` files.
That path is only for exercising the harness end-to-end; the MI455 gluon run is
what produces the `gfx1250-…` configs. Point `--out-dir` at a scratch directory
when smoke-testing so you don't overwrite installed configs.
