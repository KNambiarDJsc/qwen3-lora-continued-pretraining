# Developer Guide

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Requires Python >= 3.12 (`pyproject.toml`). The editable install is what makes `import
slm_research` resolve for both `scripts/*.py` and `tests/*` — without it you'll get
`ModuleNotFoundError: No module named 'slm_research'`.

## Layout

```
scripts/            Thin Hydra/Typer entrypoints — no business logic, see below
src/slm_research/    All actual implementation, one subpackage per responsibility
tests/               Mirrors src/ 1:1, plus tests/test_scripts/ for cli.py
configs/             Every tunable parameter — see the Configuration Guide
docs/                This directory
reports/             Output artifacts only (benchmark JSON, LaTeX, figures) — never read by code
```

See `docs/architecture.md` for the full annotated tree and the module dependency graph (Section
11) — it's the frozen source of truth for how subsystems are allowed to depend on each other. The
short version: `data/`, `modeling/`, and `tracking/` are the shared foundation; `training/`,
`evaluation/`, and `benchmarking/` each depend on that foundation but never on each other
(`evaluation/*` and `benchmarking/*` specifically never import `training/trainer.py` — they
consume checkpoints as artifacts, which is what keeps them independently testable and keeps
benchmark numbers uncontaminated by training overhead).

## Running tests

```bash
pytest
```

56 tests across:

| Directory | Tests | Coverage |
|---|---|---|
| `test_scripts/` | 15 | `cli.py` command routing (subprocess mocked), `list-checkpoints`, `export`, `download_data.py`/`preprocess.py` (via `hydra.compose` + `cfg_passthrough`, network calls mocked) |
| `test_data/` | 8 | `build_mixture`'s Dataset/IterableDataset type normalization, `packing.py`'s shared output schema |
| `test_evaluation/` | 8 | `Evaluator` (hand-computed cross-entropy checks), `generate_samples` |
| `test_benchmarking/` | 8 | latency/throughput/memory grid coverage + CPU-zero fallback |
| `test_inference/` | 7 | `generate()`/`format_generation()`, `load_model_for_inference()` merge logic |
| `test_utils/` | 5 | `logging_utils.get_logger()` idempotency/handler behavior |
| `test_tracking/` | 3 | `export_run_history()` against a mocked `wandb.Api` |
| `test_modeling/`, `test_training/` | 1 each | placeholder stubs only |

`modeling/` and `training/` are fully implemented but still only have a placeholder
`test_placeholder` in their test directories — real unit test coverage for those two is the
largest concrete gap in the test suite right now, not the modules themselves. (`data/` had the
same gap until `build_mixture`'s type-mismatch bug — see below — made it worth closing.)

Hydra-decorated scripts (`download_data.py`, `preprocess.py`, and by the same pattern `train.py` /
`evaluate.py` / `benchmark.py` / `infer.py` if you add tests for them) are testable without
touching `sys.argv`: `@hydra.main`'s wrapper accepts a `cfg_passthrough` `DictConfig` directly, and
`hydra.compose()` (inside `hydra.initialize_config_dir(...)`) builds the exact same merged config a
real invocation would from the real `configs/` tree. See `tests/test_scripts/test_preprocess.py`
for the pattern — real config validation and pipeline wiring, with only the network-touching calls
(`load_source_splits`, `load_tokenizer`) mocked.

Tests generally build a small fake model/tokenizer (see `test_evaluation/test_evaluator.py`'s
`FakeCausalLM`, which produces analytically-predictable cross-entropy so exact `val/loss`/`val/ppl`
values can be asserted) rather than loading real weights — keeps the suite fast and independent of
network/GPU access. Follow that pattern for new tests: prefer a fake with a legible, deterministic
forward/generate over mocking deep into a real HF model.

## Code style

`ruff` and `black`, both configured for `line-length = 100`, `target-version` py312
(`pyproject.toml`). `mypy` is configured with `disallow_untyped_defs = true` — type hints
everywhere, Google-style docstrings, `pathlib.Path` over string paths, no wildcard imports.

## The CLI

`scripts/cli.py` is a Typer app. Every command except `list-checkpoints` and `export` is a thin
wrapper that runs the matching `scripts/*.py` file **as a subprocess**, forwarding whatever
followed the subcommand as Hydra overrides:

```bash
python scripts/cli.py train lora=rank32
python scripts/cli.py infer checkpoint=outputs/checkpoints/run_abc prompt="hi"
```

This is subprocess dispatch rather than in-process delegation deliberately: each `scripts/*.py`
entrypoint is a `@hydra.main` app that resolves its config directory relative to its own file and
expects to run as `__main__`. Importing the script and calling its `main()` from within `cli.py`
breaks that — Hydra determines the config search path from the task function's `__module__`, and
a module loaded via `importlib` (rather than run as `__main__`) falls into a different, broken
resolution branch (`Primary config module 'configs' not found`). Running the exact command Hydra
already supports, in a fresh process, sidesteps that entirely and also sidesteps Hydra's
`GlobalHydra` singleton state, which doesn't reset cleanly across multiple `hydra.main` calls in
one interpreter.

`list-checkpoints` and `export` are the two exceptions the architecture spec attributes directly
to `cli.py` (no dedicated `scripts/*.py` file for either — see Section 12).
`list-checkpoints` is a thin wrapper over `training/checkpointing.list_checkpoints`.
`export` pulls run config/summary/history via `wandb.Api()` and writes it to a JSON file:

```bash
python scripts/cli.py export --project slm-qwen3-0.6b-lora --run-id abc123
python scripts/cli.py export --project slm-qwen3-0.6b-lora  # every run in the project
```

The actual `wandb.Api()` calls live in `tracking/wandb_logger.py::export_run_history` — not a new
module — since `wandb_logger.py` is the sole module permitted to import `wandb` (architecture spec
Section 9). This is implemented against wandb's documented public API but hasn't been exercised
against a live W&B account in this environment (no credentials here); `test_tracking/` covers it
against a mocked `wandb.Api`.

## Known gaps

- `configs/data/{name}.yaml` per-source files (other than `fineweb_edu.yaml`) still carry
  `hf_path: null` "TODO(phase 4)" placeholders — this is expected, not a bug: dataset identity
  actually lives in `data/registry.py`, and these files are unread leftovers from an earlier design
  pass. See the [Dataset Guide](dataset_guide.md#where-dataset-identity-actually-lives).
- `modeling/` and `training/` have no real unit tests yet (see above).
- `scripts/cli.py export` hasn't been run against a live W&B account (see above) — the request
  shape follows wandb's documented API, but there's no substitute for a real end-to-end check.
