# Benchmark Guide

## Why it's a separate subsystem

Benchmarking is fully decoupled from `training/trainer.py` — it loads one specific
`(checkpoint, config)` pair and runs inference-only measurement passes. If it shared code paths
with the training loop, throughput/latency numbers would be contaminated by gradient computation
and optimizer overhead, invalidating the inference-speed comparisons the numbers exist to produce.
`benchmarking/*` (like `evaluation/*`) never imports `training/trainer.py` — it consumes a saved
checkpoint as an artifact.

## Single-checkpoint run

```bash
python scripts/benchmark.py +checkpoint_path=outputs/checkpoints/run_abc123_epoch0_step500
```

Loads that checkpoint, then runs three measurement passes, each sweeping the
`batch_sizes_to_test × sequence_lengths_to_test` grid from `configs/benchmarking/default.yaml`
(default: batch sizes `[1, 4, 8, 16]`, sequence lengths `[128, 512, 1024]`, 5 warmup iterations,
20 measured iterations per grid point):

| Module | Metrics produced |
|---|---|
| `benchmarking/latency.py` | `benchmark/latency_ms_per_sequence/bs{B}_sl{S}`, `benchmark/latency_ms_per_token/bs{B}_sl{S}` — timed per forward pass |
| `benchmarking/throughput.py` | `benchmark/tokens_per_sec/bs{B}_sl{S}`, `benchmark/samples_per_sec/bs{B}_sl{S}` — sustained rate across all measured iterations |
| `benchmarking/memory.py` | `benchmark/peak_memory_mb/bs{B}_sl{S}`, `benchmark/avg_memory_mb/bs{B}_sl{S}` — `0.0` on CPU (no CUDA allocator to query) |

Results are logged to the checkpoint's original W&B run (resumed, same as `scripts/evaluate.py`)
and written to `reports/benchmark_results/{run_id}.json`:

```json
{
  "run_id": "abc123",
  "checkpoint_path": "outputs/checkpoints/run_abc123_epoch0_step500",
  "model_name": "Qwen/Qwen3-0.6B-Base",
  "lora_rank": 16,
  "precision": "bf16",
  "git_hash": "a1b2c3d",
  "metrics": { "benchmark/latency_ms_per_token/bs1_sl128": 1.83, "...": "..." }
}
```

## Comparing across rank / precision / batch size / sequence length

`precision_modes_to_test` and `lora_ranks_to_test` in `BenchmarkingConfig` are **not** swept
in-process by a single run — a checkpoint is already trained at one fixed rank and precision.
Instead, run a separate `train` + `benchmark` pair per point on that sweep (Hydra multirun works
fine here too), then aggregate every result file that's accumulated in
`reports/benchmark_results/`:

```bash
python scripts/benchmark.py +aggregate=true
```

This scans every `{run_id}.json` under `reports/benchmark_results/` (skipping
`comparison_matrix.json` itself if present), parses each `benchmark/{metric}/bs{B}_sl{S}` key back
into its grid coordinates, and writes `reports/benchmark_results/comparison_matrix.json`: one row
per `(run_id, lora_rank, precision, batch_size, sequence_length)` combination with all metrics for
that point flattened alongside it — the rank × batch size × precision × sequence length matrix
the LaTeX report (`reports/latex/main.tex`) consumes.

Aggregation logic lives directly in `scripts/benchmark.py`, not under `src/slm_research/`. That's
an explicit exception in the architecture spec (Section 10): there's no dedicated `src/` module
for the aggregation step.

## Via the Typer CLI

```bash
python scripts/cli.py benchmark +checkpoint_path=outputs/checkpoints/run_abc123_epoch0_step500
python scripts/cli.py benchmark +aggregate=true
```

`cli.py benchmark` forwards its arguments verbatim to `scripts/benchmark.py` as Hydra overrides —
see the [Developer Guide](developer_guide.md#cli) for how that delegation works.
