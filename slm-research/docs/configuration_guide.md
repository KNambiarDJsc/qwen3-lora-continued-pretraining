# Configuration Guide

## How it fits together

Two layers, in order:

1. **Hydra + OmegaConf** compose `configs/config.yaml` from one file per group (see below), merged
   with any CLI overrides, into a `DictConfig`.
2. **`utils/config_schema.py`** validates that merged `DictConfig` (via
   `OmegaConf.to_container(cfg, resolve=True)` → `validate_config(...)`) against Pydantic models,
   producing a typed `RootConfig`. Every `scripts/*.py` entrypoint does this *before* touching a
   GPU, dataset, or W&B — an invalid config fails in milliseconds with a field-level Pydantic
   error, not 20 minutes into a run.

No hyperparameter is ever hardcoded in `src/`. If a value isn't in `configs/`, it doesn't exist.

## Config groups

| Group | Default | Schema | Notes |
|---|---|---|---|
| `model` | `qwen3_0.6b_base` | `ModelConfig` | Base checkpoint identity, tokenizer, precision/device hints |
| `data` | `mixture` | `MixtureConfig` | The 8-dataset mixture — see the [Dataset Guide](dataset_guide.md) |
| `lora` | `rank16` | `LoRAConfig` | `rank8` / `rank16` / `rank32` / `rank64` |
| `training` | `runpod_4090` | `TrainingConfig` | Batch size, precision, gradient checkpointing, step intervals |
| `optimizer` | `paged_adamw_8bit` | `OptimizerConfig` | or `adamw` |
| `scheduler` | `cosine` | `SchedulerConfig` | or `linear` / `constant` |
| `evaluation` | `default` | `EvaluationConfig` | Val split size, length buckets, generation sample count |
| `benchmarking` | `default` | `BenchmarkingConfig` | Sweep grid for latency/throughput/memory |
| `logging` | `wandb` | `LoggingConfig` | W&B project/entity/tags |
| `inference` | `default` | `InferenceConfig` | Decoding params for `scripts/infer.py` |

Override any group wholesale, or any leaf value, at the CLI:

```bash
python scripts/train.py lora=rank32 training=runpod_4090 optimizer=adamw
python scripts/train.py training.per_device_train_batch_size=16 optimizer.lr=1e-4
```

Two root-level keys don't belong to a group and aren't in `RootConfig` at all — they're consumed
and popped by their script before validation:

- `checkpoint_path` — required by `scripts/evaluate.py` and `scripts/benchmark.py` via
  `+checkpoint_path=<dir>` (the `+` is needed because Hydra doesn't know about a brand-new key).
- `checkpoint` / `prompt` — required by `scripts/infer.py`. These *are* declared in
  `configs/config.yaml` (as `null` defaults), specifically so they can be set with plain
  `checkpoint=... prompt="..."` — no `+` needed — matching how you'd naturally invoke inference.

## Key fields by group

**`ModelConfig`** — `name` (HF repo id), `revision`, `trust_remote_code`, `torch_dtype`
(`auto`/`bf16`/`fp16`/`fp32`), `device_map` (default `"auto"`), `tokenizer.name`.

**`LoRAConfig`** — `r` (rank, one of 8/16/32/64), `alpha` (all four rank configs use `alpha = 2 * r`),
`dropout` (`0.05`), `bias` (`none`/`all`/`lora_only`), `target_modules` (all four rank configs
target `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` — the full attention +
MLP projection set).

**`TrainingConfig`** — `precision` (`bf16`/`fp16`/`4bit`/`8bit` — resolved by `modeling/precision.py`),
`gradient_checkpointing`, `per_device_train_batch_size`, `gradient_accumulation_steps`,
`eval_steps` / `save_steps` / `logging_steps`, `warmup_ratio`, `max_grad_norm`,
`dataloader_num_workers`. `training/default.yaml` holds environment-agnostic defaults;
`training/runpod_4090.yaml` extends it with hardware-specific overrides (see its comments for the
RunPod cost/VRAM rationale — it superseded a Colab T4 config, logged in `architecture.md`'s
Amendment Log).

**`OptimizerConfig`** — `name` (`paged_adamw_8bit`/`adamw`), `lr`, `betas`, `eps`, `weight_decay`.

**`SchedulerConfig`** — `name` (`cosine`/`linear`/`constant`), `num_warmup_steps` /
`num_training_steps` — both `null` in every scheduler config; they're derived at runtime from
`warmup_ratio` and dataset size / batch size rather than hardcoded.

**`EvaluationConfig`** — `val_split_fraction` (`0.02`), `length_buckets`
(`[128, 256, 512, 1024, 2048]` — interpreted as *position ranges within a packed sequence*, not
document lengths; see `Evaluator._build_buckets`), `num_generation_samples` (`8`),
`generation_max_new_tokens` (`128`).

**`BenchmarkingConfig`** — `batch_sizes_to_test`, `sequence_lengths_to_test` (the grid a single
`scripts/benchmark.py` run sweeps), plus `precision_modes_to_test` / `lora_ranks_to_test`, which
are *not* swept in-process — they describe the axes of the comparison matrix that
`+aggregate=true` builds from multiple prior single-checkpoint runs (each at a different
precision/rank, via a repeated `train` + `benchmark` invocation). See the
[Benchmark Guide](benchmark_guide.md).

**`LoggingConfig`** — `project`, `entity` (set your W&B entity — currently `null`), `tags`
(runtime-populated with LoRA rank/precision/optimizer — see `tracking/wandb_logger.py`),
`log_generation_table`, `log_checkpoints_as_artifacts`.

**`InferenceConfig`** — `max_new_tokens`, `temperature`, `top_p`, `do_sample`. When
`do_sample=false`, `temperature`/`top_p` are ignored (greedy decoding) — see
`inference/generator.py`.

## Validation errors

`validate_config()` raises `pydantic.ValidationError` with the offending field path and message.
Common ones:

- `lora.r` must be exactly one of `8, 16, 32, 64` (not an arbitrary int) — there's a config file
  per valid rank; there's no partial-rank override path.
- `training.precision` / `benchmarking.precision_modes_to_test` must be one of
  `bf16, fp16, 4bit, 8bit`.
- `evaluation.val_split_fraction` must satisfy `0 < x < 1`.
- `benchmarking.num_warmup_iterations` must be `>= 0`; `num_measured_iterations` must be `> 0`.
