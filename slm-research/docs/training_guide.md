# Training Guide

## Data preparation (optional)

Neither of these is required before `train` — `DataModule` builds the pipeline on demand,
in-process, every run. They exist for warming a cache ahead of time; see the
[Dataset Guide](dataset_guide.md#standalone-data-scripts) for what each actually does.

```bash
python scripts/cli.py download-data   # warm the local HF cache for every mixture source
python scripts/cli.py preprocess      # run load->clean->tokenize->pack per source, save to disk
```

## Quickstart

```bash
python scripts/train.py
```

With defaults (`lora=rank16`, `training=runpod_4090`, `optimizer=paged_adamw_8bit`,
`scheduler=cosine`), this: validates the merged config, seeds every RNG
(`training/seed.py`), generates a run ID + captures git hash/system info, initializes a W&B run,
builds the 8-dataset mixture (see the [Dataset Guide](dataset_guide.md)), loads
`Qwen/Qwen3-0.6B-Base` with LoRA adapters applied, and runs the training loop.

Override anything at the CLI — see the [Configuration Guide](configuration_guide.md) for the full
group list:

```bash
python scripts/train.py lora=rank32 training=runpod_4090 optimizer=adamw
```

Or through the Typer CLI, which forwards extra args verbatim as Hydra overrides:

```bash
python scripts/cli.py train lora=rank32
```

## What happens each step

`training/trainer.py::SLMTrainer.train()` — the loop, per optimizer step:

```
forward → loss → backward → grad clip (max_grad_norm) → optimizer.step() → scheduler.step()
```

Logged every `training.logging_steps` steps (`train/*` namespace):
`train/loss`, `train/ppl`, `train/lr`, `train/grad_norm`, `train/throughput_tokens_per_sec`,
`train/step_time_sec`, `train/epoch`.

Every `training.eval_steps` steps, `evaluation/evaluator.py::Evaluator.evaluate()` runs a full
validation pass and logs `val/*` metrics (see [Evaluation](#evaluation) below) — the *same*
`Evaluator` instance `scripts/evaluate.py` uses standalone, so periodic in-loop numbers and a
later re-evaluation of the saved checkpoint are computed identically.

Every `training.save_steps` steps, and once more at the end of training,
`training/checkpointing.py::save_checkpoint()` writes:

```
outputs/checkpoints/{run_id}_epoch{N}_step{M}/
  adapter_model.safetensors   # LoRA adapter weights only — base model is never re-saved
  adapter_config.json
  training_state.pt           # optimizer state, scheduler state, global_step, epoch, run_id
```

and (if `logging.log_checkpoints_as_artifacts`) logs it as a versioned `wandb.Artifact`.

## Resuming

```bash
python scripts/train.py +resume_from=outputs/checkpoints/run_abc123_epoch0_step500
```

Restores model weights (via `modeling/model_factory.load_model_from_checkpoint`), optimizer and
scheduler state, `global_step`, and `epoch` — and resumes logging to the *same* W&B run (its
`run_id` is read back out of `training_state.pt`), not a new one. Via the Typer CLI, `resume` is
an alias for this exact flow:

```bash
python scripts/cli.py resume +resume_from=outputs/checkpoints/run_abc123_epoch0_step500
```

## Early stopping

`SLMTrainer(..., early_stop_patience=0)` is off by default in `scripts/train.py`. Set it > 0 to
stop when `val/loss` hasn't improved for that many consecutive evaluations
(`training/checkpointing.should_early_stop`).

## Precision

`training.precision` ∈ `bf16` / `fp16` / `4bit` / `8bit`, resolved by `modeling/precision.py`:
bf16/fp16 set `torch_dtype` at load time; 4bit/8bit build a `BitsAndBytesConfig` (NF4 double
quantization for 4bit) and route through `peft.prepare_model_for_kbit_training` before LoRA is
applied. Attention implementation auto-detects flash-attn-2 → SDPA → eager, in that priority order.

## Evaluation

```bash
python scripts/evaluate.py +checkpoint_path=outputs/checkpoints/run_abc123_epoch0_step500
```

Standalone, checkpoint-driven re-evaluation — no retraining, resumes the checkpoint's original
W&B run. Computes the full `val/*` suite: `val/loss` (= `val/token_loss`), `val/sequence_loss`,
`val/ppl`, `val/bpt`, `val/n_tokens`, `val/n_sequences`, and
`val/ppl_by_length_bucket/{start}-{end}` for each range in `evaluation.length_buckets` — these are
**position ranges within a packed sequence** (e.g. `1024-2048` = perplexity over the back half of
the context window), answering "how does prediction quality change as the model accumulates more
context," not "perplexity by document length." Also generates `evaluation.num_generation_samples`
qualitative completions from held-out prompts, logged as a `wandb.Table` (`val/examples`).

## Benchmarking and inference

Covered in the [Benchmark Guide](benchmark_guide.md) (`scripts/benchmark.py` — latency/throughput/
memory, fully decoupled from the training loop so numbers aren't contaminated by gradient/optimizer
overhead) and below.

### Single-prompt inference

```bash
python scripts/infer.py checkpoint=outputs/checkpoints/run_abc123_epoch0_step500 \
    prompt="The future of AI is"
```

Pipeline (`inference/loader.py` → `inference/generator.py`): load tokenizer → load base model →
load the LoRA adapter → `merge_and_unload()` (collapses the adapter into the base weights for a
plain forward pass, skipped for 4bit/8bit checkpoints since BitsAndBytes layers can't absorb a
LoRA delta in place) → generate → pretty-printed output.

Decoding is controlled by `configs/inference/default.yaml` (`max_new_tokens`, `temperature`,
`top_p`, `do_sample`) — override at the CLI, e.g. for greedy decoding:

```bash
python scripts/infer.py checkpoint=... prompt="..." inference.do_sample=false
```

## Sweeps

There's no dedicated sweep subsystem. A LoRA rank / precision / batch-size sweep is just repeated
`train` + `benchmark` invocations with different overrides, or Hydra's built-in multirun:

```bash
python scripts/train.py -m lora=rank8,rank16,rank32,rank64
```

## Listing checkpoints

```bash
python scripts/cli.py list-checkpoints --output-dir outputs
```

Reads every `{run_id}_epoch{N}_step{M}/training_state.pt` under `outputs/checkpoints/`, sorted by
step, printing `run_id`, `epoch`, and `step` for each.

## Exporting run history for the report

```bash
python scripts/cli.py export --project slm-qwen3-0.6b-lora
python scripts/cli.py export --project slm-qwen3-0.6b-lora --run-id abc123 --output reports/run_abc123.json
```

Pulls each run's config, summary, and full metric history via `wandb.Api()`
(`tracking/wandb_logger.py::export_run_history` — the only W&B-API call outside an active run's
lifecycle) and writes it to JSON for the LaTeX report (`reports/latex/main.tex`) or ad-hoc
plotting. Omit `--run-id` to export every run in the project.
