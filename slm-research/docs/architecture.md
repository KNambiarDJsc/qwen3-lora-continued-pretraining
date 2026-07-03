# SLM Fine-Tuning Research Framework — Architecture Specification (FROZEN)

**Status:** Frozen — single source of truth for repository implementation
**Model:** `Qwen/Qwen3-0.6B-Base`
**Task:** Continued pretraining via causal language modeling (next-token prediction). NOT instruction tuning, NOT chat SFT, NOT preference optimization.
**Scope of this document:** Architecture only. Zero implementation code. Implementation begins only after this document is approved.

---

## 1. Repository Structure

```
slm-research/
├── configs/
│   ├── config.yaml                     # Hydra root — composes all defaults
│   ├── model/
│   │   └── qwen3_0.6b_base.yaml
│   ├── data/
│   │   ├── mixture.yaml                # weights, sample counts, sampling strategy
│   │   ├── wikitext.yaml
│   │   ├── openwebtext.yaml
│   │   ├── bookcorpusopen.yaml
│   │   ├── tinystories.yaml
│   │   ├── ag_news.yaml
│   │   ├── cnn_dailymail.yaml
│   │   ├── xsum.yaml
│   │   ├── daily_dialog.yaml
│   │   ├── eli5.yaml
│   │   └── yelp_review_full.yaml
│   ├── lora/
│   │   ├── rank8.yaml
│   │   ├── rank16.yaml
│   │   ├── rank32.yaml
│   │   └── rank64.yaml
│   ├── training/
│   │   ├── default.yaml
│   │   └── colab_t4.yaml               # VRAM-safe batch/accum overrides
│   ├── optimizer/
│   │   ├── paged_adamw_8bit.yaml
│   │   └── adamw.yaml
│   ├── scheduler/
│   │   ├── cosine.yaml
│   │   ├── linear.yaml
│   │   └── constant.yaml
│   ├── evaluation/
│   │   └── default.yaml
│   ├── benchmarking/
│   │   └── default.yaml
│   ├── logging/
│   │   └── wandb.yaml
│   └── inference/
│       └── default.yaml
│
├── src/
│   └── slm_research/
│       ├── data/
│       │   ├── loaders.py              # per-dataset HF loading (streaming/full)
│       │   ├── mixture.py              # weighted interleave sampler
│       │   ├── preprocessing.py        # cleaning, dedup, EOS insertion
│       │   ├── tokenization.py         # tokenizer wrapper, vocab handling
│       │   ├── packing.py              # constant-length sequence packing
│       │   └── collators.py            # attention masks, label construction
│       │
│       ├── modeling/
│       │   ├── model_factory.py        # base model load + device mapping
│       │   ├── lora_factory.py         # PEFT LoRA config application
│       │   └── precision.py            # bf16/fp16/4-bit/8-bit setup
│       │
│       ├── training/
│       │   ├── trainer.py              # core training loop
│       │   ├── checkpointing.py        # save/resume/early-stop
│       │   └── seed.py                 # deterministic seeding
│       │
│       ├── evaluation/
│       │   ├── evaluator.py            # val loop: ppl, bpt, length-bucket ppl
│       │   └── generation.py           # qualitative sample generations
│       │
│       ├── benchmarking/
│       │   ├── latency.py
│       │   ├── throughput.py
│       │   └── memory.py
│       │
│       ├── tracking/
│       │   ├── wandb_logger.py         # sole entry point for all W&B calls
│       │   └── run_metadata.py         # run ID, git hash, system/GPU info
│       │
│       └── utils/
│           ├── logging_utils.py        # structured logging, no print()
│           └── config_schema.py        # pydantic validation of merged config
│
├── scripts/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── infer.py
│   └── cli.py                          # Typer entrypoint, orchestration only
│
├── reports/
│   ├── figures/
│   ├── latex/main.tex
│   └── benchmark_results/
│
├── tests/
│   ├── test_data/
│   ├── test_modeling/
│   ├── test_training/
│   └── test_evaluation/
│
├── docs/
│   ├── architecture.md
│   ├── developer_guide.md
│   ├── training_guide.md
│   ├── benchmark_guide.md
│   ├── dataset_guide.md
│   └── configuration_guide.md
│
├── README.md
├── documentation.md
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

## 2. Folder Responsibilities

| Folder | Single Responsibility |
|---|---|
| `configs/` | Every tunable parameter in the system. Nothing outside this folder should contain a hardcoded hyperparameter. |
| `src/slm_research/data/` | Everything between "raw HF dataset" and "batch ready for the model." No training logic here. |
| `src/slm_research/modeling/` | Everything about constructing the model object (base weights, LoRA adapters, precision). No training-loop logic here. |
| `src/slm_research/training/` | The training loop and its lifecycle (checkpoint, resume, seed). Does not know dataset internals or W&B internals — depends on interfaces from `data/`, `modeling/`, `tracking/`. |
| `src/slm_research/evaluation/` | Post-hoc quality measurement. Depends on `modeling/` and `data/`, independent of `training/`. |
| `src/slm_research/benchmarking/` | Post-hoc performance measurement (latency/throughput/memory). Fully independent of `training/` so benchmark numbers aren't contaminated by training overhead. |
| `src/slm_research/tracking/` | The only code in the repo permitted to call the W&B SDK directly. |
| `src/slm_research/utils/` | Cross-cutting concerns with no domain logic: logging, config validation. |
| `scripts/` | Thin orchestration layer. A script wires configs → modules → execution. No business logic lives here. |
| `reports/` | Output artifacts only — never read by code, only written to. |
| `tests/` | Mirrors `src/` structure 1:1. |
| `docs/` | Human-facing explanation, not machine-read. |

---

## 3. Module Responsibilities

| Module | Responsibility | Depends On |
|---|---|---|
| `data/loaders.py` | Load each raw dataset (streaming or full) per its config | `configs/data/*` |
| `data/preprocessing.py` | Clean text, deduplicate, insert EOS tokens | `loaders.py` output |
| `data/tokenization.py` | Wrap tokenizer, apply to cleaned text | `preprocessing.py` output |
| `data/packing.py` | Pack tokenized sequences to constant length within each dataset | `tokenization.py` output |
| `data/mixture.py` | Interleave packed sequences across datasets per mixture weights | `packing.py` output, `configs/data/mixture.yaml` |
| `data/collators.py` | Build attention masks + labels per batch | `mixture.py` output |
| `modeling/model_factory.py` | Instantiate `Qwen3-0.6B-Base`, apply device map | `configs/model/*` |
| `modeling/precision.py` | Apply bf16/fp16/4-bit/8-bit settings to the loaded model | `model_factory.py` |
| `modeling/lora_factory.py` | Wrap the precision-configured model with PEFT LoRA adapters | `precision.py`, `configs/lora/*` |
| `training/seed.py` | Set all RNG seeds deterministically | `configs/training/*` |
| `training/trainer.py` | Run the training loop: forward, backward, optimizer step, scheduler step, periodic eval trigger | `modeling/*`, `data/*`, `tracking/wandb_logger.py` |
| `training/checkpointing.py` | Save/load/resume checkpoints, early stopping logic | `trainer.py` |
| `evaluation/evaluator.py` | Compute val/token_loss, val/ppl, val/bpt, length-bucket ppl | `modeling/*`, `data/*` |
| `evaluation/generation.py` | Generate qualitative samples for logging | `modeling/*` |
| `benchmarking/latency.py` | Measure per-token and per-sequence inference latency | `modeling/*` |
| `benchmarking/throughput.py` | Measure tokens/sec, samples/sec under load | `modeling/*` |
| `benchmarking/memory.py` | Measure peak/average GPU memory | `modeling/*` |
| `tracking/wandb_logger.py` | Init runs, log metrics/tables/artifacts, tag runs | called by `trainer.py`, `evaluator.py`, `benchmarking/*` |
| `tracking/run_metadata.py` | Generate run ID, capture git hash + system info | called by `wandb_logger.py` at run init |
| `utils/config_schema.py` | Pydantic models validating the merged Hydra config before execution starts | called first by every `scripts/*.py` |

---

## 4. Data Flow

```
Raw HF dataset (per source)
   → loaders.py         [streaming for openwebtext/bookcorpusopen; full load otherwise]
   → preprocessing.py    [clean, dedup, EOS insertion]
   → tokenization.py     [tokenize with Qwen3 tokenizer]
   → packing.py          [constant-length packing, WITHIN each dataset]
   → mixture.py          [weighted interleave ACROSS datasets, per mixture.yaml]
   → collators.py         [attention masks + labels per batch]
   → PyTorch DataLoader
   → training/trainer.py or evaluation/evaluator.py
```

Design decision: packing happens per-dataset before mixing, so `val/ppl_by_length_bucket` reflects genuine per-source sequence-length distributions rather than an averaged mixture artifact.

---

## 5. Training Flow

```
scripts/train.py
   → Hydra composes config.yaml + overrides
   → config_schema.py validates merged config (fail fast)
   → seed.py sets deterministic seeds
   → run_metadata.py generates run ID + captures git hash/system info
   → wandb_logger.py initializes run, logs config as run metadata
   → model_factory.py loads Qwen3-0.6B-Base
   → precision.py applies bf16/fp16/quantization
   → lora_factory.py applies PEFT LoRA adapters
   → data pipeline (Section 4) produces train DataLoader
   → trainer.py runs training loop:
        for each step:
           forward → loss → backward → optimizer.step → scheduler.step
           wandb_logger.log(train/* metrics)
           every N steps: checkpointing.py saves checkpoint
           every M steps: evaluator.py runs, wandb_logger.log(val/* metrics)
   → final checkpoint saved + logged as wandb.Artifact
```

---

## 6. Evaluation Flow

```
scripts/evaluate.py
   → config_schema.py validates config
   → model_factory.py + lora_factory.py load a specific checkpoint
   → data pipeline produces held-out val DataLoader
   → evaluator.py computes: val/token_loss, val/sequence_loss, val/ppl, val/bpt,
     val/ppl_by_length_bucket, val/n_tokens, val/n_sequences
   → generation.py produces sample generations (val/examples)
   → wandb_logger.py logs metrics + generation table + checkpoint ranking
```

Evaluation is checkpoint-driven and callable standalone (not only as a training side-effect), so you can re-evaluate any past checkpoint without retraining.

---

## 7. Configuration Strategy

- **Hydra + OmegaConf** compose `configs/config.yaml` from a `defaults` list referencing one file per group (`model`, `data/mixture`, `lora`, `training`, `optimizer`, `scheduler`, `evaluation`, `benchmarking`, `logging`).
- Overrides happen entirely at the CLI: `python scripts/train.py lora=rank32 training=colab_t4 optimizer=adamw`.
- **`utils/config_schema.py`** defines Pydantic models mirroring each config group. Every `scripts/*.py` entrypoint validates the merged OmegaConf dict against these models *before* touching the GPU — invalid configs fail in milliseconds, not 20 minutes into a Colab run.
- No hyperparameter is ever hardcoded in `src/`. If a value isn't in `configs/`, it doesn't exist.

---

## 8. Experiment Management Strategy

- Every run generates a UUID (`run_metadata.py`), captures the current git commit hash, and freezes a snapshot of the merged config to disk alongside the checkpoint directory.
- Deterministic seeding (`training/seed.py`) is mandatory and logged as `train/seed`.
- Checkpoint naming encodes `{run_id}_{epoch}_{step}` — no ambiguous "latest.pt" checkpoints.
- Resume logic in `checkpointing.py` reconstructs optimizer/scheduler state, not just model weights.

---

## 9. W&B Integration Strategy

- **Single entry point:** `tracking/wandb_logger.py` is the only module permitted to import `wandb`. Everything else calls this wrapper — keeps W&B swappable/mockable in `tests/`.
- **One project** (e.g. `slm-qwen3-0.6b-lora`). Runs tagged with: dataset mixture ID, LoRA rank, precision mode, optimizer — matching the assignment's requirement to tag `dataset`, `model`, `hyperparameters`, `configuration variant`.
- **Metric namespacing** follows the assignment's exact `train/*` and `val/*` prefixes — no renaming, so cross-run comparison in the W&B UI stays consistent.
- Config, system info, GPU info logged as run config at `wandb.init()`.
- Generation samples logged via `wandb.Table` (`val/examples`).
- Checkpoints logged as `wandb.Artifact` for lineage/versioning across the rank/precision sweep.

---

## 10. Benchmarking Architecture

Benchmarking is **fully decoupled from training** — a standalone harness that:

1. Loads one specific `(checkpoint, config)` pair.
2. Runs isolated measurement passes (`latency.py`, `throughput.py`, `memory.py`) with no training-loop overhead in the measurement path.
3. Writes results to `reports/benchmark_results/{run_id}.json`.
4. `scripts/benchmark.py` can aggregate multiple result files into a comparison matrix (rank × batch size × precision × sequence length) for the LaTeX report.

This isolation matters: if benchmarking shared code paths with `trainer.py`, throughput numbers would be contaminated by gradient computation and optimizer overhead, invalidating the inference-speed comparisons the assignment requires.

---

## 11. Dependency Graph

```
configs/  ──validated by──▶  utils/config_schema.py
                                     │
        ┌────────────────────────────┼─────────────────────────────┐
        ▼                            ▼                             ▼
   data/*                      modeling/*                   tracking/*
   (loaders → preprocessing        (model_factory →          (wandb_logger,
    → tokenization → packing        precision →                run_metadata)
    → mixture → collators)          lora_factory)
        │                            │                             │
        └──────────────┬─────────────┘                             │
                        ▼                                          │
                 training/trainer.py  ◀──────logs via───────────────┘
                        │
                        ├──▶ training/checkpointing.py
                        │
                        ▼
              evaluation/evaluator.py  (also standalone-callable)
                        │
                        ▼
            benchmarking/* (standalone, depends only on modeling/* + a saved checkpoint)
```

Key invariant: `benchmarking/*` and `evaluation/*` never import from `training/trainer.py` — they consume checkpoints as artifacts, keeping training, evaluation, and benchmarking as three independently testable subsystems.

---

## 12. Execution Flow (CLI)

```
cli.py commands, in typical lifecycle order:

download-data   → scripts/download_data.py   → data/loaders.py
preprocess      → scripts/preprocess.py      → data/preprocessing.py, tokenization.py, packing.py
train           → scripts/train.py           → Section 5 flow
evaluate        → scripts/evaluate.py        → Section 6 flow
benchmark       → scripts/benchmark.py       → Section 10 flow
infer           → scripts/infer.py           → modeling/* + a checkpoint, single-prompt generation
list-checkpoints→ scripts/cli.py             → reads checkpoint directory metadata
resume          → scripts/train.py --resume  → checkpointing.py resume path
export          → scripts/cli.py             → pulls W&B run history via API for report/plots
```

Sweeps (LoRA rank comparison, precision comparison, batch size comparison) are executed as repeated `train` + `benchmark` invocations with different config overrides — no separate sweep subsystem needed at this scale; Hydra multirun (`-m`) handles it.

---

## Open Decisions Requiring Your Sign-Off

1. Weighted proportional interleaving vs. epoch-balanced mixing (defaulted: proportional).
2. Streaming for `openwebtext` + `bookcorpusopen` only (defaulted: yes).
3. Pack-within-dataset-then-mix vs. pack-across-mixture (defaulted: pack-within-dataset).

If you accept these defaults, this document is frozen as-is. If not, tell me which to flip before Phase 2 (skeleton generation) starts.

---

## Amendment Log

**2026-07-03 — Compute environment changed: Colab T4 → RunPod RTX 4090 (24GB)**

- `configs/training/colab_t4.yaml` renamed to `configs/training/runpod_4090.yaml`.
- `configs/config.yaml` training default updated accordingly.
- Rationale: eliminates Colab free-tier session timeout/disconnect risk under
  a hard deadline; 24GB VRAM removes the tight-budget constraints the T4
  config was designed around.
- **Open follow-on decision (not yet resolved):** whether `optimizer/paged_adamw_8bit.yaml`
  remains the default given the 4090 no longer has the VRAM pressure that
  motivated it. Left unchanged pending a conscious Phase 3 decision — see
  `configs/training/runpod_4090.yaml` comments.
- **New operational constraint introduced by this change:** GPU time now
  costs real money (~$0.34/hr Community Cloud, ~$0.59-0.69/hr Secure Cloud
  as of this check; RunPod also bills idle storage on stopped-not-terminated
  pods). This did not exist under free Colab and should factor into phase
  sequencing — do CPU-only work (preprocessing, config validation) off-pod
  where possible.
