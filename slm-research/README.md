# SLM Fine-Tuning Research Framework

Continued pretraining (causal LM, next-token prediction) of `Qwen/Qwen3-0.6B-Base`
via PEFT LoRA, on a curated 10-dataset language modeling mixture.

**Status:** All planned phases implemented and every entrypoint is wired up — training, evaluation,
benchmarking, inference, data download/preprocessing, W&B export, and the CLI are functional end
to end. See `docs/architecture.md` for the frozen spec and `documentation.md` for the full guide
index. A short list of known gaps (missing unit tests in two modules, some dead leftover config
files) is tracked in `docs/developer_guide.md#known-gaps` rather than hidden.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .
python scripts/cli.py --help
```

```bash
# (Optional) warm the HF cache / pre-materialize a packed dataset cache to disk
python scripts/cli.py download-data
python scripts/cli.py preprocess

# Train (LoRA rank 16, bf16, the full 10-dataset mixture)
python scripts/cli.py train

# Evaluate a saved checkpoint standalone
python scripts/cli.py evaluate +checkpoint_path=outputs/checkpoints/run_abc123_epoch0_step500

# Benchmark it (latency/throughput/memory)
python scripts/cli.py benchmark +checkpoint_path=outputs/checkpoints/run_abc123_epoch0_step500

# Generate from it
python scripts/cli.py infer checkpoint=outputs/checkpoints/run_abc123_epoch0_step500 \
    prompt="The future of AI is"

# Pull W&B run history for the report
python scripts/cli.py export --project slm-qwen3-0.6b-lora
```

See the [Training Guide](docs/training_guide.md) for the full walkthrough (resume, early stopping,
sweeps) and the [Configuration Guide](docs/configuration_guide.md) for every override group.

## Repository Layout

See `docs/architecture.md` Section 1 for the full annotated tree.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — frozen architecture spec
- [`docs/developer_guide.md`](docs/developer_guide.md) — setup, tests, code style, CLI internals, known gaps
- [`docs/training_guide.md`](docs/training_guide.md) — train / resume / evaluate / benchmark / infer
- [`docs/benchmark_guide.md`](docs/benchmark_guide.md) — latency/throughput/memory measurement + comparison matrix
- [`docs/dataset_guide.md`](docs/dataset_guide.md) — the 10-dataset mixture and data pipeline
- [`docs/configuration_guide.md`](docs/configuration_guide.md) — Hydra config groups and validation

## Phase Roadmap

0. Research Specification — done
1. Software Architecture — done
2. Repository Skeleton — done
3. Configuration System — done
4. Data Pipeline — done
5. Model Pipeline — done
6. Training — done
7. Evaluation — done
8. Benchmarking — done
9. Documentation — done

Inference (`src/slm_research/inference/`, `scripts/infer.py`) and CLI wiring (`scripts/cli.py`)
were added as additional modules alongside these phases; see `docs/training_guide.md` for usage.

## License

TODO
