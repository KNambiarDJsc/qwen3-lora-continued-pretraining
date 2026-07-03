# SLM Fine-Tuning Research Framework

Continued pretraining (causal LM, next-token prediction) of `Qwen/Qwen3-0.6B-Base`
via PEFT LoRA, on a curated 10-dataset language modeling mixture.

**Status:** Phase 2 (Repository Skeleton) complete. No implementation yet —
see `docs/architecture.md` for the frozen spec and phase roadmap.

## Quickstart (not yet functional — skeleton only)

```bash
pip install -r requirements.txt
python scripts/cli.py --help
```

## Repository Layout

See `docs/architecture.md` Section 1 for the full annotated tree.

## Phase Roadmap

0. Research Specification — done
1. Software Architecture — done
2. Repository Skeleton — done (this commit)
3. Configuration System — next
4. Data Pipeline
5. Model Pipeline
6. Training
7. Evaluation
8. Benchmarking
9. Documentation

## License

TODO
