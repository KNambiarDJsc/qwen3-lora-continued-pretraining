# Documentation Index

- `docs/architecture.md` — frozen architecture spec, single source of truth for module boundaries
  and dependency rules. A few implementation-time deviations from it are called out where they
  occur (e.g. the `fineweb_edu`/`yelp_review_full` dataset swap, the CLI's subprocess-based
  delegation) rather than silently rolled back into the spec.
- `docs/developer_guide.md` — setup, layout, running tests, code style, how the CLI delegates,
  known gaps.
- `docs/training_guide.md` — running training/resume/early stopping, evaluation, benchmarking,
  and inference end to end.
- `docs/benchmark_guide.md` — single-checkpoint benchmarking and the multi-run comparison matrix.
- `docs/dataset_guide.md` — the 8-dataset mixture, where dataset identity actually lives, the
  data pipeline.
- `docs/configuration_guide.md` — Hydra config groups, override syntax, Pydantic validation.
