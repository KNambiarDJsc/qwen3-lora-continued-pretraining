"""Sole entry point for all Weights & Biases calls in this repository.

Responsibility: init runs, log train/* and val/* metrics, log generation
tables, log checkpoint artifacts, tag runs with dataset/lora rank/precision/
optimizer. No other module in this repo may import wandb directly — see
architecture spec Section 9.

Depends on: tracking/run_metadata.py
Consumed by: training/trainer.py, evaluation/evaluator.py, benchmarking/*
"""

from typing import Any


class WandbLogger:
    """Wraps the wandb SDK so it can be swapped/mocked in tests/."""

    def __init__(self, logging_config: dict[str, Any], run_metadata: dict[str, Any]) -> None:
        raise NotImplementedError("Phase 6: implement wandb.init() wrapper.")

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log a dict of train/* or val/* metrics at a given step."""
        raise NotImplementedError("Phase 6: implement metric logging.")

    def log_generation_table(self, prompts: list[str], completions: list[str]) -> None:
        """Log qualitative generations as a wandb.Table (val/examples)."""
        raise NotImplementedError("Phase 7: implement generation table logging.")

    def log_checkpoint_artifact(self, checkpoint_path: str, run_id: str) -> None:
        """Log a checkpoint as a versioned wandb.Artifact."""
        raise NotImplementedError("Phase 6: implement checkpoint artifact logging.")
