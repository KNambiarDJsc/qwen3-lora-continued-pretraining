"""Checkpoint save / resume / early-stopping logic.

Responsibility: persist and restore training state, including optimizer and
scheduler state (not just model weights) — see architecture spec Section 8.
Checkpoint naming: {run_id}_{epoch}_{step}, no ambiguous "latest.pt".

Depends on: training/trainer.py (called from within the training loop)
Consumed by: training/trainer.py, evaluation/evaluator.py, benchmarking/*
"""

from typing import Any


def save_checkpoint(state: dict[str, Any], run_id: str, epoch: int, step: int) -> str:
    """Save model + optimizer + scheduler state to a uniquely named checkpoint.

    Raises:
        NotImplementedError: implementation lands in Phase 6.
    """
    raise NotImplementedError("Phase 6: implement checkpoint saving.")


def load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """Load a checkpoint for resume, evaluation, or benchmarking."""
    raise NotImplementedError("Phase 6: implement checkpoint loading.")


def should_early_stop(val_history: list[float], patience: int) -> bool:
    """Determine whether validation loss has plateaued past the patience window."""
    raise NotImplementedError("Phase 6: implement early-stopping check.")
