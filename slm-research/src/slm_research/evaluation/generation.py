"""Qualitative sample generation for logging (val/examples).

Depends on: modeling/*
Consumed by: evaluation/evaluator.py, tracking/wandb_logger.py (as wandb.Table)
"""

from typing import Any


def generate_samples(model: Any, prompts: list[str], inference_config: dict[str, Any]) -> list[str]:
    """Generate sample completions for qualitative inspection.

    Raises:
        NotImplementedError: implementation lands in Phase 7.
    """
    raise NotImplementedError("Phase 7: implement sample generation.")
