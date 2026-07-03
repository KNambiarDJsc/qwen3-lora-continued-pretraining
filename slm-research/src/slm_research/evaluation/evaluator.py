"""Validation-loop metric computation.

Responsibility: compute val/token_loss, val/sequence_loss, val/ppl, val/bpt,
val/ppl_by_length_bucket, val/n_tokens, val/n_sequences. Callable standalone
(not only as a training side-effect) — see architecture spec Section 6.

Depends on: modeling/*, data/*
Consumed by: training/trainer.py (periodic), scripts/evaluate.py (standalone)
"""

from typing import Any


class Evaluator:
    """Runs the held-out validation pass and computes reporting metrics."""

    def __init__(self, model: Any, val_dataloader: Any, config: dict[str, Any]) -> None:
        raise NotImplementedError("Phase 7: wire up evaluator state.")

    def evaluate(self) -> dict[str, float]:
        """Run the full validation pass, return a dict of val/* metrics.

        Raises:
            NotImplementedError: implementation lands in Phase 7 (Evaluation).
        """
        raise NotImplementedError("Phase 7: implement validation metric computation.")
