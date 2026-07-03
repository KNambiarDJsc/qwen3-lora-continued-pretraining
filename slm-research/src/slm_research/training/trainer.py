"""Core training loop.

Responsibility: forward -> loss -> backward -> optimizer step -> scheduler
step, with periodic checkpointing and periodic evaluation triggers.
Per architecture spec Section 5. Does NOT know dataset internals or W&B
internals directly — depends on interfaces from data/, modeling/, tracking/.

Depends on: modeling/*, data/*, tracking/wandb_logger.py, training/checkpointing.py
Consumed by: scripts/train.py
"""

from typing import Any


class Trainer:
    """Owns the training loop lifecycle for one run."""

    def __init__(
        self,
        model: Any,
        train_dataloader: Any,
        optimizer: Any,
        scheduler: Any,
        config: dict[str, Any],
    ) -> None:
        raise NotImplementedError("Phase 6: wire up trainer state.")

    def train(self) -> None:
        """Run the full training loop per configs/training/*.yaml.

        Raises:
            NotImplementedError: implementation lands in Phase 6.
        """
        raise NotImplementedError("Phase 6: implement the training loop.")

    def training_step(self, batch: dict[str, Any]) -> float:
        """Run a single forward/backward/optimizer step, return the loss."""
        raise NotImplementedError("Phase 6: implement a single training step.")
