"""Checkpoint save, load, resume, and early-stopping utilities.

Checkpoint naming: {run_id}_epoch{epoch}_step{step}  — no ambiguous "latest".
Each checkpoint directory holds:
  adapter_model.safetensors  — PEFT LoRA adapter weights
  adapter_config.json        — PEFT adapter config (rank, alpha, modules …)
  training_state.pt          — optimizer, scheduler, global_step, epoch, run_id

The base model weights are NOT stored — only the adapter delta. The base
checkpoint is always re-loaded from the HF Hub on resume, which avoids
duplicating multi-GB weights across every checkpoint.

Depends on: PEFT, torch
Consumed by: training/trainer.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

logger = logging.getLogger(__name__)

_TRAINING_STATE_FILENAME = "training_state.pt"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: PeftModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    epoch: int,
    run_id: str,
    output_dir: str | Path,
    accelerator: Any | None = None,
) -> Path:
    """Persist adapter weights and training state.

    Args:
        model: PeftModel — only adapter weights are saved.
        optimizer: Current optimizer instance.
        scheduler: Current LR scheduler instance.
        global_step: Global step counter at save time.
        epoch: Current epoch (0-indexed).
        run_id: 8-char run identifier.
        output_dir: Root directory under which checkpoints/ lives.
        accelerator: Optional Accelerator; if provided, the model is unwrapped
            before saving so that distributed/AMP wrappers are stripped.

    Returns:
        Path to the created checkpoint directory.
    """
    checkpoint_name = f"{run_id}_epoch{epoch}_step{global_step}"
    checkpoint_dir = Path(output_dir) / "checkpoints" / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap model from accelerate wrapper if present
    save_model = accelerator.unwrap_model(model) if accelerator is not None else model
    save_model.save_pretrained(checkpoint_dir)

    # Training state — keeps optimizer and scheduler so we can resume exactly
    torch.save(
        {
            "global_step": global_step,
            "epoch": epoch,
            "run_id": run_id,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        checkpoint_dir / _TRAINING_STATE_FILENAME,
    )

    logger.info("Checkpoint saved → %s", checkpoint_dir)
    return checkpoint_dir


# ---------------------------------------------------------------------------
# Load / Resume
# ---------------------------------------------------------------------------

def load_training_state(
    checkpoint_dir: str | Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: str | torch.device = "cpu",
) -> tuple[int, int, str]:
    """Restore optimizer and scheduler state from a checkpoint.

    The model weights must be loaded separately (via load_model_from_checkpoint
    in model_factory.py) before calling this function.

    Args:
        checkpoint_dir: Path to a checkpoint directory.
        optimizer: Optimizer to load state into.
        scheduler: LR scheduler to load state into.
        device: Device to map saved tensors to.

    Returns:
        (global_step, epoch, run_id) tuple.
    """
    state_path = Path(checkpoint_dir) / _TRAINING_STATE_FILENAME
    if not state_path.exists():
        raise FileNotFoundError(f"No training_state.pt found in {checkpoint_dir}")

    state = torch.load(state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])

    global_step = state["global_step"]
    epoch = state["epoch"]
    run_id = state.get("run_id", "unknown")

    logger.info(
        "Training state restored — run_id=%s  epoch=%d  step=%d",
        run_id, epoch, global_step,
    )
    return global_step, epoch, run_id


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_checkpoints(output_dir: str | Path) -> list[Path]:
    """Return all checkpoint directories sorted by global step (ascending).

    Args:
        output_dir: Root output directory (contains a checkpoints/ subdirectory).

    Returns:
        Sorted list of checkpoint Paths. Empty list if none exist.
    """
    checkpoints_root = Path(output_dir) / "checkpoints"
    if not checkpoints_root.exists():
        return []

    dirs = [
        d for d in checkpoints_root.iterdir()
        if d.is_dir() and (d / _TRAINING_STATE_FILENAME).exists()
    ]

    def _step(d: Path) -> int:
        try:
            state = torch.load(d / _TRAINING_STATE_FILENAME, map_location="cpu")
            return state.get("global_step", 0)
        except Exception:
            return 0

    return sorted(dirs, key=_step)


def get_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the checkpoint with the highest global step, or None.

    Args:
        output_dir: Root output directory.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints exist.
    """
    checkpoints = list_checkpoints(output_dir)
    return checkpoints[-1] if checkpoints else None


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

def should_early_stop(val_history: list[float], patience: int) -> bool:
    """Return True if validation loss has not improved for `patience` evals.

    Improvement is defined as: the most recent val loss is strictly lower
    than the best loss observed before the last `patience` evaluations.

    Args:
        val_history: Chronological list of val/loss values.
        patience: Number of consecutive non-improving evals before stopping.

    Returns:
        True if training should stop early, False otherwise.
    """
    if len(val_history) <= patience:
        return False

    best_before_patience_window = min(val_history[:-patience])
    recent = val_history[-patience:]
    return all(v >= best_before_patience_window for v in recent)
