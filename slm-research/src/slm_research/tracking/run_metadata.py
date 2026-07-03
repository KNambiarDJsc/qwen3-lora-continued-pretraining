"""Run identity: UUID generation, git hash capture, system / GPU info.

Depends on: nothing (called at run init, before any model or data code)
Consumed by: tracking/wandb_logger.py, scripts/train.py
"""
from __future__ import annotations

import logging
import platform
import subprocess
import uuid
from typing import Any

import torch

logger = logging.getLogger(__name__)


def generate_run_id() -> str:
    """Return an 8-character hex run identifier.

    Short enough to embed in checkpoint directory names without clutter,
    long enough to be collision-free across the sweep.
    """
    return uuid.uuid4().hex[:8]


def capture_git_hash() -> str | None:
    """Capture the HEAD commit hash for the current repository.

    Returns None if git is unavailable or the directory is not a git repo —
    this happens during CI dry-runs and on fresh RunPod pods before the repo
    is initialised.
    """
    try:
        hash_ = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return hash_
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        logger.debug("git not available or not a git repo — skipping git hash capture.")
        return None


def capture_system_info() -> dict[str, Any]:
    """Snapshot platform, library versions, and GPU specs at run init.

    This dict is logged to W&B run config so every run is self-documenting
    and reproducible without separately recording the environment.

    Returns:
        Flat dict with string keys and JSON-serialisable values.
    """
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_memory_gb": round(props.total_memory / 1e9, 2),
                "gpu_compute_capability": f"{props.major}.{props.minor}",
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
            }
        )

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    try:
        import peft
        info["peft_version"] = peft.__version__
    except ImportError:
        pass

    return info


def build_run_name(
    model_name: str,
    lora_rank: int,
    precision: str,
    run_id: str,
) -> str:
    """Build a human-readable run name for W&B and checkpoint directories.

    Format: {model_short}-r{rank}-{precision}-{run_id}
    Example: Qwen3-0.6B-Base-r32-bf16-a1b2c3d

    Args:
        model_name: Full HF model name (e.g. "Qwen/Qwen3-0.6B-Base").
        lora_rank: LoRA rank from LoRAConfig.
        precision: Precision string from TrainingConfig.
        run_id: 8-char hex from generate_run_id().

    Returns:
        Slug-style run name string.
    """
    short_name = model_name.split("/")[-1]
    return f"{short_name}-r{lora_rank}-{precision}-{run_id}"
