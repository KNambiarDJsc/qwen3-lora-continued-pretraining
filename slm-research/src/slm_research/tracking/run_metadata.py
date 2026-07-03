"""Run identity: UUID generation, git hash capture, system/GPU info.

Depends on: nothing (called first, at run init)
Consumed by: tracking/wandb_logger.py
"""

from typing import Any


def generate_run_id() -> str:
    """Generate a unique run identifier."""
    raise NotImplementedError("Phase 6: implement run ID generation.")


def capture_git_hash() -> str:
    """Capture the current git commit hash for reproducibility tracking."""
    raise NotImplementedError("Phase 6: implement git hash capture.")


def capture_system_info() -> dict[str, Any]:
    """Capture GPU/CPU/OS/library-version info for the run config."""
    raise NotImplementedError("Phase 6: implement system info capture.")
