"""Deterministic seeding for reproducible runs.

Depends on: configs/training/*.yaml (seed field)
Consumed by: training/trainer.py (called first, before model/data construction)
"""


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch (CPU + CUDA) RNG seeds deterministically.

    Raises:
        NotImplementedError: implementation lands in Phase 6 (Training).
    """
    raise NotImplementedError("Phase 6: implement deterministic seeding.")
