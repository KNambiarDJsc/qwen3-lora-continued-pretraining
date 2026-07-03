"""Deterministic seeding for reproducible training runs.

Depends on: nothing
Consumed by: scripts/train.py (called before model or data construction)
"""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set all RNG seeds deterministically.

    Covers Python, NumPy, PyTorch CPU, and all CUDA devices. Also forces
    cuDNN into deterministic mode — this costs a small throughput penalty
    in exchange for exact reproducibility across runs with the same seed.

    Args:
        seed: Integer seed from RunConfig.seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN determinism: eliminates non-deterministic conv algorithms.
    # benchmark=False forces cuDNN to use the same algorithm every run.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # HuggingFace Transformers and Datasets use this env var for seeding.
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info("Global seed set to %d.", seed)
