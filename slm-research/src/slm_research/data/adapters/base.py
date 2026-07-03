"""Abstract base class for dataset adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DatasetAdapter(ABC):
    """Converts one raw HuggingFace dataset example to {"text": str}.

    Each dataset gets exactly one adapter subclass. Adding a new dataset
    requires only: (1) a new subclass here, (2) one entry in registry.py.
    """

    @abstractmethod
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw dataset row to a standardized {"text": str} dict.

        Args:
            example: One row from a HuggingFace Dataset or IterableDataset.

        Returns:
            Dict with a single "text" key containing plain, unnormalized text.
            Cleaning happens downstream in preprocessing.py.
        """
        ...
