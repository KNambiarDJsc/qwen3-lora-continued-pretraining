"""FineWeb-Edu dataset adapter (HuggingFaceFW/fineweb-edu)."""
from __future__ import annotations

from typing import Any

from slm_research.data.adapters.base import DatasetAdapter


class FineWebEduAdapter(DatasetAdapter):
    """Adapter for HuggingFaceFW/fineweb-edu.

    High-quality educational web text filtered for LM pretraining.
    Loaded with streaming=True (very large corpus — ~1.3 TB).
    Column used: text. All metadata columns (url, score, etc.) are dropped.
    """

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        return {"text": example["text"]}
