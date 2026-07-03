"""CNN/DailyMail dataset adapter (abisee/cnn_dailymail, config "3.0.0")."""
from __future__ import annotations

from typing import Any

from slm_research.data.adapters.base import DatasetAdapter


class CnnDailymailAdapter(DatasetAdapter):
    """Adapter for abisee/cnn_dailymail.

    Uses the article body only.
    Highlights column is dropped — this is language modeling, not summarization.
    """

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        return {"text": example["article"]}
