"""AG News dataset adapter (fancyzhx/ag_news)."""
from __future__ import annotations

from typing import Any

from slm_research.data.adapters.base import DatasetAdapter


class AgNewsAdapter(DatasetAdapter):
    """Adapter for fancyzhx/ag_news.

    Combines title and description into a single text block.
    Falls back to the pre-combined 'text' column if separate columns
    are absent (dataset schema varies by HF version).
    Labels are dropped — this is language modeling, not classification.
    """

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        if "title" in example and "description" in example:
            text = example["title"].strip() + "\n\n" + example["description"].strip()
        else:
            text = example["text"]
        return {"text": text}
