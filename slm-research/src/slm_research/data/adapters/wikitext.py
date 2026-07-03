"""WikiText dataset adapter (Salesforce/wikitext)."""
from __future__ import annotations

from typing import Any

from slm_research.data.adapters.base import DatasetAdapter


class WikitextAdapter(DatasetAdapter):
    """Adapter for Salesforce/wikitext.

    Config name: "wikitext-103-raw-v1" (set in mixture.yaml).
    Column used: text.
    """

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        return {"text": example["text"]}
