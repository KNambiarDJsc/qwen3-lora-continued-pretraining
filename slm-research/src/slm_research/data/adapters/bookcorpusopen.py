"""BookCorpusOpen dataset adapter (lucadiliello/bookcorpusopen)."""
from __future__ import annotations

from typing import Any

from slm_research.data.adapters.base import DatasetAdapter


class BookCorpusOpenAdapter(DatasetAdapter):
    """Adapter for lucadiliello/bookcorpusopen.

    Loaded with streaming=True (see mixture.yaml).
    Columns available: text, title. Only text is used.
    """

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        return {"text": example["text"]}
