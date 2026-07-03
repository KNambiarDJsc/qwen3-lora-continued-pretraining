"""XSum dataset adapter (EdinburghNLP/xsum)."""
from __future__ import annotations

from typing import Any

from slm_research.data.adapters.base import DatasetAdapter


class XSumAdapter(DatasetAdapter):
    """Adapter for EdinburghNLP/xsum.

    Uses the source document only.
    Summary column is dropped — this is language modeling, not summarization.
    """

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        return {"text": example["document"]}
