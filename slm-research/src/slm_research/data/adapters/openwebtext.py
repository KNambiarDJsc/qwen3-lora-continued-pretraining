"""OpenWebText dataset adapter (Skylion007/openwebtext)."""
from __future__ import annotations

from typing import Any

from slm_research.data.adapters.base import DatasetAdapter


class OpenWebTextAdapter(DatasetAdapter):
    """Adapter for Skylion007/openwebtext.

    Loaded with streaming=True (see mixture.yaml).
    Column used: text.
    """

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        return {"text": example["text"]}
