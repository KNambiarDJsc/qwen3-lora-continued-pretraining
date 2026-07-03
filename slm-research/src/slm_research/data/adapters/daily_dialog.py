"""DailyDialog dataset adapter (li2017dailydialog/daily_dialog)."""
from __future__ import annotations

from typing import Any

from slm_research.data.adapters.base import DatasetAdapter


class DailyDialogAdapter(DatasetAdapter):
    """Adapter for li2017dailydialog/daily_dialog.

    Joins all utterances in a dialogue turn into one text document,
    one utterance per line. Act and emotion labels are dropped.

    The HF dataset column is "dialog" (list[str]). The spec uses the
    name "utterances" — both are handled.
    """

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        utterances: list[str] = example.get("dialog", example.get("utterances", []))
        text = "\n".join(u.strip() for u in utterances if u.strip())
        return {"text": text}
