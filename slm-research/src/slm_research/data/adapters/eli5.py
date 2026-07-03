"""ELI5 dataset adapter."""
from __future__ import annotations

from typing import Any

from slm_research.data.adapters.base import DatasetAdapter


class ELI5Adapter(DatasetAdapter):
    """Adapter for the ELI5 (Explain Like I'm Five) dataset.

    Combines the question title with the highest-scored answer into one
    text document. The Q/A framing is preserved because it maintains the
    natural language structure while still training with a next-token
    prediction objective — the model sees both the question and the
    explanation as continuous prose.

    Column structure (classic eli5):
        title: str           — the question
        answers: dict        — {"text": list[str], "score": list[int]}

    Note: the canonical `eli5` HF dataset was deprecated. If the dataset
    has moved, update the hf_path in registry.py; this adapter handles any
    source with the same column structure.
    """

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        question = example.get("title", example.get("question", "")).strip()

        answers_block = example.get("answers", {})
        answer = ""
        if isinstance(answers_block, dict):
            texts: list[str] = answers_block.get("text", [])
            scores: list[int] = answers_block.get("score", [])
            if texts and scores:
                best = max(range(len(scores)), key=lambda i: scores[i])
                answer = texts[best].strip()
            elif texts:
                answer = texts[0].strip()
        elif isinstance(answers_block, (list, tuple)) and answers_block:
            answer = str(answers_block[0]).strip()

        text = f"Question:\n{question}\n\nAnswer:\n{answer}"
        return {"text": text}
