"""Batch collation: stack packed sequences into tensors for causal LM.

Responsibility: receive a list of fixed-length packed examples and return
the {input_ids, attention_mask, labels} tensors expected by the trainer.

For causal language modelling, labels == input_ids (next-token prediction
on every position). No masking, no -100 padding positions — the sequences
are already packed to exactly sequence_length with no padding tokens.

Depends on: mixture.py output (via DataLoader)
Consumed by: training/trainer.py, evaluation/evaluator.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CausalLMCollator:
    """Collates fixed-length packed examples into training tensors.

    Each example must have "input_ids" and "attention_mask" as plain
    Python lists of integers (as produced by packing.py).

    Returns:
        Dict with:
            input_ids       (batch, seq_len)  int64
            attention_mask  (batch, seq_len)  int64
            labels          (batch, seq_len)  int64  — identical to input_ids
    """

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Stack a batch of packed examples.

        Args:
            examples: List of dicts each with "input_ids" and "attention_mask"
                keys, both plain Python lists of integers.

        Returns:
            Batch dict with three tensors ready for model forward pass.
        """
        input_ids = torch.tensor(
            [ex["input_ids"] for ex in examples], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [ex["attention_mask"] for ex in examples], dtype=torch.long
        )
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
