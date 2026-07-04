"""Unit tests for evaluation/evaluator.py.

Uses a fake model whose logits are constructed from a known per-position
"confidence" so the exact cross-entropy at every position is analytically
predictable:

    logits[b, t, :] = 0                       everywhere
    logits[b, t, target_class] = confidence[t]

For a one-hot-style logit vector like this, cross-entropy reduces to:

    CE(t) = log((vocab_size - 1) + exp(confidence[t])) - confidence[t]

This lets tests assert exact expected values for val/loss, val/ppl, val/bpt,
and val/ppl_by_length_bucket without needing a real trained model.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from slm_research.evaluation.evaluator import Evaluator
from slm_research.utils.config_schema import EvaluationConfig

VOCAB_SIZE = 10


def _expected_ce(confidence: float, vocab_size: int = VOCAB_SIZE) -> float:
    return math.log((vocab_size - 1) + math.exp(confidence)) - confidence


class FakeCausalLM(nn.Module):
    """Produces logits from a per-position confidence schedule.

    `confidences[t]` is the logit assigned to the true next-token class at
    position t (shifted labels are input_ids[:, 1:], so position t's logit
    row is used to predict input_ids[:, t + 1]).
    """

    def __init__(self, confidences: list[float], vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.confidences = confidences
        self.vocab_size = vocab_size
        # Dummy parameter so nn.Module has something to move/eval/train.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask  # unused — logits are computed from input_ids only
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        for t in range(seq_len - 1):
            target_class = input_ids[:, t + 1]
            conf = self.confidences[t]
            logits[torch.arange(batch_size), t, target_class] = conf
        return SimpleNamespace(logits=logits)


def _make_batch(seq_len: int, batch_size: int = 2, vocab_size: int = VOCAB_SIZE) -> dict:
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _eval_cfg(length_buckets: list[int]) -> EvaluationConfig:
    return EvaluationConfig(
        val_split_fraction=0.1,
        length_buckets=length_buckets,
        num_generation_samples=2,
        generation_max_new_tokens=16,
    )


def test_evaluate_overall_metrics_match_hand_computed_ce() -> None:
    seq_len = 8
    confidence = 3.0
    model = FakeCausalLM([confidence] * seq_len)
    batch = _make_batch(seq_len)
    val_dl = [batch]
    eval_cfg = _eval_cfg(length_buckets=[seq_len])

    evaluator = Evaluator(model=model, val_dataloader=val_dl, eval_cfg=eval_cfg, device="cpu")
    metrics = evaluator.evaluate()

    expected_loss = _expected_ce(confidence)
    assert metrics["val/loss"] == pytest.approx(expected_loss)
    assert metrics["val/token_loss"] == metrics["val/loss"]
    assert metrics["val/ppl"] == pytest.approx(math.exp(expected_loss))
    assert metrics["val/bpt"] == pytest.approx(expected_loss / math.log(2))

    batch_size = batch["input_ids"].shape[0]
    assert metrics["val/n_tokens"] == float(batch_size * (seq_len - 1))
    assert metrics["val/n_sequences"] == float(batch_size)
    # Uniform confidence ⇒ every sequence has the same mean loss as the overall mean.
    assert metrics["val/sequence_loss"] == pytest.approx(expected_loss)


def test_length_bucket_ppl_reflects_position_dependent_confidence() -> None:
    seq_len = 8
    # Low confidence (high loss) for the first half, high confidence (low
    # loss) for the second half of the (shifted) position range.
    low_conf, high_conf = 0.5, 6.0
    confidences = [low_conf] * (seq_len // 2) + [high_conf] * (seq_len // 2)
    model = FakeCausalLM(confidences)
    batch = _make_batch(seq_len)
    val_dl = [batch]
    eval_cfg = _eval_cfg(length_buckets=[4, 8])

    evaluator = Evaluator(model=model, val_dataloader=val_dl, eval_cfg=eval_cfg, device="cpu")
    metrics = evaluator.evaluate()

    assert metrics["val/ppl_by_length_bucket/0-4"] == pytest.approx(
        math.exp(_expected_ce(low_conf))
    )
    assert metrics["val/ppl_by_length_bucket/4-8"] == pytest.approx(
        math.exp(_expected_ce(high_conf))
    )
    # Long-range (high-confidence) positions should show lower perplexity.
    assert (
        metrics["val/ppl_by_length_bucket/4-8"]
        < metrics["val/ppl_by_length_bucket/0-4"]
    )


def test_build_buckets_converts_thresholds_to_ranges() -> None:
    eval_cfg = _eval_cfg(length_buckets=[128, 256, 512])
    evaluator = Evaluator(
        model=FakeCausalLM([0.0]), val_dataloader=[], eval_cfg=eval_cfg, device="cpu"
    )
    assert evaluator._build_buckets() == {
        "0-128": (0, 128),
        "128-256": (128, 256),
        "256-512": (256, 512),
    }


def test_attention_mask_excludes_padding_from_metrics() -> None:
    seq_len = 6
    confidence = 2.0
    model = FakeCausalLM([confidence] * seq_len)
    batch = _make_batch(seq_len, batch_size=1)
    # Mark the last two positions as padding.
    batch["attention_mask"][:, -2:] = 0
    val_dl = [batch]
    eval_cfg = _eval_cfg(length_buckets=[seq_len])

    evaluator = Evaluator(model=model, val_dataloader=val_dl, eval_cfg=eval_cfg, device="cpu")
    metrics = evaluator.evaluate()

    # Only 3 of the 5 shifted positions remain unmasked.
    assert metrics["val/n_tokens"] == 3.0
    assert metrics["val/loss"] == pytest.approx(_expected_ce(confidence))
