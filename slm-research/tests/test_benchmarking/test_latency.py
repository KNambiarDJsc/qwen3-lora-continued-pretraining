"""Unit tests for benchmarking/latency.py.

Runs on CPU with a tiny fake model — real GPU timing isn't reproducible in
CI, so these tests assert on structure (one pair of keys per grid point,
correct key naming, per-token = per-sequence / seq_len) rather than on
absolute timing values.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from slm_research.benchmarking.latency import measure_latency
from slm_research.utils.config_schema import BenchmarkingConfig

VOCAB_SIZE = 16


class FakeCausalLM(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.config.vocab_size)
        return SimpleNamespace(logits=logits)


def _cfg(batch_sizes: list[int], seq_lens: list[int]) -> BenchmarkingConfig:
    return BenchmarkingConfig(
        batch_sizes_to_test=batch_sizes,
        precision_modes_to_test=["bf16"],
        sequence_lengths_to_test=seq_lens,
        lora_ranks_to_test=[8],
        num_warmup_iterations=1,
        num_measured_iterations=2,
    )


def test_measure_latency_produces_one_pair_per_grid_point() -> None:
    model = FakeCausalLM()
    cfg = _cfg(batch_sizes=[1, 2], seq_lens=[4, 8])

    metrics = measure_latency(model, cfg, device="cpu")

    expected_keys = {
        f"benchmark/{metric}/bs{b}_sl{s}"
        for metric in ("latency_ms_per_sequence", "latency_ms_per_token")
        for b in (1, 2)
        for s in (4, 8)
    }
    assert set(metrics.keys()) == expected_keys


def test_measure_latency_per_token_is_per_sequence_divided_by_seq_len() -> None:
    model = FakeCausalLM()
    cfg = _cfg(batch_sizes=[1], seq_lens=[8])

    metrics = measure_latency(model, cfg, device="cpu")

    per_seq = metrics["benchmark/latency_ms_per_sequence/bs1_sl8"]
    per_token = metrics["benchmark/latency_ms_per_token/bs1_sl8"]
    assert per_token == per_seq / 8


def test_measure_latency_all_values_nonnegative() -> None:
    model = FakeCausalLM()
    cfg = _cfg(batch_sizes=[1], seq_lens=[4])

    metrics = measure_latency(model, cfg, device="cpu")

    assert all(v >= 0.0 for v in metrics.values())
