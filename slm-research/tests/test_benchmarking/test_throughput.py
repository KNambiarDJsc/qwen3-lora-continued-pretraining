"""Unit tests for benchmarking/throughput.py.

Runs on CPU with a tiny fake model — asserts on structure and the
tokens/sec-to-samples/sec relationship (tokens_per_sec == samples_per_sec *
seq_len, since both share the same measured wall-clock window) rather than
absolute timing values.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from slm_research.benchmarking.throughput import measure_throughput
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


def test_measure_throughput_produces_one_pair_per_grid_point() -> None:
    model = FakeCausalLM()
    cfg = _cfg(batch_sizes=[1, 2], seq_lens=[4, 8])

    metrics = measure_throughput(model, cfg, device="cpu")

    expected_keys = {
        f"benchmark/{metric}/bs{b}_sl{s}"
        for metric in ("tokens_per_sec", "samples_per_sec")
        for b in (1, 2)
        for s in (4, 8)
    }
    assert set(metrics.keys()) == expected_keys


def test_measure_throughput_tokens_per_sec_matches_samples_per_sec_times_seq_len() -> None:
    model = FakeCausalLM()
    cfg = _cfg(batch_sizes=[2], seq_lens=[8])

    metrics = measure_throughput(model, cfg, device="cpu")

    tokens_per_sec = metrics["benchmark/tokens_per_sec/bs2_sl8"]
    samples_per_sec = metrics["benchmark/samples_per_sec/bs2_sl8"]
    assert tokens_per_sec == pytest.approx(samples_per_sec * 8)


def test_measure_throughput_all_values_positive() -> None:
    model = FakeCausalLM()
    cfg = _cfg(batch_sizes=[1], seq_lens=[4])

    metrics = measure_throughput(model, cfg, device="cpu")

    assert all(v > 0.0 for v in metrics.values())
