"""Unit tests for benchmarking/memory.py.

CI runs on CPU, where memory measurement is documented as a 0.0 no-op (no
CUDA allocator to query) — these tests confirm that CPU fallback behavior
and that the model is still driven through the full warmup/measured grid.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from slm_research.benchmarking.memory import measure_memory
from slm_research.utils.config_schema import BenchmarkingConfig

VOCAB_SIZE = 16


class CountingFakeCausalLM(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.dummy = nn.Parameter(torch.zeros(1))
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        self.forward_calls += 1
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.config.vocab_size)
        return SimpleNamespace(logits=logits)


def _cfg(batch_sizes: list[int], seq_lens: list[int], warmup: int = 1, measured: int = 2) -> BenchmarkingConfig:
    return BenchmarkingConfig(
        batch_sizes_to_test=batch_sizes,
        precision_modes_to_test=["bf16"],
        sequence_lengths_to_test=seq_lens,
        lora_ranks_to_test=[8],
        num_warmup_iterations=warmup,
        num_measured_iterations=measured,
    )


def test_measure_memory_cpu_reports_zero() -> None:
    model = CountingFakeCausalLM()
    cfg = _cfg(batch_sizes=[1, 2], seq_lens=[4, 8])

    metrics = measure_memory(model, cfg, device="cpu")

    expected_keys = {
        f"benchmark/{metric}/bs{b}_sl{s}"
        for metric in ("peak_memory_mb", "avg_memory_mb")
        for b in (1, 2)
        for s in (4, 8)
    }
    assert set(metrics.keys()) == expected_keys
    assert all(v == 0.0 for v in metrics.values())


def test_measure_memory_drives_full_warmup_and_measured_grid() -> None:
    model = CountingFakeCausalLM()
    cfg = _cfg(batch_sizes=[1, 2], seq_lens=[4, 8], warmup=2, measured=3)

    measure_memory(model, cfg, device="cpu")

    n_grid_points = 2 * 2
    assert model.forward_calls == n_grid_points * (2 + 3)
