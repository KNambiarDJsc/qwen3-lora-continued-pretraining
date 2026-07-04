"""Tokens/sec and samples/sec throughput measurement.

Responsibility: standalone measurement, fully decoupled from training/trainer.py
per architecture spec Section 10 — benchmarking must never share code paths
with the training loop, or numbers get contaminated by gradient/optimizer
overhead.

Sweeps the (batch_size, sequence_length) grid from BenchmarkingConfig against
a single already-loaded checkpoint, mirroring latency.py's grid so results
line up in the same comparison matrix.

Depends on: modeling/* (loads a specific checkpoint directly)
Consumed by: scripts/benchmark.py
"""
from __future__ import annotations

import logging
import time
from typing import Any

import torch

from slm_research.utils.config_schema import BenchmarkingConfig

logger = logging.getLogger(__name__)


def _random_batch(
    vocab_size: int, batch_size: int, seq_len: int, device: torch.device
) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@torch.no_grad()
def measure_throughput(
    model: Any,
    benchmark_cfg: BenchmarkingConfig,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """Measure sustained forward-pass throughput across the batch/seq-len grid.

    Unlike latency.py (which times individual passes), this sums wall-clock
    time across all measured iterations for a combination and divides by the
    total tokens/samples processed — the right denominator for a sustained
    "how many tokens per second can this checkpoint process" number.

    Args:
        model: Loaded model (e.g. from load_model_from_checkpoint).
        benchmark_cfg: Validated BenchmarkingConfig.
        device: Device to run the measurement on.

    Returns:
        Flat dict with keys:
          benchmark/tokens_per_sec/bs{B}_sl{S}
          benchmark/samples_per_sec/bs{B}_sl{S}
    """
    device = torch.device(device)
    model.eval()
    vocab_size = model.config.vocab_size
    is_cuda = device.type == "cuda"

    metrics: dict[str, float] = {}
    for batch_size in benchmark_cfg.batch_sizes_to_test:
        for seq_len in benchmark_cfg.sequence_lengths_to_test:
            batch = _random_batch(vocab_size, batch_size, seq_len, device)

            for _ in range(benchmark_cfg.num_warmup_iterations):
                model(**batch)
            if is_cuda:
                torch.cuda.synchronize(device)

            start = time.perf_counter()
            for _ in range(benchmark_cfg.num_measured_iterations):
                model(**batch)
            if is_cuda:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            n_iters = benchmark_cfg.num_measured_iterations
            tokens_per_sec = (batch_size * seq_len * n_iters) / elapsed
            samples_per_sec = (batch_size * n_iters) / elapsed

            key = f"bs{batch_size}_sl{seq_len}"
            metrics[f"benchmark/tokens_per_sec/{key}"] = tokens_per_sec
            metrics[f"benchmark/samples_per_sec/{key}"] = samples_per_sec

            logger.info(
                "Throughput  bs=%d sl=%d  %.1f tokens/sec  %.2f samples/sec",
                batch_size, seq_len, tokens_per_sec, samples_per_sec,
            )

    return metrics
