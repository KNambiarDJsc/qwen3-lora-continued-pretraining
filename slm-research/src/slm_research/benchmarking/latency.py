"""Per-token / per-sequence inference latency measurement.

Responsibility: standalone measurement, fully decoupled from training/trainer.py
per architecture spec Section 10 — benchmarking must never share code paths
with the training loop, or numbers get contaminated by gradient/optimizer
overhead.

Sweeps the (batch_size, sequence_length) grid from BenchmarkingConfig against
a single already-loaded checkpoint. Precision and LoRA rank are fixed
properties of that checkpoint — comparing across precisions/ranks happens by
running this script once per checkpoint and aggregating the resulting JSON
files (see scripts/benchmark.py), not by sweeping them in-process.

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
def measure_latency(
    model: Any,
    benchmark_cfg: BenchmarkingConfig,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """Measure forward-pass latency across the batch-size × sequence-length grid.

    For each combination, runs `num_warmup_iterations` untimed passes to let
    CUDA kernels/caches settle, then times `num_measured_iterations` passes.

    Args:
        model: Loaded model (e.g. from load_model_from_checkpoint).
        benchmark_cfg: Validated BenchmarkingConfig.
        device: Device to run the measurement on.

    Returns:
        Flat dict with keys:
          benchmark/latency_ms_per_sequence/bs{B}_sl{S}
          benchmark/latency_ms_per_token/bs{B}_sl{S}
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

            elapsed = 0.0
            for _ in range(benchmark_cfg.num_measured_iterations):
                start = time.perf_counter()
                model(**batch)
                if is_cuda:
                    torch.cuda.synchronize(device)
                elapsed += time.perf_counter() - start

            per_sequence_ms = 1000.0 * elapsed / benchmark_cfg.num_measured_iterations
            per_token_ms = per_sequence_ms / seq_len

            key = f"bs{batch_size}_sl{seq_len}"
            metrics[f"benchmark/latency_ms_per_sequence/{key}"] = per_sequence_ms
            metrics[f"benchmark/latency_ms_per_token/{key}"] = per_token_ms

            logger.info(
                "Latency  bs=%d sl=%d  %.3f ms/sequence  %.5f ms/token",
                batch_size, seq_len, per_sequence_ms, per_token_ms,
            )

    return metrics
