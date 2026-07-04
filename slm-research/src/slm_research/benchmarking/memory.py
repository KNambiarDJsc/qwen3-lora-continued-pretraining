"""Peak and average GPU memory measurement.

Responsibility: standalone measurement, fully decoupled from training/trainer.py
per architecture spec Section 10 — benchmarking must never share code paths
with the training loop, or numbers get contaminated by gradient/optimizer
overhead.

Sweeps the (batch_size, sequence_length) grid from BenchmarkingConfig against
a single already-loaded checkpoint, mirroring latency.py's grid so results
line up in the same comparison matrix.

CUDA-only: on CPU (no torch.cuda allocator to query) both metrics report 0.0
rather than raising, so this stays callable in CPU-only test/dev environments.

Depends on: modeling/* (loads a specific checkpoint directly)
Consumed by: scripts/benchmark.py
"""
from __future__ import annotations

import logging
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
def measure_memory(
    model: Any,
    benchmark_cfg: BenchmarkingConfig,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """Measure peak and average allocated GPU memory across the sweep grid.

    Args:
        model: Loaded model (e.g. from load_model_from_checkpoint).
        benchmark_cfg: Validated BenchmarkingConfig.
        device: Device to run the measurement on.

    Returns:
        Flat dict with keys:
          benchmark/peak_memory_mb/bs{B}_sl{S}
          benchmark/avg_memory_mb/bs{B}_sl{S}
        All values are 0.0 when `device` is not CUDA.
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
                torch.cuda.reset_peak_memory_stats(device)
                allocated_samples: list[float] = []
                for _ in range(benchmark_cfg.num_measured_iterations):
                    model(**batch)
                    torch.cuda.synchronize(device)
                    allocated_samples.append(torch.cuda.memory_allocated(device) / 1e6)
                peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
                avg_mb = sum(allocated_samples) / len(allocated_samples)
            else:
                for _ in range(benchmark_cfg.num_measured_iterations):
                    model(**batch)
                peak_mb = 0.0
                avg_mb = 0.0

            key = f"bs{batch_size}_sl{seq_len}"
            metrics[f"benchmark/peak_memory_mb/{key}"] = peak_mb
            metrics[f"benchmark/avg_memory_mb/{key}"] = avg_mb

            logger.info(
                "Memory  bs=%d sl=%d  peak=%.1f MB  avg=%.1f MB",
                batch_size, seq_len, peak_mb, avg_mb,
            )

    return metrics
