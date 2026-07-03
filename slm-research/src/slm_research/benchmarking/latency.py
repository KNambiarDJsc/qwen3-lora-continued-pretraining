"""Per-token / per-sequence inference latency measurement.

Responsibility: standalone measurement, fully decoupled from training/trainer.py
per architecture spec Section 10 — benchmarking must never share code paths
with the training loop, or numbers get contaminated by gradient/optimizer
overhead.

Depends on: modeling/* (loads a specific checkpoint directly)
Consumed by: scripts/benchmark.py
"""

from typing import Any


def measure_latency(model: Any, benchmark_config: dict[str, Any]) -> dict[str, float]:
    """Run the latency measurement pass per configs/benchmarking/default.yaml.

    Raises:
        NotImplementedError: implementation lands in Phase 8 (Benchmarking).
    """
    raise NotImplementedError("Phase 8: implement latency measurement.")
