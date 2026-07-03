"""Peak and average GPU memory measurement.

Responsibility: standalone measurement, fully decoupled from training/trainer.py
per architecture spec Section 10 — benchmarking must never share code paths
with the training loop, or numbers get contaminated by gradient/optimizer
overhead.

Depends on: modeling/* (loads a specific checkpoint directly)
Consumed by: scripts/benchmark.py
"""

from typing import Any


def measure_memory(model: Any, benchmark_config: dict[str, Any]) -> dict[str, float]:
    """Run the memory measurement pass per configs/benchmarking/default.yaml.

    Raises:
        NotImplementedError: implementation lands in Phase 8 (Benchmarking).
    """
    raise NotImplementedError("Phase 8: implement memory measurement.")
