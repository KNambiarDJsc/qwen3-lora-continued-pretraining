"""Entrypoint: benchmark.

Thin orchestration only per architecture spec — wires configs to the
appropriate module(s) in src/slm_research/. No business logic lives here,
with one explicit exception: architecture spec Section 10 assigns the
multi-run comparison-matrix aggregation to this script directly (there is no
dedicated src/ module for it).

Two modes, both driven by Hydra overrides:

  Single checkpoint:
      python scripts/benchmark.py +checkpoint_path=outputs/checkpoints/run_abc_epoch0_step500
    Loads that checkpoint, runs latency/throughput/memory sweeps over the
    batch_size x sequence_length grid in configs/benchmarking/default.yaml,
    writes reports/benchmark_results/{run_id}.json, and logs to the
    checkpoint's original W&B run.

  Aggregate:
      python scripts/benchmark.py +aggregate=true
    Scans reports/benchmark_results/*.json (produced by prior single-checkpoint
    runs — one per rank/precision sweep point) and writes
    reports/benchmark_results/comparison_matrix.json, the rank x batch size x
    precision x sequence length matrix consumed by the LaTeX report.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path("reports/benchmark_results")
_METRIC_KEY_RE = re.compile(r"^benchmark/(?P<metric>[a-z_]+)/bs(?P<bs>\d+)_sl(?P<sl>\d+)$")


def _run_single_checkpoint(cfg: DictConfig, checkpoint_path: str) -> None:
    from slm_research.utils.config_schema import validate_config

    plain = OmegaConf.to_container(cfg, resolve=True)
    plain.pop("checkpoint_path", None)
    root_cfg = validate_config(plain)

    from slm_research.training.seed import set_seed

    set_seed(root_cfg.run.seed)

    import torch

    from slm_research.tracking.run_metadata import build_run_name, capture_git_hash, capture_system_info

    state = torch.load(Path(checkpoint_path) / "training_state.pt", map_location="cpu")
    run_id = state.get("run_id", "unknown")
    global_step = state.get("global_step", 0)
    git_hash = capture_git_hash()
    run_name = build_run_name(
        model_name=root_cfg.model.name,
        lora_rank=root_cfg.lora.r,
        precision=root_cfg.training.precision,
        run_id=run_id,
    )
    logger.info("Benchmarking checkpoint: %s  run=%s", checkpoint_path, run_name)

    from slm_research.tracking.wandb_logger import WandbLogger

    wandb_logger = WandbLogger(
        logging_cfg=root_cfg.logging,
        root_cfg=root_cfg,
        run_name=run_name,
        run_id=run_id,
        git_hash=git_hash,
        system_info=capture_system_info(),
        resume=True,
    )

    try:
        from slm_research.benchmarking.latency import measure_latency
        from slm_research.benchmarking.memory import measure_memory
        from slm_research.benchmarking.throughput import measure_throughput
        from slm_research.modeling.model_factory import load_model_from_checkpoint

        logger.info("Loading checkpoint …")
        model = load_model_from_checkpoint(root_cfg, checkpoint_path=checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        metrics: dict[str, float] = {}
        logger.info("Measuring latency …")
        metrics.update(measure_latency(model, root_cfg.benchmarking, device=device))
        logger.info("Measuring throughput …")
        metrics.update(measure_throughput(model, root_cfg.benchmarking, device=device))
        logger.info("Measuring memory …")
        metrics.update(measure_memory(model, root_cfg.benchmarking, device=device))

        wandb_logger.log(metrics, step=global_step)

        result = {
            "run_id": run_id,
            "checkpoint_path": checkpoint_path,
            "model_name": root_cfg.model.name,
            "lora_rank": root_cfg.lora.r,
            "precision": root_cfg.training.precision,
            "git_hash": git_hash,
            "metrics": metrics,
        }
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _RESULTS_DIR / f"{run_id}.json"
        out_path.write_text(json.dumps(result, indent=2))
        logger.info("Benchmark results written → %s", out_path)

    finally:
        wandb_logger.finish()


def _aggregate_results() -> None:
    """Build the rank x batch size x precision x sequence length matrix.

    Reads every {run_id}.json under reports/benchmark_results/ (one per
    prior single-checkpoint benchmark run) and flattens each metric key
    (benchmark/{metric}/bs{B}_sl{S}) into one row per (rank, precision,
    batch_size, sequence_length, metric) combination.
    """
    if not _RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"{_RESULTS_DIR} does not exist — run at least one "
            "`+checkpoint_path=...` benchmark before aggregating."
        )

    result_files = sorted(
        p for p in _RESULTS_DIR.glob("*.json") if p.name != "comparison_matrix.json"
    )
    if not result_files:
        raise FileNotFoundError(
            f"No benchmark result files found in {_RESULTS_DIR} — run at "
            "least one `+checkpoint_path=...` benchmark before aggregating."
        )

    rows: list[dict[str, Any]] = []
    for path in result_files:
        data = json.loads(path.read_text())
        by_grid: dict[tuple[int, int], dict[str, float]] = {}
        for key, value in data["metrics"].items():
            match = _METRIC_KEY_RE.match(key)
            if not match:
                continue
            grid_key = (int(match["bs"]), int(match["sl"]))
            by_grid.setdefault(grid_key, {})[match["metric"]] = value

        for (batch_size, seq_len), metric_values in sorted(by_grid.items()):
            rows.append(
                {
                    "run_id": data["run_id"],
                    "lora_rank": data["lora_rank"],
                    "precision": data["precision"],
                    "batch_size": batch_size,
                    "sequence_length": seq_len,
                    **metric_values,
                }
            )

    out_path = _RESULTS_DIR / "comparison_matrix.json"
    out_path.write_text(json.dumps(rows, indent=2))
    logger.info(
        "Comparison matrix written → %s (%d rows from %d result files)",
        out_path, len(rows), len(result_files),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint_path = cfg.get("checkpoint_path", None)
    aggregate = cfg.get("aggregate", False)

    if checkpoint_path is not None:
        _run_single_checkpoint(cfg, checkpoint_path)
    elif aggregate:
        _aggregate_results()
    else:
        raise ValueError(
            "scripts/benchmark.py requires either +checkpoint_path=<dir> "
            "(measure one checkpoint) or +aggregate=true (build the "
            "comparison matrix from prior results)."
        )


if __name__ == "__main__":
    main()
