"""Sole entry point for all Weights & Biases calls in this repository.

No other module may import wandb directly — this wrapper is the single
seam that makes W&B swappable and mockable in tests.

Metric naming follows the assignment's train/* and val/* namespaces exactly
so cross-run comparison in the W&B UI works without renaming columns.

Depends on: tracking/run_metadata.py
Consumed by: training/trainer.py, evaluation/evaluator.py, benchmarking/*
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import wandb

from slm_research.utils.config_schema import LoggingConfig, RootConfig

logger = logging.getLogger(__name__)


class WandbLogger:
    """Wraps the wandb SDK. Instantiate once per training run."""

    def __init__(
        self,
        logging_cfg: LoggingConfig,
        root_cfg: RootConfig,
        run_name: str,
        run_id: str,
        git_hash: str | None,
        system_info: dict[str, Any],
        resume: bool = False,
    ) -> None:
        """Initialise a W&B run.

        Args:
            logging_cfg: Validated LoggingConfig (project, entity, tags).
            root_cfg: Full RootConfig — logged as run config for reproducibility.
            run_name: Human-readable run name (from run_metadata.build_run_name).
            run_id: 8-char hex ID (used as W&B run ID for resume support).
            git_hash: Git commit hash, or None if unavailable.
            system_info: Platform / GPU info dict from run_metadata.
            resume: If True, attempt to resume an existing W&B run with run_id.
        """
        # Build the structured config dict that gets logged as immutable run config.
        config = {
            "model": root_cfg.model.model_dump(),
            "lora": root_cfg.lora.model_dump(),
            "training": root_cfg.training.model_dump(),
            "optimizer": root_cfg.optimizer.model_dump(),
            "scheduler": root_cfg.scheduler.model_dump(),
            "data": {
                "sampling_strategy": root_cfg.data.sampling_strategy,
                "sequence_length": root_cfg.data.sequence_length,
                "n_sources": len(root_cfg.data.sources),
                "sources": [
                    {"name": s.name, "sample_count": s.sample_count}
                    for s in root_cfg.data.sources
                ],
            },
            "run": {
                "seed": root_cfg.run.seed,
                "run_id": run_id,
                "git_hash": git_hash,
            },
            "system": system_info,
        }

        tags = list(logging_cfg.tags) + [
            f"lora_r{root_cfg.lora.r}",
            root_cfg.training.precision,
            root_cfg.optimizer.name,
        ]

        self.run = wandb.init(
            project=logging_cfg.project,
            entity=logging_cfg.entity or None,
            name=run_name,
            id=run_id,
            tags=tags,
            config=config,
            resume="allow" if resume else None,
        )
        self._log_artifacts = logging_cfg.log_checkpoints_as_artifacts
        logger.info("W&B run initialised: %s  url=%s", run_name, self.run.url)

    # ------------------------------------------------------------------
    # Metric logging
    # ------------------------------------------------------------------

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log a flat dict of metrics at a given global step.

        Args:
            metrics: Dict with train/* or val/* keys per the assignment spec.
            step: Global training step (not epoch).
        """
        wandb.log(metrics, step=step)

    def log_generation_table(
        self,
        prompts: list[str],
        completions: list[str],
        step: int,
    ) -> None:
        """Log qualitative generation samples as a W&B Table (val/examples).

        Args:
            prompts: List of conditioning prompt strings.
            completions: Corresponding model completions.
            step: Global step at the time of generation.
        """
        table = wandb.Table(columns=["prompt", "completion"])
        for prompt, completion in zip(prompts, completions):
            table.add_data(prompt, completion)
        wandb.log({"val/examples": table}, step=step)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def log_checkpoint_artifact(
        self,
        checkpoint_path: str | Path,
        run_id: str,
        step: int,
    ) -> None:
        """Log a checkpoint directory as a versioned W&B Artifact.

        Args:
            checkpoint_path: Directory produced by PeftModel.save_pretrained.
            run_id: Run identifier embedded in the artifact name.
            step: Global step — embedded in artifact metadata.
        """
        if not self._log_artifacts:
            return
        artifact = wandb.Artifact(
            name=f"checkpoint-{run_id}",
            type="model",
            metadata={"step": step, "run_id": run_id},
        )
        artifact.add_dir(str(checkpoint_path))
        self.run.log_artifact(artifact)
        logger.debug("Checkpoint artifact logged: step=%d  path=%s", step, checkpoint_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finish(self) -> None:
        """Mark the W&B run as complete. Call at the end of training."""
        wandb.finish()
        logger.info("W&B run finished.")


# ------------------------------------------------------------------
# Offline export (report generation) — the one W&B call outside an
# active run's lifecycle, so it's a module function, not a WandbLogger method.
# ------------------------------------------------------------------

def export_run_history(
    project: str,
    entity: str | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Pull run config/summary/history via the W&B public API.

    Used by scripts/cli.py's `export` command to gather already-completed
    runs' data for the LaTeX report (reports/latex/main.tex) — offline,
    after training/evaluation/benchmarking, not during them. Kept in this
    module rather than a separate one because wandb_logger.py is the sole
    module permitted to import wandb (architecture spec Section 9).

    Args:
        project: W&B project name (see configs/logging/wandb.yaml).
        entity: W&B entity/username. None uses the API key's default entity.
        run_id: Export exactly one run by id. None exports every run in the project.

    Returns:
        One dict per run: run_id, name, state, config, summary, and history
        (a list of {metric: value} dicts, one per logged step).
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project

    runs = [api.run(f"{path}/{run_id}")] if run_id is not None else list(api.runs(path))

    exported: list[dict[str, Any]] = []
    for run in runs:
        exported.append(
            {
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "config": dict(run.config),
                "summary": dict(run.summary),
                "history": run.history(pandas=False),
            }
        )

    logger.info("Exported %d run(s) from %s.", len(exported), path)
    return exported
