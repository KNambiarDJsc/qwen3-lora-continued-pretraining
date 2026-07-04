"""Entrypoint: train.

Thin orchestration only — wires Hydra config → DataModule → Model → Trainer.
No business logic lives here (architecture spec Section 12).

Run:
    python scripts/train.py
    python scripts/train.py lora=rank32 training=runpod_4090 optimizer=adamw
    python scripts/train.py +resume_from=outputs/checkpoints/run_abc_epoch0_step500
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # ------------------------------------------------------------------ #
    # 1. Validate config                                                   #
    # ------------------------------------------------------------------ #
    from slm_research.utils.config_schema import validate_config

    plain = OmegaConf.to_container(cfg, resolve=True)
    root_cfg = validate_config(cast(dict[str, Any], plain))

    # ------------------------------------------------------------------ #
    # 2. Seed — must happen before any model or data construction          #
    # ------------------------------------------------------------------ #
    from slm_research.training.seed import set_seed

    set_seed(root_cfg.run.seed)

    # ------------------------------------------------------------------ #
    # 3. Run identity                                                       #
    # ------------------------------------------------------------------ #
    from slm_research.tracking.run_metadata import (
        build_run_name,
        capture_git_hash,
        capture_system_info,
        generate_run_id,
    )

    run_id = generate_run_id()
    git_hash = capture_git_hash()
    system_info = capture_system_info()
    run_name = build_run_name(
        model_name=root_cfg.model.name,
        lora_rank=root_cfg.lora.r,
        precision=root_cfg.training.precision,
        run_id=run_id,
    )

    # Hydra's output dir for this run (e.g. outputs/2026-07-04/14-30-00/)
    from hydra.core.hydra_config import HydraConfig
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger.info("Run: %s  output_dir=%s", run_name, output_dir)

    # ------------------------------------------------------------------ #
    # 4. W&B init                                                          #
    # ------------------------------------------------------------------ #
    from slm_research.tracking.wandb_logger import WandbLogger

    # Check if we are resuming a previous run
    resume_from: str | None = cfg.get("resume_from", None)
    existing_run_id: str | None = None
    if resume_from is not None:
        # Recover run_id from the checkpoint's training_state.pt
        import torch
        state = torch.load(Path(resume_from) / "training_state.pt", map_location="cpu")
        existing_run_id = state.get("run_id", run_id)
        run_id = existing_run_id  # Keep W&B run consistent with saved run

    wandb_logger = WandbLogger(
        logging_cfg=root_cfg.logging,
        root_cfg=root_cfg,
        run_name=run_name,
        run_id=run_id,
        git_hash=git_hash,
        system_info=system_info,
        resume=(resume_from is not None),
    )

    # Patch run.name into the config so checkpointing uses a stable name
    root_cfg.run.name = run_name  # type: ignore[misc]

    try:
        # -------------------------------------------------------------- #
        # 5. Data pipeline                                                 #
        # -------------------------------------------------------------- #
        from slm_research.data.datamodule import build_data_module
        from slm_research.data.tokenization import load_tokenizer

        logger.info("Building dataset pipeline …")
        dm = build_data_module(
            mixture_cfg=root_cfg.data,
            model_cfg=root_cfg.model,
            training_cfg=root_cfg.training,
            eval_cfg=root_cfg.evaluation,
            seed=root_cfg.run.seed,
            num_proc=root_cfg.training.dataloader_num_workers or None,
        )

        # -------------------------------------------------------------- #
        # 6. Model                                                         #
        # -------------------------------------------------------------- #
        from slm_research.modeling.model_factory import (
            build_model,
            load_model_from_checkpoint,
        )

        logger.info("Building model …")
        if resume_from is not None:
            model = load_model_from_checkpoint(root_cfg, checkpoint_path=resume_from)
            model.train()
        else:
            model = build_model(root_cfg)

        # -------------------------------------------------------------- #
        # 7. Train                                                          #
        # -------------------------------------------------------------- #
        from slm_research.training.trainer import SLMTrainer

        trainer = SLMTrainer(
            model=model,
            data_module=dm,
            root_cfg=root_cfg,
            wandb_logger=wandb_logger,
            output_dir=output_dir,
            resume_from=resume_from,
            early_stop_patience=0,  # disabled by default; override via cfg if needed
        )

        logger.info("Starting training run: %s", run_name)
        final_metrics = trainer.train()
        logger.info("Training complete. Final metrics: %s", final_metrics)

    finally:
        # Always close the W&B run, even on exception
        wandb_logger.finish()


if __name__ == "__main__":
    main()
