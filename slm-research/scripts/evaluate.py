"""Entrypoint: evaluate.

Thin orchestration only per architecture spec — wires configs to the
appropriate module(s) in src/slm_research/. No business logic lives here.

Checkpoint-driven and standalone: re-evaluates any past checkpoint without
retraining (architecture spec Section 6).

Run:
    python scripts/evaluate.py +checkpoint_path=outputs/checkpoints/run_abc_epoch0_step500
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
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError(
            "scripts/evaluate.py requires +checkpoint_path=<dir>, pointing to "
            "a checkpoint directory saved by training/checkpointing.py."
        )

    from slm_research.utils.config_schema import validate_config

    plain = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(plain, dict):
        plain.pop("checkpoint_path", None)
    root_cfg = validate_config(cast(dict[str, Any], plain))

    # ------------------------------------------------------------------ #
    # 2. Seed — must happen before any model or data construction          #
    # ------------------------------------------------------------------ #
    from slm_research.training.seed import set_seed

    set_seed(root_cfg.run.seed)

    # ------------------------------------------------------------------ #
    # 3. Run identity — reuse the checkpoint's run_id so the eval metrics  #
    #    land on the same W&B run as the training that produced it.        #
    # ------------------------------------------------------------------ #
    import torch

    from slm_research.tracking.run_metadata import build_run_name, capture_git_hash, capture_system_info

    state = torch.load(Path(checkpoint_path) / "training_state.pt", map_location="cpu")
    run_id = state.get("run_id", "unknown")
    git_hash = capture_git_hash()
    system_info = capture_system_info()
    run_name = build_run_name(
        model_name=root_cfg.model.name,
        lora_rank=root_cfg.lora.r,
        precision=root_cfg.training.precision,
        run_id=run_id,
    )
    global_step = state.get("global_step", 0)
    logger.info("Evaluating checkpoint: %s  run=%s  step=%d", checkpoint_path, run_name, global_step)

    # ------------------------------------------------------------------ #
    # 4. W&B init — resume the training run so val/* lands on its charts   #
    # ------------------------------------------------------------------ #
    from slm_research.tracking.wandb_logger import WandbLogger

    wandb_logger = WandbLogger(
        logging_cfg=root_cfg.logging,
        root_cfg=root_cfg,
        run_name=run_name,
        run_id=run_id,
        git_hash=git_hash,
        system_info=system_info,
        resume=True,
    )

    try:
        # -------------------------------------------------------------- #
        # 5. Data pipeline — validation split only                         #
        # -------------------------------------------------------------- #
        from slm_research.data.datamodule import build_data_module

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
        # 6. Model — base + this checkpoint's LoRA adapter                 #
        # -------------------------------------------------------------- #
        from slm_research.modeling.model_factory import load_model_from_checkpoint

        logger.info("Loading checkpoint …")
        model = load_model_from_checkpoint(root_cfg, checkpoint_path=checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # -------------------------------------------------------------- #
        # 7. Metrics                                                        #
        # -------------------------------------------------------------- #
        from slm_research.evaluation.evaluator import Evaluator

        evaluator = Evaluator(
            model=model,
            val_dataloader=dm.get_val_dataloader(),
            eval_cfg=root_cfg.evaluation,
            device=device,
        )

        logger.info("Running validation pass …")
        metrics = evaluator.evaluate(global_step=global_step)
        wandb_logger.log(metrics, step=global_step)
        logger.info("Metrics: %s", metrics)

        # -------------------------------------------------------------- #
        # 8. Qualitative samples (val/examples)                            #
        # -------------------------------------------------------------- #
        logger.info("Generating qualitative samples …")
        prompts, completions = evaluator.generate_qualitative_samples(
            tokenizer=dm.tokenizer,
            inference_cfg=root_cfg.inference,
        )
        wandb_logger.log_generation_table(prompts, completions, step=global_step)

    finally:
        # Always close the W&B run, even on exception
        wandb_logger.finish()


if __name__ == "__main__":
    main()
