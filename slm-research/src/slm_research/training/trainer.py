"""Core training loop for Qwen3-0.6B-Base + PEFT LoRA continued pretraining.

Responsibility: forward → loss → backward → gradient clipping → optimizer
step → scheduler step → logging → checkpoint → eval → repeat.

Uses HuggingFace Accelerate for device placement, mixed-precision AMP, and
gradient accumulation. This eliminates boilerplate (GradScaler, autocast
context managers, manual accumulation counting) while keeping the loop
fully custom so we control W&B metric names, checkpointing logic, and
evaluation triggers exactly.

Depends on: modeling/*, data/datamodule.py, tracking/wandb_logger.py,
            training/checkpointing.py, training/seed.py
Consumed by: scripts/train.py
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_scheduler

from slm_research.data.datamodule import DataModule
from slm_research.evaluation.evaluator import Evaluator
from slm_research.training.checkpointing import (
    get_latest_checkpoint,
    load_training_state,
    save_checkpoint,
    should_early_stop,
)
from slm_research.tracking.wandb_logger import WandbLogger
from slm_research.utils.config_schema import RootConfig

logger = logging.getLogger(__name__)


def _build_optimizer(model: Any, root_cfg: RootConfig) -> torch.optim.Optimizer:
    """Construct the optimizer from OptimizerConfig.

    Only LoRA adapter parameters are passed — frozen base parameters have
    requires_grad=False and are excluded by the filter.

    Args:
        model: PeftModel (already has frozen base weights).
        root_cfg: Validated RootConfig.

    Returns:
        Configured optimizer.
    """
    opt_cfg = root_cfg.optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if opt_cfg.name == "paged_adamw_8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(
                trainable_params,
                lr=opt_cfg.lr,
                betas=tuple(opt_cfg.betas),
                eps=opt_cfg.eps,
                weight_decay=opt_cfg.weight_decay,
            )
        except ImportError as exc:
            raise ImportError(
                "paged_adamw_8bit requires bitsandbytes. "
                "Install it or switch optimizer to adamw."
            ) from exc

    return torch.optim.AdamW(
        trainable_params,
        lr=opt_cfg.lr,
        betas=tuple(opt_cfg.betas),
        eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay,
        fused=torch.cuda.is_available(),  # fused AdamW is faster on CUDA
    )


def _estimate_training_steps(
    train_dl: DataLoader,
    root_cfg: RootConfig,
) -> int:
    """Estimate total training steps.

    Uses DataLoader length when available (non-streaming datasets).
    Falls back to an estimate from sample_count config for IterableDatasets.

    Args:
        train_dl: Train DataLoader (may not have __len__).
        root_cfg: Validated RootConfig.

    Returns:
        Estimated total optimizer steps across all epochs.
    """
    grad_accum = root_cfg.training.gradient_accumulation_steps
    num_epochs = root_cfg.training.num_epochs

    try:
        steps_per_epoch = math.ceil(len(train_dl) / grad_accum)
    except TypeError:
        # IterableDataset — estimate from sample_count config
        total_samples = sum(
            s.sample_count if isinstance(s.sample_count, int) else 1_000_000
            for s in root_cfg.data.sources
        )
        batch_size = root_cfg.training.per_device_train_batch_size
        steps_per_epoch = math.ceil(total_samples / (batch_size * grad_accum))
        logger.warning(
            "IterableDataset detected — estimated %d steps/epoch from sample_count config.",
            steps_per_epoch,
        )

    return steps_per_epoch * num_epochs


class SLMTrainer:
    """Training engine for Qwen3-0.6B-Base + PEFT LoRA continued pretraining.

    Usage::

        trainer = SLMTrainer(
            model=build_model(root_cfg),
            data_module=data_module,
            root_cfg=root_cfg,
            wandb_logger=wandb_logger,
            output_dir=output_dir,
        )
        trainer.train()

    Args:
        model: PeftModel from modeling.build_model.
        data_module: Configured DataModule with setup() already called.
        root_cfg: Fully validated RootConfig.
        wandb_logger: Initialised WandbLogger.
        output_dir: Directory for checkpoints and logs.
        resume_from: Optional path to a checkpoint directory to resume from.
        early_stop_patience: Stop if val/loss doesn't improve for this many
            consecutive evals. 0 disables early stopping.
    """

    def __init__(
        self,
        model: Any,
        data_module: DataModule,
        root_cfg: RootConfig,
        wandb_logger: WandbLogger,
        output_dir: str | Path,
        resume_from: str | Path | None = None,
        early_stop_patience: int = 0,
    ) -> None:
        self.model = model
        self.data_module = data_module
        self.root_cfg = root_cfg
        self.wl = wandb_logger
        self.output_dir = Path(output_dir)
        self.resume_from = Path(resume_from) if resume_from else None
        self.early_stop_patience = early_stop_patience

        self.training_cfg = root_cfg.training
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.val_loss_history: list[float] = []

        # Resolve mixed-precision mode for Accelerator.
        precision = self.training_cfg.precision
        if precision == "bf16":
            mixed_precision = "bf16"
        elif precision == "fp16":
            mixed_precision = "fp16"
        else:
            mixed_precision = "no"  # 4bit/8bit managed by BitsAndBytes

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=self.training_cfg.gradient_accumulation_steps,
            log_with=None,  # We handle logging via WandbLogger, not accelerate trackers
        )
        logger.info(
            "Accelerator: device=%s  mixed_precision=%s  grad_accum=%d",
            self.accelerator.device,
            mixed_precision,
            self.training_cfg.gradient_accumulation_steps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> dict[str, float]:
        """Run the full training loop.

        Returns:
            Dict with final train/loss and val/loss values.
        """
        train_dl = self.data_module.get_train_dataloader()
        val_dl = self.data_module.get_val_dataloader()

        optimizer = _build_optimizer(self.model, self.root_cfg)
        num_training_steps = _estimate_training_steps(train_dl, self.root_cfg)
        num_warmup_steps = int(
            self.root_cfg.training.warmup_ratio * num_training_steps
        )

        scheduler = get_scheduler(
            name=self.root_cfg.scheduler.name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.info(
            "Training plan: %d steps total  warmup=%d  lr_schedule=%s",
            num_training_steps, num_warmup_steps, self.root_cfg.scheduler.name,
        )

        # Prepare all objects with Accelerate (wraps model, optimizer, DLs, scheduler)
        self.model, optimizer, train_dl, val_dl, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dl, val_dl, scheduler
        )

        self.evaluator = Evaluator(
            model=self.model,
            val_dataloader=val_dl,
            eval_cfg=self.root_cfg.evaluation,
            device=self.accelerator.device,
        )

        # Resume from checkpoint if requested
        if self.resume_from is not None:
            self.global_step, self.current_epoch, _ = load_training_state(
                self.resume_from,
                optimizer,
                scheduler,
                device=self.accelerator.device,
            )
            logger.info(
                "Resumed from checkpoint: epoch=%d  step=%d",
                self.current_epoch, self.global_step,
            )

        final_metrics: dict[str, float] = {}

        for epoch in range(self.current_epoch, self.training_cfg.num_epochs):
            self.current_epoch = epoch
            epoch_metrics = self._train_epoch(train_dl, optimizer, scheduler, epoch)
            final_metrics.update(epoch_metrics)

            if self.early_stop_patience > 0 and should_early_stop(
                self.val_loss_history, self.early_stop_patience
            ):
                logger.info(
                    "Early stopping triggered after %d evals without improvement.",
                    self.early_stop_patience,
                )
                break

        # Final checkpoint
        ckpt_path = save_checkpoint(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=self.global_step,
            epoch=self.current_epoch,
            run_id=self.root_cfg.run.name or "run",
            output_dir=self.output_dir,
            accelerator=self.accelerator,
        )
        self.wl.log_checkpoint_artifact(
            ckpt_path, run_id=self.root_cfg.run.name or "run", step=self.global_step
        )

        return final_metrics

    # ------------------------------------------------------------------
    # Inner loops
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        train_dl: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
    ) -> dict[str, float]:
        """One full pass through the training data.

        Returns:
            Dict with the last logged train metrics for this epoch.
        """
        self.model.train()
        cfg = self.training_cfg
        last_metrics: dict[str, float] = {}
        running_loss = 0.0
        step_count = 0

        for batch in train_dl:
            step_start = time.perf_counter()

            with self.accelerator.accumulate(self.model):
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)

            # accumulate() internally tracks whether we're on an accumulation
            # boundary.  sync_gradients is True when the optimizer should step.
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step_elapsed = time.perf_counter() - step_start
                self.global_step += 1
                running_loss += loss.item() * cfg.gradient_accumulation_steps
                step_count += 1

                # --- Logging ---
                if self.global_step % cfg.logging_steps == 0:
                    avg_loss = running_loss / step_count
                    lr = scheduler.get_last_lr()[0]
                    tokens_per_step = (
                        cfg.per_device_train_batch_size
                        * self.root_cfg.data.sequence_length
                        * cfg.gradient_accumulation_steps
                    )
                    throughput = tokens_per_step / step_elapsed

                    gpu_mem_gb = 0.0
                    if torch.cuda.is_available():
                        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9
                        torch.cuda.reset_peak_memory_stats()

                    train_metrics = {
                        "train/loss": avg_loss,
                        "train/ppl": math.exp(min(avg_loss, 20)),
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
                        "train/throughput_tokens_per_sec": throughput,
                        "train/step_time_sec": step_elapsed,
                        "train/epoch": epoch + (step_count / max(step_count, 1)),
                        "gpu/memory_gb": gpu_mem_gb,
                    }
                    self.wl.log(train_metrics, step=self.global_step)
                    logger.info(
                        "step=%d  loss=%.4f  ppl=%.2f  lr=%.2e  grad_norm=%.3f  tok/s=%.0f",
                        self.global_step, avg_loss, math.exp(min(avg_loss, 20)),
                        lr, train_metrics["train/grad_norm"], throughput,
                    )
                    last_metrics = train_metrics
                    running_loss = 0.0
                    step_count = 0

                # --- Evaluation ---
                if self.global_step % cfg.eval_steps == 0:
                    val_metrics = self.evaluator.evaluate(global_step=self.global_step)
                    self.wl.log(val_metrics, step=self.global_step)
                    self.val_loss_history.append(val_metrics["val/loss"])
                    logger.info(
                        "val  step=%d  val/loss=%.4f  val/ppl=%.2f",
                        self.global_step,
                        val_metrics["val/loss"],
                        val_metrics["val/ppl"],
                    )
                    self.model.train()

                # --- Checkpoint ---
                if self.global_step % cfg.save_steps == 0:
                    ckpt_path = save_checkpoint(
                        model=self.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=self.global_step,
                        epoch=epoch,
                        run_id=self.root_cfg.run.name or "run",
                        output_dir=self.output_dir,
                        accelerator=self.accelerator,
                    )
                    self.wl.log_checkpoint_artifact(
                        ckpt_path,
                        run_id=self.root_cfg.run.name or "run",
                        step=self.global_step,
                    )

        return last_metrics
