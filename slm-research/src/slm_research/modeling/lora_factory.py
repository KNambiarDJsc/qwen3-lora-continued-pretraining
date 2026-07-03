"""Wrap a loaded base model with PEFT LoRA adapters.

Responsibility: the final model-construction step before training. Accepts
the model produced by model_factory.py and returns a PeftModel with only
the LoRA adapter weights set as trainable — the base model is frozen.

For quantized (4-bit/8-bit) models, peft.prepare_model_for_kbit_training
must be called before get_peft_model. This enables gradient flow through
quantized layers and handles gradient checkpointing setup.

Depends on: model_factory.py output, precision.py helpers, LoRAConfig
Consumed by: model_factory.build_model, evaluation/evaluator.py, benchmarking/*
"""
from __future__ import annotations

import logging

from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import PreTrainedModel

from slm_research.modeling.precision import is_quantized
from slm_research.utils.config_schema import LoRAConfig, TrainingConfig

logger = logging.getLogger(__name__)


def apply_lora(
    model: PreTrainedModel,
    lora_cfg: LoRAConfig,
    training_cfg: TrainingConfig,
) -> PeftModel:
    """Inject LoRA adapters into the model and freeze the base weights.

    Steps:
    1. If the model is quantized, call prepare_model_for_kbit_training.
    2. Build a LoraConfig from lora_cfg.
    3. Wrap with get_peft_model — makes LoRA params trainable, freezes rest.

    Gradient checkpointing is intentionally NOT enabled here. For quantized
    models it is handled by prepare_model_for_kbit_training. For bf16/fp16,
    build_model() calls gradient_checkpointing_enable() AFTER this function,
    which is the order the PEFT docs recommend.

    Args:
        model: Base model from model_factory.load_base_model.
        lora_cfg: Validated LoRAConfig from RootConfig.
        training_cfg: Validated TrainingConfig (precision, gradient_checkpointing).

    Returns:
        PeftModel with LoRA adapters injected.
    """
    if is_quantized(training_cfg.precision):
        logger.info(
            "Quantized precision (%s) — running prepare_model_for_kbit_training.",
            training_cfg.precision,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        target_modules=list(lora_cfg.target_modules),
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, peft_config)

    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info(
        "LoRA applied — rank=%d  alpha=%d  trainable=%.2fM / %.2fM params (%.2f%%)",
        lora_cfg.r,
        lora_cfg.alpha,
        trainable / 1e6,
        total / 1e6,
        100.0 * trainable / total,
    )
    return peft_model


def load_lora_checkpoint(base_model: PreTrainedModel, checkpoint_path: str) -> PeftModel:
    """Load a saved LoRA adapter onto a pre-loaded base model.

    Used by evaluation/evaluator.py and benchmarking/* to restore a specific
    checkpoint without going through the training loop. The base model weights
    are unchanged; only the adapter tensors are loaded.

    Args:
        base_model: Loaded (and possibly quantized) base model.
        checkpoint_path: Path to a directory produced by PeftModel.save_pretrained.

    Returns:
        PeftModel with the saved LoRA weights loaded.
    """
    logger.info("Loading LoRA checkpoint from %s", checkpoint_path)
    return PeftModel.from_pretrained(base_model, checkpoint_path)
