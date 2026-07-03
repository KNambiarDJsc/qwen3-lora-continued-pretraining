"""Load Qwen3-0.6B-Base and assemble the full model pipeline.

Responsibility: instantiate the base model with the right dtype, device map,
and attention implementation; hand it to lora_factory.py for adapter
injection; configure gradient checkpointing. Exposes build_model() as the
single callable that every downstream module imports.

Pipeline (build_model):
    precision.py resolves load kwargs
         ↓
    AutoModelForCausalLM.from_pretrained  (this module)
         ↓
    lora_factory.apply_lora  (freezes base, injects adapters)
         ↓
    gradient checkpointing  (if training_cfg.gradient_checkpointing)

Depends on: configs/model/*, configs/training/*, configs/lora/*
Consumed by: scripts/train.py, scripts/evaluate.py, scripts/benchmark.py
"""
from __future__ import annotations

import logging
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedModel

from slm_research.modeling.lora_factory import apply_lora, load_lora_checkpoint
from slm_research.modeling.precision import (
    detect_attention_implementation,
    get_bnb_config,
    get_torch_dtype,
    is_quantized,
)
from slm_research.utils.config_schema import LoRAConfig, ModelConfig, RootConfig, TrainingConfig

logger = logging.getLogger(__name__)


def load_base_model(
    model_cfg: ModelConfig,
    training_cfg: TrainingConfig,
    cache_dir: str | None = None,
) -> PreTrainedModel:
    """Instantiate Qwen3-0.6B-Base with precision and attention settings.

    Quantization config and torch_dtype are resolved before the call so that
    BitsAndBytes can patch linear layers during weight loading (required for
    4bit/8bit — casting after the fact does not work).

    Args:
        model_cfg: Validated ModelConfig (name, revision, device_map, …).
        training_cfg: Validated TrainingConfig (precision, …).
        cache_dir: Optional local directory for the Hugging Face model cache.

    Returns:
        Loaded, unfrozen base model (no LoRA, no adapters yet).
    """
    bnb_config = get_bnb_config(training_cfg.precision)
    torch_dtype = get_torch_dtype(training_cfg.precision, model_cfg.torch_dtype)
    attn_impl = detect_attention_implementation()

    load_kwargs: dict = {
        "pretrained_model_name_or_path": model_cfg.name,
        "revision": model_cfg.revision,
        "trust_remote_code": model_cfg.trust_remote_code,
        "device_map": model_cfg.device_map,
        "attn_implementation": attn_impl,
    }
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir

    logger.info(
        "Loading base model: %s  revision=%s  precision=%s  attn=%s  device_map=%s",
        model_cfg.name, model_cfg.revision, training_cfg.precision, attn_impl,
        model_cfg.device_map,
    )

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    # For non-quantized models with gradient checkpointing we need input
    # gradients to flow through the embedding layer to the first LoRA layer.
    # prepare_model_for_kbit_training handles this for quantized models;
    # for fp/bf models we do it manually here, before LoRA is applied.
    if not is_quantized(training_cfg.precision) and training_cfg.gradient_checkpointing:
        model.enable_input_require_grads()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Base model loaded — %.3fB parameters.", n_params / 1e9)
    return model


def build_model(
    root_cfg: RootConfig,
    cache_dir: str | None = None,
) -> PeftModel:
    """End-to-end model assembly: base → LoRA → gradient checkpointing.

    This is the single function every script imports. After calling it you
    have a PeftModel where only the LoRA adapter weights are trainable.

    Args:
        root_cfg: Fully validated RootConfig (from config_schema.validate_config).
        cache_dir: Optional Hugging Face cache directory.

    Returns:
        PeftModel ready for training or evaluation.
    """
    model = load_base_model(root_cfg.model, root_cfg.training, cache_dir=cache_dir)
    model = apply_lora(model, root_cfg.lora, root_cfg.training)

    # Gradient checkpointing for non-quantized models.
    # For quantized models it is already enabled inside apply_lora via
    # prepare_model_for_kbit_training — enabling it again here is a no-op.
    if root_cfg.training.gradient_checkpointing and not is_quantized(root_cfg.training.precision):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled.")

    return model


def load_model_from_checkpoint(
    root_cfg: RootConfig,
    checkpoint_path: str | Path,
    cache_dir: str | None = None,
) -> PeftModel:
    """Load a base model and restore a saved LoRA checkpoint.

    Used by evaluate.py and benchmark.py to load a specific run's weights
    without running the training loop.

    Args:
        root_cfg: Validated RootConfig (determines base model and precision).
        checkpoint_path: Path to a directory saved by PeftModel.save_pretrained.
        cache_dir: Optional Hugging Face cache directory.

    Returns:
        PeftModel with the checkpoint's LoRA weights loaded, in eval mode.
    """
    base = load_base_model(root_cfg.model, root_cfg.training, cache_dir=cache_dir)
    model = load_lora_checkpoint(base, str(checkpoint_path))
    model.eval()
    logger.info("Checkpoint loaded from %s — model in eval mode.", checkpoint_path)
    return model
