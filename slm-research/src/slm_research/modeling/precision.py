"""Resolve precision settings for model loading and post-load casting.

Responsibility: translate the `precision` string from TrainingConfig into
the exact kwargs that AutoModelForCausalLM.from_pretrained expects. For
4-bit and 8-bit modes the quantization config must be supplied AT load time
(BitsAndBytes patches the linear layers before weights land in memory) — this
module returns those objects so model_factory.py can use them.

For bf16/fp16, dtype casting can happen post-load but is most efficient when
set at load time via torch_dtype.

Depends on: TrainingConfig.precision, ModelConfig.torch_dtype
Consumed by: model_factory.py
"""
from __future__ import annotations

import logging

import torch
from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)

# Maps the config string to a torch dtype.  "auto" is passed verbatim to
# transformers so it can pick the best dtype from the model's config.
_DTYPE_MAP: dict[str, torch.dtype | str] = {
    "auto": "auto",
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def get_bnb_config(precision: str) -> BitsAndBytesConfig | None:
    """Return a BitsAndBytesConfig for quantized precisions, None otherwise.

    Args:
        precision: One of "bf16", "fp16", "4bit", "8bit".

    Returns:
        BitsAndBytesConfig for 4bit/8bit, None for bf16/fp16/fp32.
    """
    if precision == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,   # nested quantization saves ~0.4 bits/param
            bnb_4bit_quant_type="nf4",        # NF4 minimises quantization error for normal distributions
        )
    if precision == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def get_torch_dtype(precision: str, model_torch_dtype: str) -> torch.dtype | str | None:
    """Resolve the torch_dtype argument for from_pretrained.

    For quantized modes (4bit/8bit) the compute dtype is inside
    BitsAndBytesConfig, so we return None here to avoid conflicts.

    Args:
        precision: TrainingConfig.precision ("bf16", "fp16", "4bit", "8bit").
        model_torch_dtype: ModelConfig.torch_dtype ("auto", "bf16", etc.).

    Returns:
        torch.dtype, "auto", or None.
    """
    if precision in ("4bit", "8bit"):
        return None  # BitsAndBytes owns dtype for quantized modes

    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16

    # For non-quantized modes, honour the model config's torch_dtype
    return _DTYPE_MAP.get(model_torch_dtype, "auto")


def is_quantized(precision: str) -> bool:
    """Return True for modes that require BitsAndBytes-specific PEFT prep."""
    return precision in ("4bit", "8bit")


def detect_attention_implementation() -> str:
    """Pick the best available attention implementation.

    Priority:
    1. Flash Attention 2 (fastest; requires flash-attn package + Ampere+ GPU)
    2. PyTorch SDPA (built-in since PyTorch 2.0; efficient on all CUDA GPUs)
    3. "eager" fallback

    Returns:
        String accepted by AutoModelForCausalLM.from_pretrained's
        attn_implementation argument.
    """
    try:
        import flash_attn  # noqa: F401
        logger.info("Flash Attention 2 detected — using flash_attention_2.")
        return "flash_attention_2"
    except ImportError:
        pass

    # SDPA is in torch.nn.functional.scaled_dot_product_attention (PyTorch ≥ 2.0)
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        logger.info("flash-attn not installed — using PyTorch SDPA.")
        return "sdpa"

    logger.warning("Neither flash-attn nor SDPA available — falling back to eager attention.")
    return "eager"
