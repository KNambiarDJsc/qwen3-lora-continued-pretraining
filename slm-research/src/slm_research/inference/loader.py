"""Checkpoint → tokenizer → base model → LoRA adapter → merged model.

Responsibility: assemble a single ready-to-generate model from a saved LoRA
checkpoint. Standalone — depends only on modeling/* and data/tokenization.py,
never on training/trainer.py, mirroring the isolation rule that already
applies to evaluation/* and benchmarking/*: inference consumes checkpoints
as artifacts.

Depends on: modeling/model_factory.py, modeling/lora_factory.py,
    modeling/precision.py, data/tokenization.py
Consumed by: scripts/infer.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

from slm_research.data.tokenization import load_tokenizer
from slm_research.modeling.lora_factory import load_lora_checkpoint
from slm_research.modeling.model_factory import load_base_model
from slm_research.modeling.precision import is_quantized
from slm_research.utils.config_schema import RootConfig

logger = logging.getLogger(__name__)


def load_model_for_inference(
    root_cfg: RootConfig,
    checkpoint_path: str | Path,
    cache_dir: str | None = None,
) -> tuple[Any, PreTrainedTokenizerBase]:
    """Load tokenizer + base model + LoRA adapter, merged into one model.

    Pipeline: tokenizer -> base model -> LoRA adapter -> merge_and_unload.
    Merging collapses the adapter delta into the base weights so generation
    runs a plain forward pass with no PEFT dispatch overhead.

    Quantized checkpoints (4bit/8bit) are NOT merged: BitsAndBytes linear
    layers can't absorb a LoRA delta in place, so for those precisions the
    adapter is left active on top of the quantized base instead.

    Args:
        root_cfg: Validated RootConfig (determines base model, tokenizer, precision).
        checkpoint_path: Path to a directory saved by PeftModel.save_pretrained.
        cache_dir: Optional Hugging Face cache directory.

    Returns:
        (model, tokenizer) — model in eval mode, ready for .generate().
    """
    tokenizer = load_tokenizer(root_cfg.model)

    logger.info("Loading base model …")
    base = load_base_model(root_cfg.model, root_cfg.training, cache_dir=cache_dir)

    logger.info("Loading LoRA adapter from %s …", checkpoint_path)
    model = load_lora_checkpoint(base, str(checkpoint_path))

    if is_quantized(root_cfg.training.precision):
        logger.warning(
            "Precision=%s is quantized — skipping adapter merge; running "
            "with the LoRA adapter active on top of the quantized base.",
            root_cfg.training.precision,
        )
    else:
        logger.info("Merging LoRA adapter into base weights …")
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer
