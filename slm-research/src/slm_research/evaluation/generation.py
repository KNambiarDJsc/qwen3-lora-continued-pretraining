"""Qualitative sample generation for logging (val/examples).

Depends on: modeling/*
Consumed by: evaluation/evaluator.py, tracking/wandb_logger.py (as wandb.Table)
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from slm_research.utils.config_schema import InferenceConfig

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_samples(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    inference_cfg: InferenceConfig,
    device: str | torch.device = "cuda",
) -> list[str]:
    """Generate completions for a list of prompts.

    Args:
        model: PeftModel (or any model with a `.generate` method).
        tokenizer: Tokenizer matching the model, used to encode prompts and
            decode generated token ids.
        prompts: Conditioning prompt strings.
        inference_cfg: Validated InferenceConfig (max_new_tokens, temperature,
            top_p, do_sample).
        device: Device to run generation on.

    Returns:
        Completions (decoded text, prompt stripped), one per prompt, in the
        same order as `prompts`.
    """
    was_training = model.training
    model.eval()

    completions: list[str] = []
    try:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=inference_cfg.max_new_tokens,
                temperature=inference_cfg.temperature,
                top_p=inference_cfg.top_p,
                do_sample=inference_cfg.do_sample,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
            completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    finally:
        if was_training:
            model.train()

    logger.info("Generated %d qualitative samples.", len(completions))
    return completions
