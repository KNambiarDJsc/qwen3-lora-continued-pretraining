"""Single-prompt text generation for interactive inference.

Responsibility: turn one prompt string into one generated completion using a
merged (or adapter-active) inference model, plus a pretty-printed CLI-ready
format. Distinct from evaluation/generation.py, which batch-generates
qualitative samples drawn from the validation set for W&B logging — this
module is the standalone user-facing "give me one prompt back" path.

Depends on: nothing beyond a loaded model/tokenizer (produced by inference/loader.py)
Consumed by: scripts/infer.py
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from slm_research.utils.config_schema import InferenceConfig

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    inference_cfg: InferenceConfig,
    device: str | torch.device = "cuda",
) -> str:
    """Generate one completion for one prompt.

    Greedy decoding when inference_cfg.do_sample is False — temperature and
    top_p are omitted from the generate() call in that case, since they have
    no effect under greedy search and recent transformers versions warn if
    they're passed anyway.

    Args:
        model: Loaded model (e.g. from inference/loader.py), in eval mode.
        tokenizer: Tokenizer matching the model.
        prompt: Conditioning prompt string.
        inference_cfg: Validated InferenceConfig.
        device: Device to run generation on.

    Returns:
        Decoded completion text (prompt stripped, special tokens skipped).
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": inference_cfg.max_new_tokens,
        "do_sample": inference_cfg.do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if inference_cfg.do_sample:
        gen_kwargs["temperature"] = inference_cfg.temperature
        gen_kwargs["top_p"] = inference_cfg.top_p

    output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    logger.info(
        "Generated %d new tokens for a %d-token prompt.",
        new_tokens.shape[0],
        inputs["input_ids"].shape[1],
    )
    return completion


def format_generation(prompt: str, completion: str, inference_cfg: InferenceConfig) -> str:
    """Format a prompt/completion pair for pretty CLI output.

    Args:
        prompt: Conditioning prompt string.
        completion: Model-generated completion.
        inference_cfg: Validated InferenceConfig — decoding params are echoed
            in the header for reproducibility.

    Returns:
        A multi-line, human-readable string block.
    """
    decoding = (
        "greedy"
        if not inference_cfg.do_sample
        else f"sampling (temperature={inference_cfg.temperature}, top_p={inference_cfg.top_p})"
    )
    lines = [
        "=" * 80,
        f"Decoding: {decoding}  max_new_tokens={inference_cfg.max_new_tokens}",
        "-" * 80,
        "Prompt:",
        prompt,
        "-" * 80,
        "Completion:",
        completion,
        "=" * 80,
    ]
    return "\n".join(lines)
