"""Tokenizer loading and dataset tokenization.

Responsibility: instantiate the Qwen3 tokenizer once, provide a function
that maps it over an HF dataset's "text" column. No packing happens here —
sequences remain variable-length; packing.py handles chunking.

Depends on: preprocessing.py output, configs/model/* (tokenizer.name)
Consumed by: packing.py
"""
from __future__ import annotations

import logging
from typing import Any, Union

from datasets import Dataset, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from slm_research.utils.config_schema import ModelConfig

logger = logging.getLogger(__name__)

HFDataset = Union[Dataset, IterableDataset]


def load_tokenizer(model_cfg: ModelConfig) -> PreTrainedTokenizerBase:
    """Instantiate the tokenizer from the model config.

    Args:
        model_cfg: Validated ModelConfig from RootConfig.

    Returns:
        A PreTrainedTokenizerBase ready for encoding.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.tokenizer.name,
        trust_remote_code=model_cfg.trust_remote_code,
        use_fast=True,
    )
    # Qwen3 tokenizer has an EOS token; expose it explicitly so the
    # tokenization step can append it consistently.
    if tokenizer.eos_token_id is None:
        raise ValueError(
            f"Tokenizer {model_cfg.tokenizer.name!r} has no EOS token. "
            "Set tokenizer.eos_token in the model config."
        )
    logger.info(
        "Loaded tokenizer %s  vocab_size=%d  eos_token_id=%d",
        model_cfg.tokenizer.name,
        tokenizer.vocab_size,
        tokenizer.eos_token_id,
    )
    return tokenizer


def tokenize_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizerBase,
    add_eos_token: bool = True,
    num_proc: int | None = None,
    text_column: str = "text",
) -> HFDataset:
    """Tokenize a dataset's text column into variable-length token ID lists.

    Each example becomes {"input_ids": list[int]}. No truncation or padding —
    packing.py will chunk the concatenated stream into fixed-length blocks.

    Args:
        dataset: HF Dataset or IterableDataset with a text column.
        tokenizer: Loaded tokenizer.
        add_eos_token: If True, append the EOS token id to every sequence.
            This boundary marker lets the model learn to distinguish documents
            during packing, even though sequences are concatenated.
        num_proc: Worker count for Dataset.map (unused for IterableDataset).
        text_column: Name of the column containing raw text.

    Returns:
        Dataset with columns ["input_ids"] only.
    """
    eos_id: int = tokenizer.eos_token_id

    def _tokenize(examples: dict[str, Any]) -> dict[str, Any]:
        encoded = tokenizer(
            examples[text_column],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        if add_eos_token:
            encoded["input_ids"] = [ids + [eos_id] for ids in encoded["input_ids"]]
        return {"input_ids": encoded["input_ids"]}

    is_iterable = isinstance(dataset, IterableDataset)
    remove_cols = [text_column]

    if is_iterable:
        return dataset.map(
            _tokenize,
            batched=True,
            remove_columns=remove_cols,
        )
    else:
        return dataset.map(
            _tokenize,
            batched=True,
            remove_columns=remove_cols,
            num_proc=num_proc,
            desc="Tokenizing",
        )
