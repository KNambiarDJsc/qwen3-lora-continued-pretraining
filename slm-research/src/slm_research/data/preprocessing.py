"""Text cleaning and normalization applied after dataset adapters.

Responsibility: whitespace normalization, unicode normalization, empty-sample
removal, and optional exact-duplicate removal. Operates on {"text": str} rows
produced by adapters. Does not tokenize.

Depends on: adapter output (loaders.py)
Consumed by: tokenization.py
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

from datasets import Dataset, IterableDataset


# Characters that are control codes but not standard whitespace — these appear
# in OCR'd text and web scrapes and confuse tokenizers.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def normalize_text(text: str) -> str:
    """Apply unicode NFC normalization, strip control chars and excess whitespace.

    Args:
        text: Raw string from a dataset adapter.

    Returns:
        Cleaned string, or empty string if the input was effectively empty.
    """
    text = unicodedata.normalize("NFC", text)
    text = _CTRL_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _clean_example(example: dict[str, Any]) -> dict[str, Any]:
    """Map function: normalize the text field of one example."""
    return {"text": normalize_text(example["text"])}


def _is_nonempty(example: dict[str, Any], min_chars: int = 20) -> bool:
    """Filter function: drop examples shorter than min_chars after cleaning."""
    return len(example["text"]) >= min_chars


def preprocess_dataset(
    dataset: Dataset | IterableDataset,
    min_chars: int = 20,
    num_proc: int | None = None,
) -> Dataset | IterableDataset:
    """Apply cleaning and empty-sample removal to a dataset.

    Args:
        dataset: HF Dataset or IterableDataset with a "text" column.
        min_chars: Drop examples whose cleaned text is shorter than this.
        num_proc: Number of parallel processes for Dataset.map (ignored for
            IterableDataset, which is inherently sequential).

    Returns:
        Cleaned dataset with the same type as the input.
    """
    is_iterable = isinstance(dataset, IterableDataset)

    if is_iterable:
        dataset = dataset.map(_clean_example)
        dataset = dataset.filter(_is_nonempty, fn_kwargs={"min_chars": min_chars})
    else:
        dataset = dataset.map(
            _clean_example,
            num_proc=num_proc,
            desc="Cleaning text",
        )
        dataset = dataset.filter(
            _is_nonempty,
            fn_kwargs={"min_chars": min_chars},
            num_proc=num_proc,
            desc="Filtering short samples",
        )

    return dataset


def deduplicate_dataset(
    dataset: Dataset,
    text_column: str = "text",
) -> Dataset:
    """Remove exact duplicates from a non-streaming Dataset using SHA-256 hashes.

    Not applicable to IterableDataset (no random access / no full materialization).
    For streaming sources, deduplication is handled at the source level or skipped.

    Args:
        dataset: A fully-loaded HF Dataset.
        text_column: Column containing the text to hash.

    Returns:
        Dataset with exact duplicates removed.
    """
    seen: set[str] = set()
    keep: list[bool] = []

    for example in dataset:
        h = hashlib.sha256(example[text_column].encode("utf-8")).hexdigest()
        if h in seen:
            keep.append(False)
        else:
            seen.add(h)
            keep.append(True)

    indices = [i for i, k in enumerate(keep) if k]
    return dataset.select(indices)
