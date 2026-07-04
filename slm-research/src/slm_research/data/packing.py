"""Constant-length sequence packing, within a single dataset.

Packing happens PER DATASET before mixing (architecture spec Section 4,
open decision #3). This preserves per-source length distributions so that
val/ppl_by_length_bucket reflects genuine source characteristics rather
than mixture artifacts.

Design: concatenate all token ids for a dataset into one stream, then slice
into blocks of exactly `sequence_length`. The last partial block is discarded
so every block is a complete, uniform tensor row — no padding, no waste.

For non-streaming Dataset: batched map with a large group_size to minimise
boundary waste (~sequence_length tokens per batch boundary, negligible).
For streaming IterableDataset: stateful generator that carries a buffer
across batch boundaries, so no tokens are wasted.

Depends on: tokenization.py output
Consumed by: mixture.py
"""
from __future__ import annotations

import itertools
import logging
from typing import Any, Generator, Union

from datasets import Dataset, Features, IterableDataset, Sequence, Value

logger = logging.getLogger(__name__)

HFDataset = Union[Dataset, IterableDataset]

# Explicit output schema for both packers. Without this, Arrow infers the
# narrowest dtype that fits the *actual token values present* (e.g. int32 for
# input_ids, int8 for an all-ones attention_mask on the Dataset path, vs. a
# generator-based IterableDataset defaulting to int64) — two packed sources
# can end up with silently mismatched schemas, which makes mixture.py's
# interleave_datasets() raise when mixing a streaming and a non-streaming
# source. int64 matches what torch/collators expect anyway.
_PACKED_FEATURES = Features(
    {"input_ids": Sequence(Value("int64")), "attention_mask": Sequence(Value("int64"))}
)


# ---------------------------------------------------------------------------
# Non-streaming (Dataset) packer
# ---------------------------------------------------------------------------

def _pack_batch(examples: dict[str, Any], sequence_length: int) -> dict[str, Any]:
    """Batched map function: concatenate and chunk into fixed-length blocks.

    Any leftover tokens at the end of the batch (< sequence_length) are
    dropped. With a large batch_size, waste is negligible.
    """
    all_ids: list[int] = list(itertools.chain.from_iterable(examples["input_ids"]))
    n_blocks = len(all_ids) // sequence_length
    return {
        "input_ids": [
            all_ids[i * sequence_length : (i + 1) * sequence_length]
            for i in range(n_blocks)
        ],
        "attention_mask": [[1] * sequence_length for _ in range(n_blocks)],
    }


def pack_dataset(
    dataset: Dataset,
    sequence_length: int,
    group_size: int = 4096,
    num_proc: int | None = None,
) -> Dataset:
    """Pack a non-streaming Dataset into fixed-length blocks.

    Args:
        dataset: HF Dataset with an "input_ids" column (from tokenization.py).
        sequence_length: Target block length in tokens.
        group_size: Number of examples concatenated per map call. Larger
            values waste fewer tokens at batch boundaries.
        num_proc: Parallel workers for Dataset.map.

    Returns:
        Dataset with columns ["input_ids", "attention_mask"], all rows of
        shape (sequence_length,).
    """
    return dataset.map(
        _pack_batch,
        fn_kwargs={"sequence_length": sequence_length},
        batched=True,
        batch_size=group_size,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        features=_PACKED_FEATURES,
        desc=f"Packing (seq_len={sequence_length})",
    )


# ---------------------------------------------------------------------------
# Streaming (IterableDataset) packer
# ---------------------------------------------------------------------------

def _stateful_pack_generator(
    tokenized_iter,
    sequence_length: int,
) -> Generator[dict[str, list[int]], None, None]:
    """Yield fixed-length blocks, carrying a token buffer across examples.

    Zero tokens are wasted: the buffer persists between HF batches.
    The partial block at the very end of the stream is discarded.
    """
    buffer: list[int] = []
    for example in tokenized_iter:
        buffer.extend(example["input_ids"])
        while len(buffer) >= sequence_length:
            block = buffer[:sequence_length]
            buffer = buffer[sequence_length:]
            yield {
                "input_ids": block,
                "attention_mask": [1] * sequence_length,
            }


def pack_iterable_dataset(
    dataset: IterableDataset,
    sequence_length: int,
) -> IterableDataset:
    """Pack a streaming IterableDataset into fixed-length blocks.

    Uses a stateful generator so no tokens are wasted at batch boundaries.

    Args:
        dataset: IterableDataset with an "input_ids" column.
        sequence_length: Target block length in tokens.

    Returns:
        IterableDataset yielding {"input_ids": list[int], "attention_mask": list[int]}.
    """
    return IterableDataset.from_generator(
        _stateful_pack_generator,
        gen_kwargs={"tokenized_iter": dataset, "sequence_length": sequence_length},
        features=_PACKED_FEATURES,
    )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def pack(
    dataset: HFDataset,
    sequence_length: int,
    group_size: int = 4096,
    num_proc: int | None = None,
) -> HFDataset:
    """Pack tokenized sequences into fixed-length blocks.

    Dispatches to pack_dataset or pack_iterable_dataset based on type.

    Args:
        dataset: Tokenized HF Dataset or IterableDataset.
        sequence_length: Block length in tokens (from mixture_cfg.sequence_length).
        group_size: Batch size for non-streaming packing.
        num_proc: Parallel workers for non-streaming packing.

    Returns:
        Packed dataset with columns ["input_ids", "attention_mask"].
    """
    if isinstance(dataset, IterableDataset):
        return pack_iterable_dataset(dataset, sequence_length)
    return pack_dataset(dataset, sequence_length, group_size=group_size, num_proc=num_proc)
