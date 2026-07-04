"""Unit tests for data/packing.py.

Covers the packing logic itself (fixed-length blocks, no waste for the
streaming path, partial-block dropping) and the explicit output schema that
both packers now share — the schema is what makes mixture.py's
interleave_datasets() call safe when mixing a streaming and a non-streaming
source (see test_data/test_mixture.py for the mixed-interleave case).
"""
from __future__ import annotations

from datasets import Dataset, IterableDataset

from slm_research.data.packing import pack, pack_dataset, pack_iterable_dataset


def test_pack_dataset_produces_fixed_length_blocks_and_drops_remainder() -> None:
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4, 5, 6, 7]]})  # 7 tokens, seq_len=4 -> 1 block, 3 dropped

    packed = pack_dataset(ds, sequence_length=4)

    assert len(packed) == 1
    assert packed[0]["input_ids"] == [1, 2, 3, 4]
    assert packed[0]["attention_mask"] == [1, 1, 1, 1]


def test_pack_iterable_dataset_carries_buffer_across_examples_without_waste() -> None:
    def gen():
        yield {"input_ids": [1, 2, 3]}
        yield {"input_ids": [4, 5, 6, 7, 8]}

    packed = pack_iterable_dataset(IterableDataset.from_generator(gen), sequence_length=4)

    blocks = [ex["input_ids"] for ex in packed]
    # 8 total tokens, seq_len=4 -> 2 full blocks, nothing wasted at the
    # example-1/example-2 boundary (unlike a per-example non-streaming pack).
    assert blocks == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_pack_dataset_and_pack_iterable_dataset_share_output_schema() -> None:
    """The dtype alignment bug: Arrow narrows dtypes per actual value range
    unless an explicit schema is given, so the two packers could silently
    diverge (int32/int8 vs int64) and break mixture.py's interleave call."""
    full = pack_dataset(Dataset.from_dict({"input_ids": [[1, 2, 3, 4]]}), sequence_length=4)

    def gen():
        yield {"input_ids": [9, 9, 9, 9]}

    streaming = pack_iterable_dataset(IterableDataset.from_generator(gen), sequence_length=4)

    assert full.features == streaming.features


def test_pack_dispatches_on_dataset_type() -> None:
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4]]})
    assert isinstance(pack(ds, sequence_length=4), Dataset)

    def gen():
        yield {"input_ids": [1, 2, 3, 4]}

    assert isinstance(pack(IterableDataset.from_generator(gen), sequence_length=4), IterableDataset)
