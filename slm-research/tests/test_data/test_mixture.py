"""Unit tests for data/mixture.py::build_mixture.

The real configs/data/mixture.yaml combines streaming (IterableDataset) and
full (Dataset) sources. datasets.interleave_datasets() requires every input
to be the same type and raises otherwise, so build_mixture must normalize
before calling it — these tests exercise exactly that mixed-type path, plus
the two same-type paths it must leave alone.
"""
from __future__ import annotations

from datasets import Dataset, IterableDataset

from slm_research.data.mixture import build_mixture
from slm_research.data.packing import pack_dataset, pack_iterable_dataset
from slm_research.utils.config_schema import DataSourceConfig, MixtureConfig


def _mixture_cfg(sources: list[DataSourceConfig], strategy: str = "proportional_interleave") -> MixtureConfig:
    # MixtureConfig normally requires exactly 8 sources (see config_schema's
    # validator) — model_construct bypasses validation so tests can use a
    # small, focused source list instead.
    return MixtureConfig.model_construct(
        sampling_strategy=strategy,
        packing_granularity="within_dataset",
        sequence_length=4,
        sources=sources,
    )


def _packed_full(values: list[list[int]]) -> Dataset:
    return pack_dataset(Dataset.from_dict({"input_ids": values}), sequence_length=4)


def _packed_streaming(values: list[list[int]]) -> IterableDataset:
    def gen():
        for row in values:
            yield {"input_ids": row}

    return pack_iterable_dataset(IterableDataset.from_generator(gen), sequence_length=4)


def test_build_mixture_with_only_full_datasets_stays_a_dataset() -> None:
    d1 = _packed_full([[1, 2, 3, 4, 5, 6, 7, 8]])
    d2 = _packed_full([[9, 10, 11, 12]])
    cfg = _mixture_cfg([
        DataSourceConfig(name="wikitext", sample_count=2),
        DataSourceConfig(name="ag_news", sample_count=1),
    ])

    mixture = build_mixture([d1, d2], cfg, seed=0)

    assert isinstance(mixture, Dataset)


def test_build_mixture_with_only_streaming_datasets_stays_iterable() -> None:
    d1 = _packed_streaming([[1, 2, 3, 4, 5, 6, 7, 8]])
    d2 = _packed_streaming([[9, 10, 11, 12]])
    cfg = _mixture_cfg([
        DataSourceConfig(name="openwebtext", sample_count=2, streaming=True),
        DataSourceConfig(name="bookcorpusopen", sample_count=1, streaming=True),
    ])

    mixture = build_mixture([d1, d2], cfg, seed=0)

    assert isinstance(mixture, IterableDataset)


def test_build_mixture_with_mixed_types_normalizes_to_iterable_and_yields_both() -> None:
    full = _packed_full([[100, 200, 300, 400]])
    streaming = _packed_streaming([[1000, 1001, 1002, 1003]])
    # epoch_balanced -> stopping_strategy="all_exhausted", so both single-
    # example sources are guaranteed to appear; proportional_interleave's
    # "first_exhausted" would otherwise stop after whichever source (by
    # random draw) happens to run out first, making the count nondeterministic.
    cfg = _mixture_cfg(
        [
            DataSourceConfig(name="wikitext", sample_count=1),
            DataSourceConfig(name="openwebtext", sample_count=1, streaming=True),
        ],
        strategy="epoch_balanced",
    )

    mixture = build_mixture([full, streaming], cfg, seed=0)

    assert isinstance(mixture, IterableDataset)
    examples = list(mixture)
    assert len(examples) == 2
    seen_input_ids = {tuple(ex["input_ids"]) for ex in examples}
    assert seen_input_ids == {(100, 200, 300, 400), (1000, 1001, 1002, 1003)}
    assert all(ex["attention_mask"] == [1, 1, 1, 1] for ex in examples)


def test_build_mixture_epoch_balanced_uses_uniform_probabilities() -> None:
    full = _packed_full([[1, 2, 3, 4]])
    streaming = _packed_streaming([[5, 6, 7, 8]])
    cfg = _mixture_cfg(
        [
            DataSourceConfig(name="wikitext", sample_count=1),
            DataSourceConfig(name="openwebtext", sample_count=1, streaming=True),
        ],
        strategy="epoch_balanced",
    )

    mixture = build_mixture([full, streaming], cfg, seed=0)

    assert isinstance(mixture, IterableDataset)
    examples = list(mixture)
    assert len(examples) == 2
