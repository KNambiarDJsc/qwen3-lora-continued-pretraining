"""Load raw datasets from the Hugging Face Hub, one source at a time.

Responsibility: call load_dataset with the right arguments (streaming vs.
full, config name, split), apply the source's adapter to standardize to
{"text": str}, and apply the sample_count limit. No cleaning or tokenization.

Depends on: registry.py, configs/data/mixture.yaml (via DataSourceConfig)
Consumed by: datamodule.py
"""
from __future__ import annotations

import logging
from typing import Union

from datasets import Dataset, IterableDataset, load_dataset

from slm_research.data.registry import get_dataset_info
from slm_research.utils.config_schema import DataSourceConfig

logger = logging.getLogger(__name__)

HFDataset = Union[Dataset, IterableDataset]

# Proxy sample count for sources with sample_count="all" when computing weights.
_ALL_PROXY_COUNT: int = 1_000_000


def _sample_count_int(source: DataSourceConfig) -> int:
    """Return the numeric sample count, substituting proxy for 'all'."""
    return source.sample_count if isinstance(source.sample_count, int) else _ALL_PROXY_COUNT


def load_source(
    source: DataSourceConfig,
    split: str,
    seed: int,
    cache_dir: str | None = None,
) -> HFDataset:
    """Load and adapt one dataset source.

    Performs these steps in order:
    1. load_dataset (streaming or full)
    2. apply dataset adapter → {"text": str}
    3. shuffle (full) or shuffle-buffer (streaming)
    4. apply sample_count limit

    Args:
        source: One entry from MixtureConfig.sources.
        split: HF split name, e.g. "train" or "validation".
        seed: RNG seed for shuffling.
        cache_dir: Optional local directory for HF dataset cache.

    Returns:
        Dataset or IterableDataset containing only a "text" column.
    """
    info = get_dataset_info(source.name)
    load_kwargs: dict = {
        "streaming": source.streaming,
        "trust_remote_code": False,
    }
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    if source.config is not None:
        load_kwargs["name"] = source.config

    logger.info(
        "Loading %s (hf_path=%s, config=%s, split=%s, streaming=%s)",
        source.name, info.hf_path, source.config, split, source.streaming,
    )

    raw: HFDataset = load_dataset(info.hf_path, split=split, **load_kwargs)

    # Apply adapter: standardize all columns to {"text": str}
    adapter = info.adapter
    if source.streaming:
        remove_columns = list(raw.features.keys()) if getattr(raw, "features", None) else None
        dataset: HFDataset = raw.map(adapter, remove_columns=remove_columns)
        dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
        if isinstance(source.sample_count, int):
            dataset = dataset.take(source.sample_count)
    else:
        assert isinstance(raw, Dataset)
        col_names = raw.column_names
        dataset = raw.map(adapter, remove_columns=col_names, desc=f"Adapting {source.name}")
        dataset = dataset.shuffle(seed=seed)
        if isinstance(source.sample_count, int):
            n = min(source.sample_count, len(dataset))
            dataset = dataset.select(range(n))

    return dataset


def load_source_splits(
    source: DataSourceConfig,
    val_split_fraction: float,
    seed: int,
    cache_dir: str | None = None,
) -> tuple[HFDataset, HFDataset]:
    """Load train and validation splits for one source.

    Strategy:
    - If the dataset has a pre-existing validation split in the registry, use it.
    - Otherwise, carve val_split_fraction from the training data.
      For streaming sources this uses take/skip on the sample count.

    Args:
        source: DataSourceConfig entry from MixtureConfig.
        val_split_fraction: Fraction of training data to use for validation.
        seed: RNG seed.
        cache_dir: Optional HF cache directory.

    Returns:
        (train_dataset, val_dataset) tuple.
    """
    info = get_dataset_info(source.name)

    if info.default_val_split is not None:
        # Use the dataset's own validation split
        train_ds = load_source(source, split=info.default_train_split, seed=seed, cache_dir=cache_dir)
        val_ds = load_source(source, split=info.default_val_split, seed=seed, cache_dir=cache_dir)
        return train_ds, val_ds

    # Carve validation from training data
    n_total = _sample_count_int(source)
    n_val = max(1, int(n_total * val_split_fraction))
    n_train = n_total - n_val

    if source.streaming:
        full = load_source(
            DataSourceConfig(
                name=source.name,
                config=source.config,
                weight=source.weight,
                sample_count=n_total,
                streaming=source.streaming,
            ),
            split=info.default_train_split,
            seed=seed,
            cache_dir=cache_dir,
        )
        # IterableDataset supports take/skip for split without materializing
        val_ds = full.take(n_val)
        train_ds = full.skip(n_val)
        if n_train < n_total:
            train_ds = train_ds.take(n_train)
    else:
        full: Dataset = load_source(
            DataSourceConfig(
                name=source.name,
                config=source.config,
                weight=source.weight,
                sample_count="all" if source.sample_count == "all" else n_total,
                streaming=False,
            ),
            split=info.default_train_split,
            seed=seed,
            cache_dir=cache_dir,
        )
        splits = full.train_test_split(test_size=val_split_fraction, seed=seed)
        train_ds, val_ds = splits["train"], splits["test"]

    return train_ds, val_ds
