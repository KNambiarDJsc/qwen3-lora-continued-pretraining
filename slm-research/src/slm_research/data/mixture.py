"""Weighted interleaving of per-dataset packed sequences into one stream.

Responsibility: given N packed datasets (one per source), interleave them
according to mixture_cfg.sampling_strategy and per-source weights.
Packing is complete before this module runs (architecture spec Section 4).

Sampling strategies:
  proportional_interleave — datasets.interleave_datasets with computed
      probabilities. Sources sampled proportionally by weight/sample_count.
  epoch_balanced — uniform probabilities, stopping_strategy="all_exhausted"
      so every source completes one epoch before the mixture stops.

Depends on: packing.py output, MixtureConfig
Consumed by: datamodule.py
"""
from __future__ import annotations

import logging
from typing import Union

from datasets import Dataset, IterableDataset, interleave_datasets

from slm_research.utils.config_schema import DataSourceConfig, MixtureConfig

logger = logging.getLogger(__name__)

HFDataset = Union[Dataset, IterableDataset]

# Proxy sample count for sources that specify sample_count="all".
_ALL_PROXY: int = 1_000_000


def _compute_probabilities(sources: list[DataSourceConfig]) -> list[float]:
    """Derive interleaving probabilities from sample_count when weights are all None.

    Treats sample_count="all" as _ALL_PROXY for proportion computation.

    Args:
        sources: List of DataSourceConfig entries from MixtureConfig.

    Returns:
        Normalized probability list (sums to 1.0).
    """
    raw = [
        float(s.sample_count) if isinstance(s.sample_count, int) else float(_ALL_PROXY)
        for s in sources
    ]
    total = sum(raw)
    return [r / total for r in raw]


def build_mixture(
    packed_datasets: list[HFDataset],
    mixture_cfg: MixtureConfig,
    seed: int,
) -> HFDataset:
    """Interleave packed per-source datasets into a single stream.

    Args:
        packed_datasets: Packed Dataset or IterableDataset per source, in the
            same order as mixture_cfg.sources.
        mixture_cfg: Validated MixtureConfig from RootConfig.
        seed: RNG seed for deterministic interleaving.

    Returns:
        A single interleaved Dataset or IterableDataset.
    """
    sources = mixture_cfg.sources
    strategy = mixture_cfg.sampling_strategy

    # Resolve interleaving probabilities
    all_weights_none = all(s.weight is None for s in sources)
    if all_weights_none:
        probabilities = _compute_probabilities(sources)
        logger.info(
            "Mixture weights derived from sample_count: %s",
            {s.name: f"{p:.4f}" for s, p in zip(sources, probabilities)},
        )
    else:
        total = sum(s.weight for s in sources)  # type: ignore[arg-type]
        probabilities = [s.weight / total for s in sources]  # type: ignore[operator]
        logger.info(
            "Mixture weights from config: %s",
            {s.name: f"{p:.4f}" for s, p in zip(sources, probabilities)},
        )

    if strategy == "proportional_interleave":
        stopping_strategy = "first_exhausted"
    else:  # epoch_balanced
        stopping_strategy = "all_exhausted"
        # Uniform probabilities for epoch-balanced
        n = len(packed_datasets)
        probabilities = [1.0 / n] * n

    logger.info(
        "Building mixture: strategy=%s  stopping=%s  n_sources=%d",
        strategy, stopping_strategy, len(packed_datasets),
    )

    # interleave_datasets requires every input to be the same type, but the
    # real mixture legitimately combines streaming sources (IterableDataset)
    # with fully-materialized ones (Dataset) — e.g. wikitext alongside
    # openwebtext. Normalize to IterableDataset only when the list is mixed,
    # so an all-Dataset or all-IterableDataset mixture is untouched (and the
    # former keeps shuffling/indexing support in DataModule).
    has_iterable = any(isinstance(d, IterableDataset) for d in packed_datasets)
    has_dataset = any(isinstance(d, Dataset) for d in packed_datasets)
    if has_iterable and has_dataset:
        logger.info("Mixture combines streaming and full sources — normalizing to IterableDataset.")
        packed_datasets = [
            d.to_iterable_dataset() if isinstance(d, Dataset) else d for d in packed_datasets
        ]

    return interleave_datasets(
        packed_datasets,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
