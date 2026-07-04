"""Entrypoint: download_data.

Thin orchestration only per architecture spec — wires configs to the
appropriate module(s) in src/slm_research/. No business logic lives here
(the actual loading logic is entirely in data/loaders.py; this script just
calls it per source and iterates the result to force materialization).

Warms the local Hugging Face cache for every source in the configured
mixture (configs/data/mixture.yaml), so a later train/evaluate/preprocess
run doesn't pay download latency mid-pipeline.

For non-streaming sources, load_source_splits already fully materializes
the Dataset (load_dataset + shuffle), so iterating it here just forces that
to happen now rather than lazily. For streaming sources (openwebtext,
bookcorpusopen, fineweb_edu), "downloading" means iterating through the
mixture-configured sample_count bound once — that's the same bounded slice
training will read, so it warms the Hugging Face Hub's local shard cache for
exactly that data without materializing the full (much larger) source to
disk, which would defeat the point of streaming.

Run:
    python scripts/download_data.py
    python scripts/download_data.py data.sources.0.sample_count=1000
"""
from __future__ import annotations

import logging
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from slm_research.utils.config_schema import validate_config

    plain = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(plain, dict):
        raise TypeError("Config must resolve to a dictionary")
    plain.pop("checkpoint", None)
    plain.pop("prompt", None)
    root_cfg = validate_config(cast(dict[str, Any], plain))

    from slm_research.data.loaders import load_source_splits

    sources = root_cfg.data.sources
    logger.info("Downloading/caching %d mixture sources …", len(sources))

    for i, source in enumerate(sources, start=1):
        logger.info("[%d/%d] %s (streaming=%s) …", i, len(sources), source.name, source.streaming)

        train_ds, val_ds = load_source_splits(
            source=source,
            val_split_fraction=root_cfg.evaluation.val_split_fraction,
            seed=root_cfg.run.seed,
        )
        n_train = sum(1 for _ in train_ds)
        n_val = sum(1 for _ in val_ds)

        logger.info("%s: cached %d train / %d val examples.", source.name, n_train, n_val)

    logger.info("All %d sources downloaded/cached.", len(sources))


if __name__ == "__main__":
    main()
