"""Entrypoint: preprocess.

Thin orchestration only per architecture spec — wires configs to the
appropriate module(s) in src/slm_research/. No business logic lives here:
loading, cleaning, tokenizing, and packing are entirely implemented in
data/loaders.py, preprocessing.py, tokenization.py, and packing.py — this
script just calls them per source, in the same order data/datamodule.py
does, and persists the result.

Writes each source's packed train/val Dataset to disk under
<run.output_dir>/preprocessed/<source_name>/{train,val}/, via
datasets.Dataset.save_to_disk. Streaming sources are bounded by their
configured sample_count (same bound training reads) and materialized into a
genuine on-disk Dataset in the process — save_to_disk isn't defined for a
lazy IterableDataset.

This does NOT wire into DataModule: scripts/train.py and scripts/evaluate.py
still build the pipeline on demand, in-process, every run. This is a
standalone, inspectable cache — see docs/dataset_guide.md.

Run:
    python scripts/preprocess.py
    python scripts/preprocess.py run.output_dir=my_cache
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from slm_research.utils.config_schema import validate_config

    plain = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(plain, dict):
        plain.pop("checkpoint", None)
        plain.pop("prompt", None)
    root_cfg = validate_config(cast(dict[str, Any], plain))

    from datasets import Dataset, IterableDataset

    from slm_research.data.loaders import load_source_splits
    from slm_research.data.packing import pack
    from slm_research.data.preprocessing import preprocess_dataset
    from slm_research.data.tokenization import load_tokenizer, tokenize_dataset

    tokenizer = load_tokenizer(root_cfg.model)
    out_root = Path(root_cfg.run.output_dir) / "preprocessed"
    sources = root_cfg.data.sources

    for i, source in enumerate(sources, start=1):
        logger.info("[%d/%d] Preprocessing source: %s", i, len(sources), source.name)

        train_raw, val_raw = load_source_splits(
            source=source,
            val_split_fraction=root_cfg.evaluation.val_split_fraction,
            seed=root_cfg.run.seed,
        )

        for split_name, raw in (("train", train_raw), ("val", val_raw)):
            clean = preprocess_dataset(raw)
            tokenized = tokenize_dataset(
                clean, tokenizer, add_eos_token=root_cfg.model.tokenizer.add_eos_token
            )
            packed = pack(tokenized, root_cfg.data.sequence_length)

            if isinstance(packed, IterableDataset):
                # Streaming sources are bounded by sample_count — the same
                # bound training reads — so collecting here doesn't
                # materialize an unbounded corpus, just this configured slice.
                packed = Dataset.from_list(list(packed))

            split_dir = out_root / source.name / split_name
            packed.save_to_disk(str(split_dir))
            logger.info(
                "Saved %s/%s: %d packed sequences -> %s",
                source.name, split_name, len(packed), split_dir,
            )

    logger.info("Preprocessing complete -> %s", out_root)


if __name__ == "__main__":
    main()
