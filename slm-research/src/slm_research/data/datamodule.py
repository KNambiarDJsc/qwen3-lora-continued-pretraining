"""DataModule — orchestrates the full dataset pipeline for one training run.

Responsibility: wire together loaders → preprocessing → tokenization →
packing → mixture → DataLoader. This is the single entry point that
training/trainer.py and evaluation/evaluator.py call to get DataLoaders.

Pipeline per source:
    load_source  →  preprocess  →  tokenize  →  pack
                                                  ↓
                              mixture (interleave across sources)
                                                  ↓
                                             DataLoader

The train/val split happens at load time (before packing) so each source's
val examples are representative of that source's content distribution.

Depends on: loaders, preprocessing, tokenization, packing, mixture, collators
Consumed by: scripts/train.py, scripts/evaluate.py
"""
from __future__ import annotations

import logging
from typing import Union

from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from slm_research.data.collators import CausalLMCollator
from slm_research.data.loaders import load_source_splits
from slm_research.data.mixture import build_mixture
from slm_research.data.packing import pack
from slm_research.data.preprocessing import preprocess_dataset
from slm_research.data.tokenization import load_tokenizer, tokenize_dataset
from slm_research.utils.config_schema import (
    EvaluationConfig,
    MixtureConfig,
    ModelConfig,
    TrainingConfig,
)

logger = logging.getLogger(__name__)

HFDataset = Union[Dataset, IterableDataset]


class DataModule:
    """Builds train and validation DataLoaders for the full dataset mixture.

    Args:
        mixture_cfg: Validated MixtureConfig (sources, weights, strategy).
        model_cfg: Validated ModelConfig (tokenizer name, trust_remote_code).
        training_cfg: Validated TrainingConfig (batch size, workers).
        eval_cfg: Validated EvaluationConfig (val_split_fraction).
        seed: Global RNG seed from RunConfig.
        cache_dir: Optional local directory for HF dataset caching.
        deduplicate: If True, run exact dedup on non-streaming sources.
        num_proc: Parallel workers for non-streaming map operations.
    """

    def __init__(
        self,
        mixture_cfg: MixtureConfig,
        model_cfg: ModelConfig,
        training_cfg: TrainingConfig,
        eval_cfg: EvaluationConfig,
        seed: int,
        cache_dir: str | None = None,
        deduplicate: bool = False,
        num_proc: int | None = None,
    ) -> None:
        self.mixture_cfg = mixture_cfg
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        self.eval_cfg = eval_cfg
        self.seed = seed
        self.cache_dir = cache_dir
        self.deduplicate = deduplicate
        self.num_proc = num_proc

        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._train_dataset: HFDataset | None = None
        self._val_dataset: HFDataset | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Lazily load and cache the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.model_cfg)
        return self._tokenizer

    def setup(self) -> None:
        """Build train and val datasets. Call once before get_*_dataloader."""
        logger.info("Setting up dataset pipeline for %d sources.", len(self.mixture_cfg.sources))

        train_packed: list[HFDataset] = []
        val_packed: list[HFDataset] = []

        for source in self.mixture_cfg.sources:
            logger.info("Processing source: %s", source.name)

            train_raw, val_raw = load_source_splits(
                source=source,
                val_split_fraction=self.eval_cfg.val_split_fraction,
                seed=self.seed,
                cache_dir=self.cache_dir,
            )

            train_clean = preprocess_dataset(train_raw, num_proc=self.num_proc)
            val_clean = preprocess_dataset(val_raw, num_proc=self.num_proc)

            if self.deduplicate and isinstance(train_clean, Dataset):
                from slm_research.data.preprocessing import deduplicate_dataset
                train_clean = deduplicate_dataset(train_clean)

            train_tok = tokenize_dataset(
                train_clean,
                self.tokenizer,
                add_eos_token=self.model_cfg.tokenizer.add_eos_token,
                num_proc=self.num_proc,
            )
            val_tok = tokenize_dataset(
                val_clean,
                self.tokenizer,
                add_eos_token=self.model_cfg.tokenizer.add_eos_token,
                num_proc=self.num_proc,
            )

            seq_len = self.mixture_cfg.sequence_length
            train_packed.append(pack(train_tok, seq_len, num_proc=self.num_proc))
            val_packed.append(pack(val_tok, seq_len, num_proc=self.num_proc))

        self._train_dataset = build_mixture(train_packed, self.mixture_cfg, seed=self.seed)
        self._val_dataset = build_mixture(val_packed, self.mixture_cfg, seed=self.seed)

        logger.info("Dataset pipeline ready.")

    def get_train_dataloader(self) -> DataLoader:
        """Return the training DataLoader.

        Raises:
            RuntimeError: If setup() has not been called.
        """
        if self._train_dataset is None:
            raise RuntimeError("Call DataModule.setup() before get_train_dataloader().")

        is_iterable = isinstance(self._train_dataset, IterableDataset)
        return DataLoader(
            self._train_dataset,
            batch_size=self.training_cfg.per_device_train_batch_size,
            shuffle=(not is_iterable),
            num_workers=self.training_cfg.dataloader_num_workers,
            collate_fn=CausalLMCollator(),
            pin_memory=True,
            drop_last=True,
        )

    def get_val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader.

        Raises:
            RuntimeError: If setup() has not been called.
        """
        if self._val_dataset is None:
            raise RuntimeError("Call DataModule.setup() before get_val_dataloader().")

        return DataLoader(
            self._val_dataset,
            batch_size=self.training_cfg.per_device_train_batch_size,
            shuffle=False,
            num_workers=self.training_cfg.dataloader_num_workers,
            collate_fn=CausalLMCollator(),
            pin_memory=True,
            drop_last=False,
        )


def build_data_module(
    mixture_cfg: MixtureConfig,
    model_cfg: ModelConfig,
    training_cfg: TrainingConfig,
    eval_cfg: EvaluationConfig,
    seed: int,
    cache_dir: str | None = None,
    deduplicate: bool = False,
    num_proc: int | None = None,
) -> DataModule:
    """Convenience factory — construct and set up a DataModule in one call.

    Args:
        mixture_cfg: MixtureConfig from RootConfig.
        model_cfg: ModelConfig from RootConfig.
        training_cfg: TrainingConfig from RootConfig.
        eval_cfg: EvaluationConfig from RootConfig.
        seed: Global seed from RunConfig.
        cache_dir: Optional HF cache directory.
        deduplicate: Run exact dedup on non-streaming sources.
        num_proc: Parallel workers for non-streaming map operations.

    Returns:
        DataModule with setup() already called.
    """
    dm = DataModule(
        mixture_cfg=mixture_cfg,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        eval_cfg=eval_cfg,
        seed=seed,
        cache_dir=cache_dir,
        deduplicate=deduplicate,
        num_proc=num_proc,
    )
    dm.setup()
    return dm
