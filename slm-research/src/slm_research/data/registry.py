"""Dataset registry — maps short dataset names to HF identifiers and adapters.

Adding a new dataset requires exactly one change: a new entry in REGISTRY.
Everything else (loading, cleaning, tokenizing, packing) is generic.
"""
from __future__ import annotations

from dataclasses import dataclass

from slm_research.data.adapters.base import DatasetAdapter
from slm_research.data.adapters.wikitext import WikitextAdapter
from slm_research.data.adapters.openwebtext import OpenWebTextAdapter
from slm_research.data.adapters.bookcorpusopen import BookCorpusOpenAdapter
from slm_research.data.adapters.tinystories import TinyStoriesAdapter
from slm_research.data.adapters.ag_news import AgNewsAdapter
from slm_research.data.adapters.xsum import XSumAdapter
from slm_research.data.adapters.cnn_dailymail import CnnDailymailAdapter
from slm_research.data.adapters.daily_dialog import DailyDialogAdapter
from slm_research.data.adapters.eli5 import ELI5Adapter
from slm_research.data.adapters.fineweb_edu import FineWebEduAdapter


@dataclass(frozen=True)
class DatasetInfo:
    """Static metadata for one registered dataset."""

    hf_path: str
    """HuggingFace dataset repository path (first arg to load_dataset)."""

    adapter: DatasetAdapter
    """Adapter that converts raw rows to {"text": str}."""

    default_train_split: str = "train"
    """Split name to use when loading training data."""

    default_val_split: str | None = None
    """Pre-existing validation split, if the dataset provides one.
    When None, a held-out fraction is carved from the train split instead."""


REGISTRY: dict[str, DatasetInfo] = {
    "wikitext": DatasetInfo(
        hf_path="Salesforce/wikitext",
        adapter=WikitextAdapter(),
        default_train_split="train",
        default_val_split="validation",
    ),
    "openwebtext": DatasetInfo(
        hf_path="Skylion007/openwebtext",
        adapter=OpenWebTextAdapter(),
        default_train_split="train",
        default_val_split=None,
    ),
    "bookcorpusopen": DatasetInfo(
        hf_path="lucadiliello/bookcorpusopen",
        adapter=BookCorpusOpenAdapter(),
        default_train_split="train",
        default_val_split=None,
    ),
    "tinystories": DatasetInfo(
        hf_path="roneneldan/TinyStories",
        adapter=TinyStoriesAdapter(),
        default_train_split="train",
        default_val_split="validation",
    ),
    "ag_news": DatasetInfo(
        hf_path="fancyzhx/ag_news",
        adapter=AgNewsAdapter(),
        default_train_split="train",
        default_val_split="test",
    ),
    "cnn_dailymail": DatasetInfo(
        hf_path="abisee/cnn_dailymail",
        adapter=CnnDailymailAdapter(),
        default_train_split="train",
        default_val_split="validation",
    ),
    "xsum": DatasetInfo(
        hf_path="EdinburghNLP/xsum",
        adapter=XSumAdapter(),
        default_train_split="train",
        default_val_split="validation",
    ),
    "daily_dialog": DatasetInfo(
        hf_path="li2017dailydialog/daily_dialog",
        adapter=DailyDialogAdapter(),
        default_train_split="train",
        default_val_split="validation",
    ),
    "eli5": DatasetInfo(
        hf_path="eli5",
        adapter=ELI5Adapter(),
        default_train_split="train_eli5",
        default_val_split="validation_eli5",
    ),
    "fineweb_edu": DatasetInfo(
        hf_path="HuggingFaceFW/fineweb-edu",
        adapter=FineWebEduAdapter(),
        default_train_split="train",
        default_val_split=None,
    ),
}


def get_dataset_info(name: str) -> DatasetInfo:
    """Return the DatasetInfo for a registered dataset name.

    Args:
        name: Short dataset name as used in mixture.yaml (e.g. "wikitext").

    Returns:
        DatasetInfo with hf_path, adapter, and split names.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in REGISTRY:
        raise KeyError(
            f"Dataset {name!r} not in registry. "
            f"Registered names: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]
