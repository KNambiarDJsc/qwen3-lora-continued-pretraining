"""Unit tests for scripts/preprocess.py.

Like test_download_data.py, uses hydra.compose() to build a real merged
config from the actual configs/ tree and calls the @hydra.main-decorated
main() via cfg_passthrough, bypassing sys.argv parsing. load_source_splits
and load_tokenizer are mocked (no network access), but preprocess_dataset /
tokenize_dataset / pack run for real against small in-memory fakes, and the
save_to_disk output is loaded back and checked.

sequence_length is overridden down to 4 (from mixture.yaml's real 2048) so
the tiny fake texts here actually produce at least one packed block.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from datasets import Dataset, IterableDataset, load_from_disk
from hydra import compose, initialize_config_dir

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = str(_REPO_ROOT / "configs")

_spec = importlib.util.spec_from_file_location("preprocess", _REPO_ROOT / "scripts" / "preprocess.py")
assert _spec is not None and _spec.loader is not None
preprocess = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preprocess)


class FakeTokenizer:
    eos_token_id = 999

    def __call__(self, texts, add_special_tokens=False, truncation=False, padding=False):
        return {"input_ids": [[len(w) for w in t.split()] for t in texts]}


def _fake_split(name: str, n: int, streaming: bool):
    texts = [f"{name} sample number {i} with quite a few words in it" for i in range(n)]
    if streaming:
        def gen():
            for t in texts:
                yield {"text": t}

        return IterableDataset.from_generator(gen)
    return Dataset.from_dict({"text": texts})


def _fake_load_source_splits(source, val_split_fraction, seed, cache_dir=None):
    return _fake_split(source.name, 6, source.streaming), _fake_split(source.name, 2, source.streaming)


def test_main_preprocesses_and_saves_every_source(tmp_path, monkeypatch) -> None:
    import slm_research.data.loaders as loaders_module
    import slm_research.data.tokenization as tokenization_module

    monkeypatch.setattr(loaders_module, "load_source_splits", _fake_load_source_splits)
    monkeypatch.setattr(tokenization_module, "load_tokenizer", lambda model_cfg: FakeTokenizer())

    with initialize_config_dir(config_dir=os.path.abspath(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"run.output_dir={tmp_path}", "data.sequence_length=4"],
        )

    preprocess.main(cfg_passthrough=cfg)

    out_root = tmp_path / "preprocessed"
    source_names = [s.name for s in cfg.data.sources]
    assert len(source_names) == 10

    for name in source_names:
        for split in ("train", "val"):
            split_dir = out_root / name / split
            assert split_dir.exists(), f"missing {split_dir}"
            ds = load_from_disk(str(split_dir))
            assert set(ds.column_names) == {"input_ids", "attention_mask"}
            assert all(len(row) == 4 for row in ds["input_ids"])
            assert all(row == [1, 1, 1, 1] for row in ds["attention_mask"])
