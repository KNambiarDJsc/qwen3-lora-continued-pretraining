"""Unit tests for scripts/download_data.py.

@hydra.main's decorated function accepts a cfg_passthrough DictConfig,
bypassing CLI/sys.argv parsing entirely — hydra.compose() builds the exact
same merged config a real invocation would, from the real configs/ tree, so
these tests exercise real config validation and wiring. The only thing
mocked out is load_source_splits, since real HF datasets require network
access this test suite shouldn't depend on.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from datasets import Dataset
from hydra import compose, initialize_config_dir

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = str(_REPO_ROOT / "configs")

_spec = importlib.util.spec_from_file_location("download_data", _REPO_ROOT / "scripts" / "download_data.py")
assert _spec is not None and _spec.loader is not None
download_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download_data)


def _fake_load_source_splits(source, val_split_fraction, seed, cache_dir=None):
    train_ds = Dataset.from_dict({"text": [f"{source.name}-train-{i}" for i in range(3)]})
    val_ds = Dataset.from_dict({"text": [f"{source.name}-val-{i}" for i in range(1)]})
    return train_ds, val_ds


def test_main_downloads_every_configured_source(monkeypatch) -> None:
    calls: list[str] = []

    def tracking_load_source_splits(source, val_split_fraction, seed, cache_dir=None):
        calls.append(source.name)
        return _fake_load_source_splits(source, val_split_fraction, seed, cache_dir)

    # download_data.py imports load_source_splits inside main() via
    # `from slm_research.data.loaders import load_source_splits` — that
    # import statement resolves the attribute at call time, so patching the
    # source module's attribute before calling main() is enough to intercept it.
    import slm_research.data.loaders as loaders_module

    monkeypatch.setattr(loaders_module, "load_source_splits", tracking_load_source_splits)

    with initialize_config_dir(config_dir=os.path.abspath(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config")

    download_data.main(cfg_passthrough=cfg)

    assert len(calls) == 8
    assert "wikitext" in calls
    assert "fineweb_edu" in calls
