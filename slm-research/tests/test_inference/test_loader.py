"""Unit tests for inference/loader.py.

Mocks the tokenizer/model-loading calls (they hit the HF Hub) and exercises
only load_model_for_inference's own logic: wiring those calls together and
deciding whether to merge the LoRA adapter based on precision. root_cfg only
needs the two attributes loader.py actually reads (model, training.precision)
so a SimpleNamespace stands in for the full validated RootConfig.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from slm_research.inference import loader as loader_module
from slm_research.inference.loader import load_model_for_inference


class FakeTokenizer:
    pass


class FakeMergedModel:
    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True


class FakePeftModel:
    def __init__(self) -> None:
        self.merge_called = False
        self.eval_called = False
        self.merged = FakeMergedModel()

    def merge_and_unload(self) -> FakeMergedModel:
        self.merge_called = True
        return self.merged

    def eval(self) -> None:
        self.eval_called = True


def _root_cfg(precision: str) -> Any:
    return SimpleNamespace(
        model=SimpleNamespace(name="fake/model"),
        training=SimpleNamespace(precision=precision),
    )


def _patch_loading(monkeypatch, fake_peft_model: FakePeftModel) -> None:
    monkeypatch.setattr(loader_module, "load_tokenizer", lambda model_cfg: FakeTokenizer())
    monkeypatch.setattr(
        loader_module, "load_base_model", lambda model_cfg, training_cfg, cache_dir=None: "base"
    )
    monkeypatch.setattr(
        loader_module, "load_lora_checkpoint", lambda base, path: fake_peft_model
    )


def test_load_model_for_inference_merges_for_non_quantized_precision(monkeypatch) -> None:
    fake_peft_model = FakePeftModel()
    _patch_loading(monkeypatch, fake_peft_model)

    model, tokenizer = load_model_for_inference(_root_cfg("bf16"), checkpoint_path="ckpt")

    assert fake_peft_model.merge_called is True
    assert model is fake_peft_model.merged
    assert model.eval_called is True
    assert isinstance(tokenizer, FakeTokenizer)


def test_load_model_for_inference_skips_merge_for_quantized_precision(monkeypatch) -> None:
    fake_peft_model = FakePeftModel()
    _patch_loading(monkeypatch, fake_peft_model)

    model, _ = load_model_for_inference(_root_cfg("4bit"), checkpoint_path="ckpt")

    assert fake_peft_model.merge_called is False
    assert model is fake_peft_model
    assert model.eval_called is True
