"""Unit tests for inference/generator.py."""
from __future__ import annotations

import torch
from torch import nn

from slm_research.inference.generator import format_generation, generate
from slm_research.utils.config_schema import InferenceConfig


class FakeBatchEncoding(dict):
    """Minimal stand-in for transformers.BatchEncoding — supports .to(device)."""

    def to(self, device: str | torch.device) -> "FakeBatchEncoding":
        return FakeBatchEncoding({k: v.to(device) for k, v in self.items()})


class FakeTokenizer:
    """Encodes prompts as a length-1 sequence keyed on prompt length; decodes by echoing ids."""

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> FakeBatchEncoding:
        ids = torch.tensor([[len(text)]], dtype=torch.long)
        return FakeBatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return f"completion-{token_ids.tolist()}"


class FakeGenerativeModel(nn.Module):
    """Records the kwargs passed to .generate() and returns prompt + fixed suffix."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.last_generate_kwargs: dict | None = None

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        self.last_generate_kwargs = kwargs
        new_tokens = torch.tensor([[7, 8]], dtype=torch.long)
        return torch.cat([input_ids, new_tokens], dim=1)


def _cfg(do_sample: bool = True, **overrides) -> InferenceConfig:
    base = {"max_new_tokens": 32, "temperature": 0.7, "top_p": 0.9, "do_sample": do_sample}
    base.update(overrides)
    return InferenceConfig(**base)


def test_generate_returns_decoded_completion() -> None:
    model = FakeGenerativeModel()
    completion = generate(model, FakeTokenizer(), "hello", _cfg(), device="cpu")
    assert completion.startswith("completion-")


def test_generate_includes_temperature_top_p_when_sampling() -> None:
    model = FakeGenerativeModel()
    cfg = _cfg(do_sample=True, temperature=0.5, top_p=0.8)

    generate(model, FakeTokenizer(), "hi", cfg, device="cpu")

    assert model.last_generate_kwargs == {
        "max_new_tokens": 32,
        "do_sample": True,
        "pad_token_id": 0,
        "temperature": 0.5,
        "top_p": 0.8,
    }


def test_generate_omits_temperature_top_p_for_greedy_decoding() -> None:
    model = FakeGenerativeModel()
    cfg = _cfg(do_sample=False)

    generate(model, FakeTokenizer(), "hi", cfg, device="cpu")

    assert model.last_generate_kwargs == {
        "max_new_tokens": 32,
        "do_sample": False,
        "pad_token_id": 0,
    }


def test_format_generation_greedy_label() -> None:
    out = format_generation("hi", "there", _cfg(do_sample=False))
    assert "Decoding: greedy" in out
    assert "Prompt:" in out and "hi" in out
    assert "Completion:" in out and "there" in out


def test_format_generation_sampling_label_includes_params() -> None:
    out = format_generation("hi", "there", _cfg(do_sample=True, temperature=0.3, top_p=0.6))
    assert "temperature=0.3" in out
    assert "top_p=0.6" in out
