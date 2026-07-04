"""Unit tests for evaluation/generation.py and Evaluator.generate_qualitative_samples."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from slm_research.evaluation import evaluator as evaluator_module
from slm_research.evaluation.evaluator import Evaluator
from slm_research.evaluation.generation import generate_samples
from slm_research.utils.config_schema import EvaluationConfig, InferenceConfig


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
        return FakeBatchEncoding(
            {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        )

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
        new_tokens = torch.tensor([[99, 99]], dtype=torch.long)
        return torch.cat([input_ids, new_tokens], dim=1)


def _inference_cfg(max_new_tokens: int = 128) -> InferenceConfig:
    return InferenceConfig(
        max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True
    )


def test_generate_samples_returns_one_completion_per_prompt() -> None:
    model = FakeGenerativeModel()
    tokenizer = FakeTokenizer()
    prompts = ["hello", "world!"]

    completions = generate_samples(model, tokenizer, prompts, _inference_cfg(), device="cpu")

    assert len(completions) == len(prompts)
    assert all(c.startswith("completion-") for c in completions)


def test_generate_samples_forwards_inference_config_to_generate() -> None:
    model = FakeGenerativeModel()
    tokenizer = FakeTokenizer()
    cfg = _inference_cfg(max_new_tokens=42)
    cfg = cfg.model_copy(update={"temperature": 0.3, "top_p": 0.5, "do_sample": False})

    generate_samples(model, tokenizer, ["prompt"], cfg, device="cpu")

    assert model.last_generate_kwargs == {
        "max_new_tokens": 42,
        "temperature": 0.3,
        "top_p": 0.5,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }


def test_generate_samples_restores_training_mode() -> None:
    model = FakeGenerativeModel()
    model.train()
    tokenizer = FakeTokenizer()

    generate_samples(model, tokenizer, ["prompt"], _inference_cfg(), device="cpu")

    assert model.training is True


class FakeGenTokenizer:
    """Tokenizer stub for Evaluator.generate_qualitative_samples: decode returns joined ids."""

    pad_token_id = 0
    eos_token_id = 0

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "prompt:" + ",".join(str(i) for i in token_ids.tolist())


def test_generate_qualitative_samples_draws_prompts_from_val_set(monkeypatch) -> None:
    captured: dict = {}

    def fake_generate_samples(model, tokenizer, prompts, inference_cfg, device="cuda"):
        captured["prompts"] = prompts
        captured["inference_cfg"] = inference_cfg
        return [f"gen-{p}" for p in prompts]

    monkeypatch.setattr(evaluator_module, "generate_samples", fake_generate_samples)

    batch = {
        "input_ids": torch.arange(3 * 6).reshape(3, 6),
        "attention_mask": torch.ones(3, 6, dtype=torch.long),
    }
    eval_cfg = EvaluationConfig(
        val_split_fraction=0.1,
        length_buckets=[6],
        num_generation_samples=2,
        generation_max_new_tokens=7,
    )
    evaluator = Evaluator(
        model=nn.Module(), val_dataloader=[batch], eval_cfg=eval_cfg, device="cpu"
    )
    inference_cfg = InferenceConfig(
        max_new_tokens=128, temperature=0.7, top_p=0.9, do_sample=True
    )

    prompts, completions = evaluator.generate_qualitative_samples(
        tokenizer=FakeGenTokenizer(), inference_cfg=inference_cfg, prompt_tokens=3
    )

    # num_generation_samples=2 caps the draw even though the batch has 3 rows.
    assert prompts == ["prompt:0,1,2", "prompt:6,7,8"]
    assert completions == [f"gen-{p}" for p in prompts]
    # generation_max_new_tokens overrides inference_cfg.max_new_tokens.
    assert captured["inference_cfg"].max_new_tokens == 7
    assert captured["inference_cfg"].temperature == 0.7
