"""Instantiate the base Qwen3-0.6B-Base model.

Responsibility: load base weights, apply device mapping. Does NOT apply
precision settings (precision.py) or LoRA adapters (lora_factory.py) —
those wrap the object this module returns.

Depends on: configs/model/qwen3_0.6b_base.yaml
Consumed by: precision.py -> lora_factory.py -> training/trainer.py
"""

from typing import Any


def load_base_model(model_config: dict[str, Any]) -> Any:
    """Load the base causal LM from Hugging Face per model_config.

    Raises:
        NotImplementedError: implementation lands in Phase 5 (Model Pipeline).
    """
    raise NotImplementedError("Phase 5: implement AutoModelForCausalLM.from_pretrained call.")
