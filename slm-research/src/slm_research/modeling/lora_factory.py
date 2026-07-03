"""Wrap a precision-configured model with PEFT LoRA adapters.

Responsibility: apply the LoRA config (configs/lora/rank*.yaml) to the
precision-configured model. This is the last step before the model is
handed to training/trainer.py.

Depends on: precision.py output, configs/lora/*.yaml
Consumed by: training/trainer.py, evaluation/evaluator.py, benchmarking/*
"""

from typing import Any


def apply_lora(model: Any, lora_config: dict[str, Any]) -> Any:
    """Wrap the model with PEFT LoRA adapters per lora_config.

    Raises:
        NotImplementedError: implementation lands in Phase 5.
    """
    raise NotImplementedError("Phase 5: implement PEFT LoRA application.")


def load_lora_checkpoint(base_model: Any, checkpoint_path: str) -> Any:
    """Load a previously trained LoRA adapter onto a base model.

    Used by evaluation/evaluator.py and benchmarking/* to load a specific
    checkpoint without going through the training loop.
    """
    raise NotImplementedError("Phase 5: implement LoRA checkpoint loading.")
