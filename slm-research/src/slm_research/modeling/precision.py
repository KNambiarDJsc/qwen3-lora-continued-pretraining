"""Apply precision settings (bf16 / fp16 / 4-bit / 8-bit) to a loaded model.

Responsibility: wrap a model_factory.py output with the precision config
specified in configs/training/*.yaml (`precision` field).

Depends on: model_factory.py output
Consumed by: lora_factory.py
"""

from typing import Any


def apply_precision(model: Any, precision_mode: str) -> Any:
    """Apply the requested precision/quantization mode to the model.

    Args:
        model: Output of model_factory.load_base_model.
        precision_mode: One of "bf16", "fp16", "4bit", "8bit".

    Raises:
        NotImplementedError: implementation lands in Phase 5.
    """
    raise NotImplementedError("Phase 5: implement precision/quantization application.")
