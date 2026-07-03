"""slm_research.modeling — model loading, precision, LoRA, and loss utilities."""
from slm_research.modeling.model_factory import build_model, load_base_model, load_model_from_checkpoint
from slm_research.modeling.lora_factory import apply_lora, load_lora_checkpoint
from slm_research.modeling.losses import causal_lm_loss, perplexity, bits_per_token

__all__ = [
    "build_model",
    "load_base_model",
    "load_model_from_checkpoint",
    "apply_lora",
    "load_lora_checkpoint",
    "causal_lm_loss",
    "perplexity",
    "bits_per_token",
]
