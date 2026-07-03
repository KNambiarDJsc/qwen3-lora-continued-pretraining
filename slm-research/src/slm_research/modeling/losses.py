"""Loss functions for causal language model training and evaluation.

HuggingFace AutoModelForCausalLM computes cross-entropy loss internally
when labels are passed to the forward call. These standalone functions exist
for: (1) custom training loops that need the loss separate from the model
forward pass, (2) evaluation code that re-computes loss from logits without
a second forward pass, (3) future extensions (e.g. length-weighted loss).

Depends on: torch
Consumed by: training/trainer.py, evaluation/evaluator.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy next-token prediction loss.

    Shifts logits and labels by one position so that position i predicts
    token i+1. This is the standard causal LM objective — identical to what
    AutoModelForCausalLM computes internally when labels are passed.

    Args:
        logits: Model output, shape (batch, seq_len, vocab_size).
        labels: Target token ids, shape (batch, seq_len).
            Positions with ignore_index are excluded from the loss.
        ignore_index: Token id to skip in the loss (default -100, PyTorch convention).

    Returns:
        Scalar mean cross-entropy loss over all non-ignored positions.
    """
    # Shift: logits[..., :-1] predicts labels[..., 1:]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def bits_per_token(loss: torch.Tensor) -> torch.Tensor:
    """Convert nats (natural-log cross-entropy) to bits per token.

    Args:
        loss: Scalar cross-entropy loss in nats.

    Returns:
        Scalar bits per token (loss / ln(2)).
    """
    return loss / torch.log(torch.tensor(2.0, device=loss.device, dtype=loss.dtype))


def perplexity(loss: torch.Tensor) -> torch.Tensor:
    """Exponentiate a mean cross-entropy loss to obtain perplexity.

    Args:
        loss: Scalar cross-entropy loss (nats).

    Returns:
        Scalar perplexity: exp(loss).
    """
    return torch.exp(loss)
