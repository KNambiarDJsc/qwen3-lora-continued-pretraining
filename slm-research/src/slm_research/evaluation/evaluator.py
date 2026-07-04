"""Validation-loop metric computation — standalone and training-periodic.

Computes the full suite of val/* metrics required by the assignment:
  val/loss          — mean cross-entropy over all positions
  val/token_loss    — alias for val/loss (per-token average)
  val/sequence_loss — mean loss averaged per sequence then across sequences
  val/ppl           — perplexity = exp(val/loss)
  val/bpt           — bits per token = val/loss / ln(2)
  val/n_tokens      — total token positions evaluated
  val/n_sequences   — total sequences evaluated
  val/ppl_by_length_bucket/{start}-{end}
                    — position-range perplexity within each packed sequence.

Length-bucket PPL interprets the bucket thresholds as POSITION RANGES within
packed sequences of length `sequence_length`. This answers: "how does the
model's prediction quality change as it accumulates more context?"  With
sequence_length=2048 and buckets=[128, 256, 512, 1024, 2048]:
  • 0–128:    first 128 positions (short-range dependencies)
  • 128–256:  positions 128–255
  • …
  • 1024–2048: final half of the context window (long-range dependencies)

This is callable both from trainer.py (quick in-loop pass) and from the
standalone scripts/evaluate.py (full checkpoint evaluation).

Qualitative sample generation (val/examples) is a separate, more expensive
step — generate_qualitative_samples() is only called from the standalone
scripts/evaluate.py flow, not from trainer.py's periodic in-loop eval.

Depends on: modeling/losses.py, data/datamodule.py, evaluation/generation.py
Consumed by: training/trainer.py (periodic), scripts/evaluate.py (standalone)
"""
from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from slm_research.evaluation.generation import generate_samples
from slm_research.utils.config_schema import EvaluationConfig, InferenceConfig

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs the held-out validation pass and computes all reporting metrics.

    Args:
        model: PeftModel (or any model with a causal-LM forward signature).
        val_dataloader: DataLoader yielding {input_ids, attention_mask, labels}.
        eval_cfg: Validated EvaluationConfig.
        device: Device to run evaluation on.
    """

    def __init__(
        self,
        model: Any,
        val_dataloader: DataLoader,
        eval_cfg: EvaluationConfig,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.val_dl = val_dataloader
        self.eval_cfg = eval_cfg
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, global_step: int = 0) -> dict[str, float]:
        """Run the full validation pass.

        Args:
            global_step: Current training step — included in logged metrics
                for traceability but not used in the computation.

        Returns:
            Flat dict of val/* metrics (values are Python floats, not tensors).
        """
        self.model.eval()

        total_token_loss = 0.0        # sum of per-token loss × n_tokens
        total_seq_loss = 0.0          # sum of per-sequence mean losses
        total_tokens = 0
        total_sequences = 0

        # Position-range accumulators for length-bucket PPL.
        # Each bucket is (start, end) position range.
        buckets = self._build_buckets()
        bucket_loss_sums: dict[str, float] = {k: 0.0 for k in buckets}
        bucket_token_counts: dict[str, int] = {k: 0 for k in buckets}

        with torch.no_grad():
            for batch in self.val_dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                input_ids = batch["input_ids"]          # (B, T)
                attention_mask = batch["attention_mask"] # (B, T)
                batch_size, seq_len = input_ids.shape

                # Full forward — let the model compute logits
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits  # (B, T, V)

                # Shift: logit at position i predicts token at position i+1
                shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
                shift_labels = input_ids[:, 1:].contiguous()   # (B, T-1)
                T = shift_labels.size(1)  # T-1

                # Per-token loss without reduction — shape (B, T-1)
                per_token_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(batch_size, T)

                # --- Overall metrics ---
                token_count = attention_mask[:, 1:].sum().item()
                masked_loss_sum = (per_token_loss * attention_mask[:, 1:]).sum().item()
                total_token_loss += masked_loss_sum
                total_tokens += int(token_count)

                seq_mean_loss = per_token_loss.mean(dim=-1)  # (B,)
                total_seq_loss += seq_mean_loss.sum().item()
                total_sequences += batch_size

                # --- Length-bucket PPL (position-range perplexity) ---
                for key, (start, end) in buckets.items():
                    end_clamp = min(end, T)
                    if start >= end_clamp:
                        continue
                    bucket_slice = per_token_loss[:, start:end_clamp]     # (B, width)
                    mask_slice = attention_mask[:, 1 + start: 1 + end_clamp]
                    bucket_loss_sums[key] += (bucket_slice * mask_slice).sum().item()
                    bucket_token_counts[key] += mask_slice.sum().item()

        # --- Aggregate ---
        val_loss = total_token_loss / max(total_tokens, 1)
        val_seq_loss = total_seq_loss / max(total_sequences, 1)

        metrics: dict[str, float] = {
            "val/loss": val_loss,
            "val/token_loss": val_loss,
            "val/sequence_loss": val_seq_loss,
            "val/ppl": math.exp(min(val_loss, 20.0)),
            "val/bpt": val_loss / math.log(2),
            "val/n_tokens": float(total_tokens),
            "val/n_sequences": float(total_sequences),
        }

        # Length-bucket PPL
        for key, loss_sum in bucket_loss_sums.items():
            n = bucket_token_counts[key]
            if n > 0:
                bucket_loss = loss_sum / n
                metrics[f"val/ppl_by_length_bucket/{key}"] = math.exp(
                    min(bucket_loss, 20.0)
                )

        logger.info(
            "Evaluation complete — val/loss=%.4f  val/ppl=%.2f  "
            "val/bpt=%.4f  n_seqs=%d  n_tokens=%d",
            metrics["val/loss"],
            metrics["val/ppl"],
            metrics["val/bpt"],
            total_sequences,
            total_tokens,
        )
        return metrics

    def generate_qualitative_samples(
        self,
        tokenizer: PreTrainedTokenizerBase,
        inference_cfg: InferenceConfig,
        prompt_tokens: int = 32,
    ) -> tuple[list[str], list[str]]:
        """Draw prompts from the validation set and generate completions.

        Takes the first `prompt_tokens` positions of the first
        eval_cfg.num_generation_samples sequences in the validation set,
        decodes them to text as conditioning prompts, and generates
        completions for qualitative inspection (val/examples).

        Args:
            tokenizer: Tokenizer matching self.model.
            inference_cfg: Validated InferenceConfig — supplies temperature,
                top_p, and do_sample. max_new_tokens is overridden with
                eval_cfg.generation_max_new_tokens.
            prompt_tokens: Number of leading token positions used as the
                conditioning prompt for each sample.

        Returns:
            (prompts, completions) — parallel lists, one entry per sample.
        """
        num_samples = self.eval_cfg.num_generation_samples
        prompts: list[str] = []
        for batch in self.val_dl:
            for row in batch["input_ids"]:
                if len(prompts) >= num_samples:
                    break
                prompts.append(
                    tokenizer.decode(row[:prompt_tokens], skip_special_tokens=True)
                )
            if len(prompts) >= num_samples:
                break

        gen_cfg = inference_cfg.model_copy(
            update={"max_new_tokens": self.eval_cfg.generation_max_new_tokens}
        )
        completions = generate_samples(
            self.model, tokenizer, prompts, gen_cfg, device=self.device
        )
        return prompts, completions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_buckets(self) -> dict[str, tuple[int, int]]:
        """Convert EvaluationConfig.length_buckets into (start, end) ranges.

        Example: [128, 256, 512, 1024, 2048]
          → {"0-128": (0, 128), "128-256": (128, 256), …, "1024-2048": (1024, 2048)}
        """
        thresholds = self.eval_cfg.length_buckets
        buckets: dict[str, tuple[int, int]] = {}
        prev = 0
        for end in thresholds:
            key = f"{prev}-{end}"
            buckets[key] = (prev, end)
            prev = end
        return buckets
