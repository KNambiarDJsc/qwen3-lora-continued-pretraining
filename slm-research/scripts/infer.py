"""Entrypoint: infer.

Thin orchestration only per architecture spec — wires configs to the
appropriate module(s) in src/slm_research/. No business logic lives here.

Checkpoint-driven, single-prompt inference:
    checkpoint -> load tokenizer -> load base model -> load LoRA adapter ->
    merge adapter -> generate -> pretty output.

Run:
    python scripts/infer.py \
        checkpoint=outputs/checkpoints/run_abc_epoch0_step500 \
        prompt="The future of AI is"

Decoding is controlled by configs/inference/default.yaml (max_new_tokens,
temperature, top_p, do_sample) — override any of them at the CLI, e.g.
`inference.do_sample=false` for greedy decoding.
"""
from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint_path = cfg.get("checkpoint", None)
    prompt = cfg.get("prompt", None)
    if checkpoint_path is None:
        raise ValueError(
            "scripts/infer.py requires checkpoint=<dir>, pointing to a "
            "checkpoint directory saved by training/checkpointing.py."
        )
    if not prompt:
        raise ValueError('scripts/infer.py requires prompt="<text>".')

    from slm_research.utils.config_schema import validate_config

    plain = OmegaConf.to_container(cfg, resolve=True)
    plain.pop("checkpoint", None)
    plain.pop("prompt", None)
    root_cfg = validate_config(plain)

    from slm_research.training.seed import set_seed

    set_seed(root_cfg.run.seed)

    import torch

    from slm_research.inference.generator import format_generation, generate
    from slm_research.inference.loader import load_model_for_inference

    logger.info("Loading model from checkpoint: %s", checkpoint_path)
    model, tokenizer = load_model_for_inference(root_cfg, checkpoint_path=checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    completion = generate(model, tokenizer, prompt, root_cfg.inference, device=device)
    print(format_generation(prompt, completion, root_cfg.inference))


if __name__ == "__main__":
    main()
