"""Entrypoint: evaluate.

Thin orchestration only per architecture spec — wires configs to the
appropriate module(s) in src/slm_research/. No business logic lives here.
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    raise NotImplementedError(
        "Phase 2 follow-up: wire this entrypoint once its underlying "
        "module(s) are implemented in the corresponding phase."
    )


if __name__ == "__main__":
    main()
