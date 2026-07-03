"""Entrypoint: train.

Thin orchestration only per architecture spec — wires configs to the
appropriate module(s) in src/slm_research/. No business logic lives here.
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from slm_research.utils.config_schema import validate_config


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    plain = OmegaConf.to_container(cfg, resolve=True)
    validate_config(plain)
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
