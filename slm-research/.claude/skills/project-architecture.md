# Project Architecture

This repository follows a frozen architecture specification.

Rules:

- Never redesign the repository structure.
- Never move files unless explicitly instructed.
- Respect module boundaries.
- Every module has one responsibility.
- No business logic inside scripts/.
- Configuration is entirely Hydra driven.
- Hyperparameters must never be hardcoded.
- Configuration must flow through Hydra → OmegaConf → Pydantic.
- Follow the dependency graph exactly.
- Keep evaluation and benchmarking independent from training.
