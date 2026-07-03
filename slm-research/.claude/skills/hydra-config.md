# Hydra Configuration

Always use Hydra correctly.

Rules:

- No hardcoded hyperparameters.
- Every configurable value belongs in configs/.
- Use OmegaConf.
- Resolve interpolations before validation.
- Validate merged configs with Pydantic.
- Fail fast before GPU initialization.
