"""Structured logging setup. No print() anywhere else in this repository.

Depends on: nothing
Consumed by: every scripts/*.py entrypoint
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured structured logger for the given module name.

    Raises:
        NotImplementedError: implementation lands in Phase 2 follow-up /
            Phase 3 (this is intentionally one of the first real modules
            you should implement yourself — it's low-risk and used everywhere).
    """
    raise NotImplementedError("Implement structured logging (rich/standard logging) here first.")
