"""Structured logging setup. No print() anywhere else in this repository.

Depends on: nothing
Consumed by: every scripts/*.py entrypoint
"""
from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured: bool = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a structured logger for the given module name.

    Idempotent: attaches one stream handler to the root logger the first
    time this is called in a process, then reuses it — safe to call from
    every module without producing duplicate log lines.

    Every scripts/*.py entrypoint is a @hydra.main app, and Hydra's own
    job_logging config already attaches a root handler before the task
    function runs — this only adds a handler when none exists yet, so it
    defers to Hydra's setup there and only takes effect for plain scripts,
    tests, or a REPL.

    Args:
        name: Usually __name__ of the calling module.
        level: Log level for the root logger, applied only on first setup.

    Returns:
        A logging.Logger for `name`.
    """
    global _configured
    root = logging.getLogger()
    if not _configured and not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
        root.setLevel(level)
        _configured = True
    return logging.getLogger(name)
