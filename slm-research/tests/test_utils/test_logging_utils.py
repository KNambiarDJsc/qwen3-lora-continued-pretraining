"""Unit tests for utils/logging_utils.py.

logging.getLogger() (root) is monkeypatched to return a throwaway Logger
rather than asserting against the real root logger — pytest's own logging
plugin attaches its own LogCaptureHandler to the real root around every test
phase, which would make handler-count assertions flaky/order-dependent for
reasons unrelated to get_logger's own behavior.
"""
from __future__ import annotations

import logging

import pytest

from slm_research.utils import logging_utils


@pytest.fixture(autouse=True)
def _reset_configured_flag():
    original = logging_utils._configured
    logging_utils._configured = False
    yield
    logging_utils._configured = original


def _patch_root(monkeypatch) -> logging.Logger:
    """Redirect logging.getLogger() (no args) to a fresh, isolated Logger."""
    fake_root = logging.Logger(f"fake-root-{id(monkeypatch)}")
    real_get_logger = logging.getLogger

    def fake_get_logger(name: str | None = None) -> logging.Logger:
        return fake_root if name is None else real_get_logger(name)

    monkeypatch.setattr(logging_utils.logging, "getLogger", fake_get_logger)
    return fake_root


def test_get_logger_returns_logger_for_given_name(monkeypatch) -> None:
    _patch_root(monkeypatch)

    logger = logging_utils.get_logger("my.module")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "my.module"


def test_get_logger_attaches_a_handler_when_root_has_none(monkeypatch) -> None:
    fake_root = _patch_root(monkeypatch)

    logging_utils.get_logger(__name__)

    assert len(fake_root.handlers) == 1
    assert isinstance(fake_root.handlers[0], logging.StreamHandler)


def test_get_logger_does_not_attach_a_second_handler_on_repeat_calls(monkeypatch) -> None:
    fake_root = _patch_root(monkeypatch)

    logging_utils.get_logger("first")
    logging_utils.get_logger("second")

    assert len(fake_root.handlers) == 1


def test_get_logger_defers_to_an_existing_root_handler(monkeypatch) -> None:
    """Simulates Hydra's job_logging having already configured the root
    logger before get_logger() is called — it must not add a second one."""
    fake_root = _patch_root(monkeypatch)
    existing = logging.StreamHandler()
    fake_root.addHandler(existing)

    logging_utils.get_logger(__name__)

    assert fake_root.handlers == [existing]
    assert logging_utils._configured is False


def test_get_logger_sets_configured_flag_on_first_setup(monkeypatch) -> None:
    _patch_root(monkeypatch)

    assert logging_utils._configured is False
    logging_utils.get_logger(__name__)
    assert logging_utils._configured is True
