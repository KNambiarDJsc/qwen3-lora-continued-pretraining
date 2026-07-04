"""Unit tests for scripts/cli.py — Typer command routing only.

subprocess.run is monkeypatched so these tests never actually spawn a Hydra
app; they assert on *what* would be run (script path + forwarded args) and
that the subprocess's exit code propagates back out through Typer.

scripts/cli.py is loaded by file path (it's a standalone script, not part of
the installed slm_research package) so these tests don't depend on sys.path
or an __init__.py existing under scripts/.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

_CLI_PATH = Path(__file__).resolve().parents[2] / "scripts" / "cli.py"
_spec = importlib.util.spec_from_file_location("cli", _CLI_PATH)
assert _spec is not None and _spec.loader is not None
cli_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli_module)

runner = CliRunner()


class FakeCompletedProcess:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def test_train_delegates_to_train_py(monkeypatch) -> None:
    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return FakeCompletedProcess(0)

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)

    result = runner.invoke(cli_module.app, ["train", "lora=rank32"])

    assert result.exit_code == 0
    assert captured["argv"][0] == sys.executable
    assert captured["argv"][1] == str(cli_module._SCRIPTS_DIR / "train.py")
    assert captured["argv"][2:] == ["lora=rank32"]


def test_infer_forwards_checkpoint_and_prompt(monkeypatch) -> None:
    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return FakeCompletedProcess(0)

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)

    runner.invoke(cli_module.app, ["infer", "checkpoint=ckpt", "prompt=hi there"])

    assert captured["argv"][1] == str(cli_module._SCRIPTS_DIR / "infer.py")
    assert captured["argv"][2:] == ["checkpoint=ckpt", "prompt=hi there"]


def test_nonzero_subprocess_exit_code_propagates(monkeypatch) -> None:
    monkeypatch.setattr(cli_module.subprocess, "run", lambda argv, **kw: FakeCompletedProcess(3))

    result = runner.invoke(cli_module.app, ["evaluate"])

    assert result.exit_code == 3


@pytest.mark.parametrize(
    "command,script",
    [
        ("train", "train"),
        ("evaluate", "evaluate"),
        ("benchmark", "benchmark"),
        ("infer", "infer"),
        ("download-data", "download_data"),
        ("preprocess", "preprocess"),
        ("resume", "train"),
    ],
)
def test_each_command_delegates_to_expected_script(monkeypatch, command, script) -> None:
    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return FakeCompletedProcess(0)

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)

    runner.invoke(cli_module.app, [command])

    assert captured["argv"][1] == str(cli_module._SCRIPTS_DIR / f"{script}.py")


def test_list_checkpoints_reports_empty_directory(tmp_path) -> None:
    result = runner.invoke(cli_module.app, ["list-checkpoints", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "No checkpoints found" in result.output


def test_list_checkpoints_lists_saved_checkpoints(tmp_path) -> None:
    import torch

    ckpt_dir = tmp_path / "checkpoints" / "abc123_epoch0_step10"
    ckpt_dir.mkdir(parents=True)
    torch.save(
        {"run_id": "abc123", "epoch": 0, "global_step": 10},
        ckpt_dir / "training_state.pt",
    )

    result = runner.invoke(cli_module.app, ["list-checkpoints", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "abc123_epoch0_step10" in result.output
    assert "run_id=abc123" in result.output
    assert "epoch=0" in result.output
    assert "step=10" in result.output


def test_export_pulls_run_history_and_writes_json(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    def fake_export_run_history(project, entity=None, run_id=None):
        captured["project"] = project
        captured["entity"] = entity
        captured["run_id"] = run_id
        return [{"run_id": "abc123", "name": "run-abc", "state": "finished", "config": {}, "summary": {}, "history": []}]

    # export() imports export_run_history from slm_research.tracking.wandb_logger
    # inside the command body — patch it at the source module.
    import slm_research.tracking.wandb_logger as wandb_logger_module

    monkeypatch.setattr(wandb_logger_module, "export_run_history", fake_export_run_history)

    out_path = tmp_path / "export.json"
    result = runner.invoke(
        cli_module.app,
        ["export", "--project", "my-proj", "--run-id", "abc123", "--output", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert captured == {"project": "my-proj", "entity": None, "run_id": "abc123"}
    assert out_path.exists()
    assert "abc123" in out_path.read_text()
    assert "Exported 1 run(s)" in result.output
