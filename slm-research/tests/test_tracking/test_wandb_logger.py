"""Unit tests for tracking/wandb_logger.py::export_run_history.

wandb.Api is monkeypatched — there's no live W&B account/network access in
this test environment, so these verify export_run_history's own wiring
(path construction, single-run vs. whole-project, shape of the returned
dicts) against a fake Api/Run pair built to match the real wandb public API
surface (run.id/.name/.state/.config/.summary/.history()).
"""
from __future__ import annotations

from slm_research.tracking import wandb_logger


class FakeRun:
    def __init__(self, run_id: str, name: str) -> None:
        self.id = run_id
        self.name = name
        self.state = "finished"
        self.config = {"lora": {"r": 16}}
        self.summary = {"val/loss": 1.23}

    def history(self, pandas: bool = True):
        return [{"_step": 0, "train/loss": 2.0}, {"_step": 1, "train/loss": 1.5}]


class FakeApi:
    def __init__(self, runs_by_path: dict[str, list[FakeRun]], run_by_full_path: dict[str, FakeRun]) -> None:
        self._runs_by_path = runs_by_path
        self._run_by_full_path = run_by_full_path
        self.requested_paths: list[str] = []
        self.requested_run_paths: list[str] = []

    def runs(self, path: str):
        self.requested_paths.append(path)
        return self._runs_by_path[path]

    def run(self, path: str):
        self.requested_run_paths.append(path)
        return self._run_by_full_path[path]


def test_export_run_history_without_run_id_exports_whole_project(monkeypatch) -> None:
    fake_runs = [FakeRun("r1", "run-one"), FakeRun("r2", "run-two")]
    fake_api = FakeApi(runs_by_path={"my-project": fake_runs}, run_by_full_path={})
    monkeypatch.setattr(wandb_logger.wandb, "Api", lambda: fake_api)

    exported = wandb_logger.export_run_history(project="my-project")

    assert fake_api.requested_paths == ["my-project"]
    assert [r["run_id"] for r in exported] == ["r1", "r2"]
    assert exported[0]["config"] == {"lora": {"r": 16}}
    assert exported[0]["summary"] == {"val/loss": 1.23}
    assert exported[0]["history"] == [{"_step": 0, "train/loss": 2.0}, {"_step": 1, "train/loss": 1.5}]


def test_export_run_history_with_entity_builds_entity_slash_project_path(monkeypatch) -> None:
    fake_api = FakeApi(runs_by_path={"me/my-project": []}, run_by_full_path={})
    monkeypatch.setattr(wandb_logger.wandb, "Api", lambda: fake_api)

    wandb_logger.export_run_history(project="my-project", entity="me")

    assert fake_api.requested_paths == ["me/my-project"]


def test_export_run_history_with_run_id_fetches_single_run(monkeypatch) -> None:
    single_run = FakeRun("abc123", "run-abc")
    fake_api = FakeApi(runs_by_path={}, run_by_full_path={"my-project/abc123": single_run})
    monkeypatch.setattr(wandb_logger.wandb, "Api", lambda: fake_api)

    exported = wandb_logger.export_run_history(project="my-project", run_id="abc123")

    assert fake_api.requested_run_paths == ["my-project/abc123"]
    assert len(exported) == 1
    assert exported[0]["run_id"] == "abc123"
    assert exported[0]["state"] == "finished"
