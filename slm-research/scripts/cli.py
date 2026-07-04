"""Typer CLI entrypoint.

Commands (see architecture spec Section 12): train, evaluate, benchmark,
infer, download-data, preprocess, resume, list-checkpoints, export.
Orchestration only — delegates to scripts/*.py and src/slm_research/*.

Most commands are thin wrappers that run a sibling scripts/*.py file as a
subprocess, forwarding whatever followed the subcommand as Hydra overrides:

    python scripts/cli.py train lora=rank32 training=runpod_4090
    python scripts/cli.py infer checkpoint=outputs/checkpoints/run_abc prompt="hi"

Subprocess (rather than importing the script and calling its main() in this
process) is deliberate: each scripts/*.py entrypoint is a @hydra.main app that
resolves its config directory relative to its own file and expects to run as
__main__. Reproducing that in-process means fighting Hydra's module/frame
introspection (and its GlobalHydra singleton); running the exact same command
Hydra already supports, in a fresh process, has neither problem and is a more
literal form of "delegation" besides.

`resume` is a discoverability alias for `train` — resuming is really just
training with `+resume_from=<checkpoint_dir>` set (see scripts/train.py).

`list-checkpoints` and `export` are attributed directly to scripts/cli.py by
the architecture spec (no dedicated scripts/*.py file for either).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="SLM research framework CLI.")

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


def _delegate(script_name: str, ctx: typer.Context) -> None:
    """Run scripts/{script_name}.py as a subprocess, forwarding extra args as Hydra overrides."""
    script_path = _SCRIPTS_DIR / f"{script_name}.py"
    result = subprocess.run([sys.executable, str(script_path), *ctx.args])
    raise typer.Exit(code=result.returncode)


@app.command(context_settings=_PASSTHROUGH)
def train(ctx: typer.Context) -> None:
    """Run training. See scripts/train.py."""
    _delegate("train", ctx)


@app.command(context_settings=_PASSTHROUGH)
def evaluate(ctx: typer.Context) -> None:
    """Run standalone evaluation. See scripts/evaluate.py."""
    _delegate("evaluate", ctx)


@app.command(context_settings=_PASSTHROUGH)
def benchmark(ctx: typer.Context) -> None:
    """Run standalone benchmarking. See scripts/benchmark.py."""
    _delegate("benchmark", ctx)


@app.command(context_settings=_PASSTHROUGH)
def infer(ctx: typer.Context) -> None:
    """Run single-prompt inference. See scripts/infer.py."""
    _delegate("infer", ctx)


@app.command(name="download-data", context_settings=_PASSTHROUGH)
def download_data(ctx: typer.Context) -> None:
    """Download and cache configured datasets. See scripts/download_data.py."""
    _delegate("download_data", ctx)


@app.command(context_settings=_PASSTHROUGH)
def preprocess(ctx: typer.Context) -> None:
    """Run the preprocessing pipeline. See scripts/preprocess.py."""
    _delegate("preprocess", ctx)


@app.command(context_settings=_PASSTHROUGH)
def resume(ctx: typer.Context) -> None:
    """Resume training from a checkpoint - alias for `train +resume_from=<dir>`."""
    _delegate("train", ctx)


@app.command(name="list-checkpoints")
def list_checkpoints(
    output_dir: str = typer.Option("outputs", help="Root output directory containing checkpoints/."),
) -> None:
    """List available checkpoints and metadata."""
    import torch

    from slm_research.training.checkpointing import list_checkpoints as get_checkpoints

    checkpoints = get_checkpoints(output_dir)
    if not checkpoints:
        typer.echo(f"No checkpoints found under {output_dir}/checkpoints")
        raise typer.Exit()

    for path in checkpoints:
        state = torch.load(path / "training_state.pt", map_location="cpu")
        typer.echo(
            f"{path.name}  run_id={state.get('run_id', 'unknown')}  "
            f"epoch={state.get('epoch')}  step={state.get('global_step')}"
        )


@app.command()
def export() -> None:
    """Pull W&B run history via API for report/plot generation."""
    raise NotImplementedError("Phase 9: implement W&B API export for the LaTeX report.")


if __name__ == "__main__":
    app()
