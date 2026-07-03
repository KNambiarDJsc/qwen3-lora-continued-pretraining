"""Typer CLI entrypoint.

Commands (see architecture spec Section 12): train, evaluate, benchmark,
infer, download-data, preprocess, resume, list-checkpoints, export.
Orchestration only — delegates to scripts/*.py and src/slm_research/*.
"""

import typer

app = typer.Typer(help="SLM research framework CLI.")


@app.command()
def train() -> None:
    """Run training. See scripts/train.py."""
    raise NotImplementedError("Phase 2 follow-up: delegate to scripts/train.py")


@app.command()
def evaluate() -> None:
    """Run standalone evaluation. See scripts/evaluate.py."""
    raise NotImplementedError("Phase 2 follow-up: delegate to scripts/evaluate.py")


@app.command()
def benchmark() -> None:
    """Run standalone benchmarking. See scripts/benchmark.py."""
    raise NotImplementedError("Phase 2 follow-up: delegate to scripts/benchmark.py")


@app.command()
def infer() -> None:
    """Run single-prompt inference. See scripts/infer.py."""
    raise NotImplementedError("Phase 2 follow-up: delegate to scripts/infer.py")


@app.command(name="download-data")
def download_data() -> None:
    """Download and cache configured datasets. See scripts/download_data.py."""
    raise NotImplementedError("Phase 2 follow-up: delegate to scripts/download_data.py")


@app.command()
def preprocess() -> None:
    """Run the preprocessing pipeline. See scripts/preprocess.py."""
    raise NotImplementedError("Phase 2 follow-up: delegate to scripts/preprocess.py")


@app.command()
def resume() -> None:
    """Resume training from a checkpoint."""
    raise NotImplementedError("Phase 6: implement resume path in checkpointing.py first.")


@app.command(name="list-checkpoints")
def list_checkpoints() -> None:
    """List available checkpoints and metadata."""
    raise NotImplementedError("Phase 6 follow-up: implement checkpoint directory listing.")


@app.command()
def export() -> None:
    """Pull W&B run history via API for report/plot generation."""
    raise NotImplementedError("Phase 9: implement W&B API export for the LaTeX report.")


if __name__ == "__main__":
    app()
