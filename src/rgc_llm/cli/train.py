from __future__ import annotations

from pathlib import Path

import typer

from rgc_llm.config import AppConfig
from rgc_llm.training.trainer import Trainer
from rgc_llm.utils.checkpointing import save_checkpoint

app = typer.Typer(help="Train the RGC-LLM engineering prototype.")


@app.callback()
def main() -> None:
    """Training commands."""
    return None


@app.command()
def fit(
    config: str = typer.Option(..., help="Path to yaml config."),
    save: str = typer.Option("", help="Optional final checkpoint path. Defaults to <output_dir>/final.pt"),
    device: str = typer.Option("cpu", help="Execution device, e.g. cpu or cuda."),
) -> None:
    cfg = AppConfig.from_yaml(config)
    trainer = Trainer(cfg, device=device)
    model = trainer.fit()
    save_path = Path(save) if save else Path(cfg.training.output_dir) / "final.pt"
    save_checkpoint(model, cfg, save_path, epoch=cfg.training.epochs)
    typer.echo(f"saved model to {save_path}")
