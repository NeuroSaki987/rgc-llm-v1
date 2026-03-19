from __future__ import annotations

import json
from pathlib import Path

import typer

from rgc_llm.config import AppConfig
from rgc_llm.model import RGCLLM
from rgc_llm.utils.checkpointing import load_checkpoint_into_model

app = typer.Typer(help="Run inference with the RGC-LLM engineering prototype.")


@app.callback()
def main() -> None:
    """Inference commands."""
    return None


@app.command()
def run(
    config: str = typer.Option(..., help="Path to yaml config."),
    text: str = typer.Option(..., help="Input text."),
    checkpoint: str = typer.Option("", help="Optional checkpoint path."),
    deep: bool = typer.Option(True, help="Enable deep reasoning path."),
    urgency: float = typer.Option(0.4, help="Urgency in [0,1]. Higher favors fast path."),
) -> None:
    cfg = AppConfig.from_yaml(config)
    model = RGCLLM(cfg.model)
    checkpoint_meta = {}
    if checkpoint and Path(checkpoint).exists():
        checkpoint_meta = load_checkpoint_into_model(model, checkpoint, map_location="cpu")
    model.eval()
    with __import__("torch").no_grad():
        out = model(text, deep=deep, urgency=urgency)
    payload = {
        "text": out.text,
        "graph_summary": out.graph_summary,
        "operator_logs": out.operator_logs,
        "goals": out.goals,
        "decoder": {k: v for k, v in out.raw.items() if not hasattr(v, "shape")},
        "checkpoint": {
            "path": checkpoint or None,
            "epoch": checkpoint_meta.get("epoch") if isinstance(checkpoint_meta, dict) else None,
            "created_at": checkpoint_meta.get("created_at") if isinstance(checkpoint_meta, dict) else None,
        },
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
