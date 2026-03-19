from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch
import typer

from rgc_llm.config import AppConfig
from rgc_llm.model import RGCLLM
from rgc_llm.utils.checkpointing import load_checkpoint_into_model

app = typer.Typer(help="Export or inspect checkpoints for RGC-LLM.")


@app.callback()
def main() -> None:
    """Export commands."""
    return None


@app.command("state-dict")
def export_state_dict(
    config: str = typer.Option(..., help="Path to yaml config."),
    checkpoint: str = typer.Option(..., help="Checkpoint path."),
    output: str = typer.Option(..., help="Output .pt path for a pure state_dict."),
) -> None:
    cfg = AppConfig.from_yaml(config)
    model = RGCLLM(cfg.model)
    load_checkpoint_into_model(model, checkpoint, map_location="cpu")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    typer.echo(f"exported state_dict to {out}")


@app.command("modules")
def export_modules(
    config: str = typer.Option(..., help="Path to yaml config."),
    checkpoint: str = typer.Option(..., help="Checkpoint path."),
    output_dir: str = typer.Option(..., help="Directory to store module weights."),
    include: Optional[List[str]] = typer.Option(None, help="Limit export to specific submodules."),
) -> None:
    cfg = AppConfig.from_yaml(config)
    model = RGCLLM(cfg.model)
    load_checkpoint_into_model(model, checkpoint, map_location="cpu")
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    module_map = {
        "encoder": model.encoder,
        "resonance": model.resonance,
        "operators": model.operators,
        "scheduler": model.scheduler,
        "decoder": model.decoder,
    }
    selected = set(include) if include else set(module_map.keys())
    for name, module in module_map.items():
        if name not in selected:
            continue
        target = outdir / f"{name}.pt"
        torch.save(module.state_dict(), target)
        typer.echo(f"exported {name} -> {target}")


@app.command("inspect")
def inspect_checkpoint(checkpoint: str = typer.Option(..., help="Checkpoint path.")) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        meta = {k: v for k, v in payload.items() if k != "state_dict"}
    else:
        state_dict = payload
        meta = {"format": "raw-state-dict"}
    summary = {
        "meta": meta,
        "num_tensors": len(state_dict),
        "total_params": int(sum(v.numel() for v in state_dict.values())),
        "tensors": [
            {"name": k, "shape": list(v.shape), "dtype": str(v.dtype)}
            for k, v in state_dict.items()
        ],
    }
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
