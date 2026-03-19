from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from rgc_llm.config import AppConfig
from rgc_llm.model import RGCLLM


def build_checkpoint_payload(model: RGCLLM, config: AppConfig, epoch: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "format": "rgc-llm-checkpoint",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "epoch": epoch,
        "model_config": config.model.model_dump(),
        "app_config": config.model_dump(),
        "state_dict": model.state_dict(),
    }
    if extra:
        payload["extra"] = extra
    return payload


def save_checkpoint(model: RGCLLM, config: AppConfig, path: str | Path, epoch: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_checkpoint_payload(model, config, epoch=epoch, extra=extra), path)
    return path


def load_checkpoint_into_model(model: RGCLLM, checkpoint_path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict)
    return payload if isinstance(payload, dict) else {"state_dict": state_dict}
