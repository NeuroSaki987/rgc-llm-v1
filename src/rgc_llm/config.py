from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    hidden_dim: int = 128
    phase_dim: int = 64
    max_nodes: int = 256
    resonance_steps: int = 3
    deep_steps: int = 4
    operator_threshold: float = 0.55
    scheduler_actions: List[str]


class TrainingConfig(BaseModel):
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    epochs: int = 3
    grad_clip: float = 1.0
    text_loss_weight: float = 1.0
    graph_consistency_weight: float = 0.2
    operator_weight: float = 0.2
    scheduler_weight: float = 0.1
    self_driven_weight: float = 0.1
    num_workers: int = 0
    shuffle: bool = True
    checkpoint_every: int = 1
    output_dir: str = "outputs/default"


class Datum(BaseModel):
    input: str
    target: str
    metadata: dict = Field(default_factory=dict)


class DataConfig(BaseModel):
    toy_samples: List[Datum] = Field(default_factory=list)
    train_file: Optional[str] = None
    valid_file: Optional[str] = None
    file_format: str = "auto"
    text_field: str = "input"
    target_field: str = "target"

    @model_validator(mode="after")
    def validate_source(self) -> "DataConfig":
        if not self.toy_samples and not self.train_file:
            raise ValueError("Provide either data.toy_samples or data.train_file")
        return self


class AppConfig(BaseModel):
    seed: int = 42
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    @staticmethod
    def from_yaml(path: str | Path) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return AppConfig.model_validate(raw)
