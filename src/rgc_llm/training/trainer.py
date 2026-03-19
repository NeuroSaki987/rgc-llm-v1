from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from rich.console import Console
from torch.optim import AdamW
from torch.utils.data import DataLoader

from rgc_llm.config import AppConfig
from rgc_llm.model import RGCLLM
from rgc_llm.training.dataset import build_dataset
from rgc_llm.training.losses import CompositeLoss
from rgc_llm.utils.checkpointing import save_checkpoint


console = Console()


class Trainer:
    def __init__(self, config: AppConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        self._seed_all(config.seed)
        self.model = RGCLLM(config.model, device=device).to(device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
        self.loss_fn = CompositeLoss(
            config.training.text_loss_weight,
            config.training.graph_consistency_weight,
            config.training.operator_weight,
            config.training.scheduler_weight,
            config.training.self_driven_weight,
        )
        self.train_dataset = build_dataset(config.data, split="train")
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=config.training.shuffle,
            num_workers=config.training.num_workers,
            collate_fn=self._collate,
        )
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def _collate(batch: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        texts, targets = zip(*batch)
        return list(texts), list(targets)

    def fit(self) -> RGCLLM:
        self.model.train()
        for epoch in range(self.config.training.epochs):
            losses = []
            for texts, targets in self.train_loader:
                batch_loss = torch.tensor(0.0, device=self.device)
                self.optimizer.zero_grad()
                for text, target in zip(texts, targets):
                    out = self.model(text, deep=True, urgency=0.4)
                    batch_loss = batch_loss + self.loss_fn(out, target)
                batch_loss = batch_loss / max(1, len(texts))
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
                self.optimizer.step()
                losses.append(float(batch_loss.item()))
            avg_loss = sum(losses) / max(1, len(losses))
            console.print(f"[green]Epoch {epoch + 1}[/green] loss={avg_loss:.4f}")
            if (epoch + 1) % self.config.training.checkpoint_every == 0:
                ckpt_path = self.output_dir / f"epoch_{epoch + 1}.pt"
                save_checkpoint(self.model, self.config, ckpt_path, epoch=epoch + 1, extra={"avg_loss": avg_loss})
                console.print(f"[cyan]checkpoint[/cyan] {ckpt_path}")
        return self.model
