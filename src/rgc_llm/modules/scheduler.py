from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from rgc_llm.core.graph import DynamicGraph


class MetaScheduler(nn.Module):
    def __init__(self, input_dim: int, actions: List[str]) -> None:
        super().__init__()
        self.actions = actions
        self.policy = nn.Sequential(
            nn.Linear(input_dim + 3, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, len(actions)),
        )

    def forward(self, graph: DynamicGraph, urgency: float = 0.5) -> Tuple[str, torch.Tensor]:
        pooled = graph.pooled_state()
        uncertainty = 1.0 - sum(n.state.confidence for n in graph.nodes.values()) / max(1, len(graph.nodes))
        conflict = float(graph.conflict_energy().item())
        x = torch.cat([pooled, torch.tensor([urgency, uncertainty, conflict], device=pooled.device)], dim=0)
        logits = self.policy(x)
        probs = torch.softmax(logits, dim=0)
        action_id = torch.argmax(probs).item()
        return self.actions[action_id], probs
