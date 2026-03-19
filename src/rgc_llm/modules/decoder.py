from __future__ import annotations

from typing import List

import torch
from torch import nn

from rgc_llm.core.graph import DynamicGraph
from rgc_llm.modules.goal_field import SpawnedGoal


class FastDecoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, graph: DynamicGraph) -> tuple[str, torch.Tensor]:
        top = graph.top_active_nodes(3)
        summary = " | ".join(n.text for n in top) if top else "<empty graph>"
        confidence = torch.sigmoid(self.score(graph.pooled_state())).squeeze()
        return f"FAST: {summary}", confidence


class DeepDecoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, graph: DynamicGraph, goals: List[SpawnedGoal]) -> tuple[str, torch.Tensor]:
        active = graph.top_active_nodes(5)
        goal_text = ", ".join(f"{g.objective_type}@{g.target_node}" for g in goals[:3]) or "none"
        detail = " ; ".join(n.text for n in active) if active else "<empty graph>"
        confidence = torch.sigmoid(self.score(graph.pooled_state())).squeeze()
        return f"DEEP: detail=[{detail}] goals=[{goal_text}]", confidence


class DualDecoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fast = FastDecoder(input_dim)
        self.deep = DeepDecoder(input_dim)
        self.mix = nn.Linear(3, 1)

    def forward(
        self,
        graph: DynamicGraph,
        goals: List[SpawnedGoal],
        urgency: float,
        uncertainty: float,
        conflict: float,
    ) -> dict:
        fast_text, fast_c = self.fast(graph)
        deep_text, deep_c = self.deep(graph, goals)
        rho = torch.sigmoid(self.mix(torch.tensor([urgency, uncertainty, conflict], dtype=torch.float32))).squeeze()
        final_text = fast_text if rho.item() >= 0.5 else deep_text
        final_conf = rho * fast_c + (1 - rho) * deep_c
        return {
            "fast_text": fast_text,
            "deep_text": deep_text,
            "final_text": final_text,
            "mix": float(rho.item()),
            "confidence": float(final_conf.item()),
            "mix_tensor": rho,
            "confidence_tensor": final_conf,
            "fast_conf_tensor": fast_c,
            "deep_conf_tensor": deep_c,
        }
