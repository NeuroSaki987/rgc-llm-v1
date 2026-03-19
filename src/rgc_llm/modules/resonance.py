from __future__ import annotations

import math

import torch
from torch import nn

from rgc_llm.core.graph import DynamicGraph


class ResonantPropagationEngine(nn.Module):
    def __init__(self, hidden_dim: int, phase_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.phase_dim = phase_dim
        self.coupling = nn.Linear(hidden_dim + phase_dim, hidden_dim)
        self.phase_update = nn.Linear(hidden_dim + phase_dim, phase_dim)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, graph: DynamicGraph, steps: int = 1) -> DynamicGraph:
        if not graph.nodes:
            return graph
        for _ in range(steps):
            new_h = {}
            new_phase = {}
            for nid, node in graph.nodes.items():
                incoming = graph.neighbors(nid)
                if not incoming:
                    new_h[nid] = node.state.h
                    new_phase[nid] = node.state.phase
                    continue
                msg_h = torch.zeros_like(node.state.h)
                msg_p = torch.zeros_like(node.state.phase)
                for edge in incoming:
                    src = graph.nodes[edge.src]
                    phase_gap = (node.state.phase - src.state.phase).mean()
                    resonance = edge.weight * math.cos(float(phase_gap.item()) - edge.tau)
                    feature = torch.cat([src.state.h, src.state.phase], dim=0)
                    msg_h = msg_h + resonance * self.coupling(feature)
                    msg_p = msg_p + resonance * self.phase_update(feature)
                new_h[nid] = torch.tanh(msg_h - self.bias + node.state.h)
                new_phase[nid] = torch.tanh(msg_p + node.state.phase)
            for nid in graph.nodes:
                graph.nodes[nid].state.h = new_h[nid]
                graph.nodes[nid].state.phase = new_phase[nid]
                graph.nodes[nid].state.activity = float(torch.sigmoid(new_h[nid].norm() / 10).item())
                graph.nodes[nid].state.confidence = float(torch.sigmoid(new_phase[nid].mean()).item())
        return graph
