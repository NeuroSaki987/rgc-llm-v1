from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from rgc_llm.core.graph import DynamicGraph, EdgeType, NodeType


@dataclass
class Event:
    kind: str
    text: str


class EventEncoder(nn.Module):
    def __init__(self, hidden_dim: int, phase_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.phase_dim = phase_dim
        self.proj = nn.Linear(32, hidden_dim)
        self.phase_proj = nn.Linear(32, phase_dim)

    def _embed_text(self, text: str) -> torch.Tensor:
        vec = torch.zeros(32)
        for ch in text[:128]:
            vec[ord(ch) % 32] += 1.0
        return vec / max(1.0, vec.sum().item())

    def extract_events(self, text: str) -> List[Event]:
        parts = [p.strip() for p in re.split(r"[，,。.!?;；]\s*", text) if p.strip()]
        events: List[Event] = []
        for p in parts:
            lower = p.lower()
            if any(k in lower for k in ["if", "如果", "若"]):
                kind = "hypothesis"
            elif any(k in lower for k in ["should", "需要", "必须", "goal", "目标"]):
                kind = "goal"
            elif any(k in lower for k in ["but", "however", "但是", "但"]):
                kind = "contrast"
            elif any(k in lower for k in ["because", "caused", "导致", "因为"]):
                kind = "causal"
            elif any(k in lower for k in ["compare", "versus", "vs", "比较"]):
                kind = "comparison"
            else:
                kind = "claim"
            events.append(Event(kind=kind, text=p))
        return events

    def build_graph(self, text: str, graph: DynamicGraph) -> DynamicGraph:
        events = self.extract_events(text)
        prev_id = None
        for event in events:
            base = self._embed_text(event.text)
            h = self.proj(base.to(self.proj.weight.device))
            phase = self.phase_proj(base.to(self.phase_proj.weight.device))
            if event.kind == "goal":
                ntype = NodeType.GOAL
            elif event.kind == "hypothesis":
                ntype = NodeType.HYPOTHESIS
            elif event.kind in {"causal", "comparison", "contrast"}:
                ntype = NodeType.RELATION
            else:
                ntype = NodeType.CLAIM
            nid = graph.add_node(
                text=event.text,
                node_type=ntype,
                h=h,
                phase=phase,
                activity=0.7,
                confidence=0.6,
                utility=0.5 if ntype != NodeType.GOAL else 0.8,
                novelty=0.4,
                conflict=0.2 if event.kind == "contrast" else 0.0,
                solvability=0.7,
            )
            if prev_id is not None:
                graph.add_edge(prev_id, nid, EdgeType.TEMPORAL, weight=0.7, tau=0.1)
            prev_id = nid
        for i in list(graph.nodes):
            for j in list(graph.nodes):
                if i >= j:
                    continue
                ti = graph.nodes[i].text.lower()
                tj = graph.nodes[j].text.lower()
                if len(set(ti.split()) & set(tj.split())) > 0:
                    graph.add_edge(i, j, EdgeType.REFERS, weight=0.4)
                    graph.add_edge(j, i, EdgeType.REFERS, weight=0.4)
        return graph
