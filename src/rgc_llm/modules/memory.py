from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from rgc_llm.core.graph import DynamicGraph


@dataclass
class MemoryItem:
    text: str
    vector: torch.Tensor
    level: str


@dataclass
class HierarchicalMemoryField:
    stm: List[MemoryItem] = field(default_factory=list)
    mtm: List[MemoryItem] = field(default_factory=list)
    ltm: List[MemoryItem] = field(default_factory=list)

    def ingest_graph(self, graph: DynamicGraph) -> None:
        self.stm = [MemoryItem(text=n.text, vector=n.state.h.detach().clone(), level="stm") for n in graph.top_active_nodes(8)]
        self.mtm = [MemoryItem(text=n.text, vector=n.state.h.detach().clone(), level="mtm") for n in graph.nodes.values() if n.node_type.value == "macro"]

    def retrieve(self, query: torch.Tensor, top_k: int = 3) -> List[MemoryItem]:
        pool = self.stm + self.mtm + self.ltm
        if not pool:
            return []
        scored = []
        for item in pool:
            score = torch.cosine_similarity(query.unsqueeze(0), item.vector.unsqueeze(0)).item()
            scored.append((score, item))
        return [it for _, it in sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]]
