from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch


class NodeType(str, Enum):
    ENTITY = "entity"
    RELATION = "relation"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    HYPOTHESIS = "hypothesis"
    CLAIM = "claim"
    MACRO = "macro"


class EdgeType(str, Enum):
    DEPENDS = "depends"
    IMPLIES = "implies"
    CONFLICTS = "conflicts"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    REFERS = "refers"
    ALTERNATIVE = "alternative"
    SUPPORTS = "supports"


@dataclass
class NodeState:
    h: torch.Tensor
    phase: torch.Tensor
    activity: float = 0.5
    confidence: float = 0.5
    utility: float = 0.5
    novelty: float = 0.5
    conflict: float = 0.0
    solvability: float = 0.5


@dataclass
class Node:
    node_id: int
    text: str
    node_type: NodeType
    state: NodeState
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class Edge:
    src: int
    dst: int
    edge_type: EdgeType
    weight: float = 1.0
    tau: float = 0.0


class DynamicGraph:
    def __init__(self, hidden_dim: int, phase_dim: int, device: str = "cpu") -> None:
        self.hidden_dim = hidden_dim
        self.phase_dim = phase_dim
        self.device = device
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []
        self._next_id = 0

    def add_node(
        self,
        text: str,
        node_type: NodeType,
        h: Optional[torch.Tensor] = None,
        phase: Optional[torch.Tensor] = None,
        **state_kwargs,
    ) -> int:
        h = h if h is not None else torch.zeros(self.hidden_dim, device=self.device)
        phase = phase if phase is not None else torch.zeros(self.phase_dim, device=self.device)
        node = Node(
            node_id=self._next_id,
            text=text,
            node_type=node_type,
            state=NodeState(h=h, phase=phase, **state_kwargs),
        )
        self.nodes[node.node_id] = node
        self._next_id += 1
        return node.node_id

    def add_edge(self, src: int, dst: int, edge_type: EdgeType, weight: float = 1.0, tau: float = 0.0) -> None:
        self.edges.append(Edge(src=src, dst=dst, edge_type=edge_type, weight=weight, tau=tau))

    def neighbors(self, node_id: int) -> List[Edge]:
        return [e for e in self.edges if e.dst == node_id]

    def outgoing(self, node_id: int) -> List[Edge]:
        return [e for e in self.edges if e.src == node_id]

    def pooled_state(self) -> torch.Tensor:
        if not self.nodes:
            return torch.zeros(self.hidden_dim + self.phase_dim, device=self.device)
        hs = torch.stack([n.state.h for n in self.nodes.values()])
        ps = torch.stack([n.state.phase for n in self.nodes.values()])
        return torch.cat([hs.mean(dim=0), ps.mean(dim=0)], dim=0)

    def top_active_nodes(self, k: int = 5) -> List[Node]:
        return sorted(self.nodes.values(), key=lambda n: n.state.activity, reverse=True)[:k]

    def conflict_energy(self) -> torch.Tensor:
        energy = torch.tensor(0.0, device=self.device)
        for e in self.edges:
            if e.edge_type == EdgeType.CONFLICTS:
                a = self.nodes[e.src].state.confidence
                b = self.nodes[e.dst].state.confidence
                energy = energy + torch.tensor(a * b * e.weight, device=self.device)
        return energy

    def prune_isolated(self) -> None:
        connected = {e.src for e in self.edges} | {e.dst for e in self.edges}
        remove = [nid for nid in self.nodes if nid not in connected and self.nodes[nid].state.activity < 0.05]
        for nid in remove:
            del self.nodes[nid]

    def summary(self) -> dict:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "top_active": [n.text for n in self.top_active_nodes()],
            "conflict_energy": float(self.conflict_energy().item()),
        }
