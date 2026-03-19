from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from rgc_llm.core.graph import DynamicGraph, EdgeType, NodeType


@dataclass
class OperatorLog:
    name: str
    detail: str


class GraphCalculusOperators(nn.Module):
    def __init__(self, hidden_dim: int, threshold: float = 0.55) -> None:
        super().__init__()
        self.threshold = threshold
        self.merge_scorer = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.split_scorer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def merge(self, graph: DynamicGraph) -> List[OperatorLog]:
        logs: List[OperatorLog] = []
        node_ids = list(graph.nodes.keys())
        merged = set()
        for i, nid_i in enumerate(node_ids):
            if nid_i in merged:
                continue
            for nid_j in node_ids[i + 1:]:
                if nid_j in merged:
                    continue
                a = graph.nodes[nid_i]
                b = graph.nodes[nid_j]
                score = torch.sigmoid(self.merge_scorer(torch.cat([a.state.h, b.state.h]))).item()
                token_overlap = len(set(a.text.lower().split()) & set(b.text.lower().split()))
                if score > self.threshold and token_overlap > 1:
                    a.text = f"{a.text} | {b.text}"
                    a.state.h = (a.state.h + b.state.h) / 2
                    a.state.phase = (a.state.phase + b.state.phase) / 2
                    a.state.activity = max(a.state.activity, b.state.activity)
                    for e in graph.edges:
                        if e.src == nid_j:
                            e.src = nid_i
                        if e.dst == nid_j:
                            e.dst = nid_i
                    merged.add(nid_j)
                    logs.append(OperatorLog("merge", f"merged {nid_i} <- {nid_j}"))
        for nid in merged:
            graph.nodes.pop(nid, None)
        return logs

    def split(self, graph: DynamicGraph) -> List[OperatorLog]:
        logs: List[OperatorLog] = []
        additions = []
        for nid, node in list(graph.nodes.items()):
            score = torch.sigmoid(self.split_scorer(node.state.h)).item()
            chunks = [c.strip() for c in node.text.replace(" and ", ",").replace("与", "，").split(",") if c.strip()]
            if score > self.threshold and len(chunks) >= 2 and len(node.text) > 25:
                for chunk in chunks[:3]:
                    new_id = graph.add_node(
                        text=chunk,
                        node_type=node.node_type,
                        h=node.state.h.clone(),
                        phase=node.state.phase.clone(),
                        activity=node.state.activity * 0.9,
                        confidence=node.state.confidence,
                        utility=node.state.utility,
                        novelty=node.state.novelty,
                        conflict=node.state.conflict,
                        solvability=node.state.solvability,
                    )
                    additions.append((nid, new_id))
                logs.append(OperatorLog("split", f"split node {nid}"))
        for parent, child in additions:
            graph.add_edge(parent, child, EdgeType.DEPENDS, weight=0.8)
        return logs

    def infer(self, graph: DynamicGraph) -> List[OperatorLog]:
        logs: List[OperatorLog] = []
        goals = [n for n in graph.nodes.values() if n.node_type == NodeType.GOAL]
        claims = [n for n in graph.nodes.values() if n.node_type in {NodeType.CLAIM, NodeType.HYPOTHESIS}]
        for goal in goals[:3]:
            for claim in claims[:3]:
                overlap = len(set(goal.text.lower().split()) & set(claim.text.lower().split()))
                if overlap > 0:
                    new_id = graph.add_node(
                        text=f"Inference: {claim.text} supports goal {goal.text}",
                        node_type=NodeType.CLAIM,
                        h=(goal.state.h + claim.state.h) / 2,
                        phase=(goal.state.phase + claim.state.phase) / 2,
                        activity=0.65,
                        confidence=0.55,
                        utility=0.75,
                        novelty=0.5,
                        conflict=0.0,
                        solvability=0.7,
                    )
                    graph.add_edge(claim.node_id, new_id, EdgeType.SUPPORTS, weight=0.7)
                    graph.add_edge(new_id, goal.node_id, EdgeType.IMPLIES, weight=0.7)
                    logs.append(OperatorLog("infer", f"created inference node {new_id}"))
        return logs

    def resolve(self, graph: DynamicGraph) -> List[OperatorLog]:
        logs: List[OperatorLog] = []
        for edge in graph.edges:
            if edge.edge_type == EdgeType.CONFLICTS:
                src = graph.nodes[edge.src]
                dst = graph.nodes[edge.dst]
                if src.state.confidence >= dst.state.confidence:
                    dst.state.confidence *= 0.9
                else:
                    src.state.confidence *= 0.9
                logs.append(OperatorLog("resolve", f"resolved conflict {edge.src}<->{edge.dst}"))
        return logs

    def compress(self, graph: DynamicGraph) -> List[OperatorLog]:
        logs: List[OperatorLog] = []
        active = graph.top_active_nodes(k=3)
        if len(active) >= 2:
            text = " ; ".join(n.text for n in active)
            h = torch.stack([n.state.h for n in active]).mean(dim=0)
            phase = torch.stack([n.state.phase for n in active]).mean(dim=0)
            macro_id = graph.add_node(
                text=f"Macro[{text}]",
                node_type=NodeType.MACRO,
                h=h,
                phase=phase,
                activity=0.8,
                confidence=0.7,
                utility=0.8,
                novelty=0.6,
                conflict=0.0,
                solvability=0.8,
            )
            for n in active:
                graph.add_edge(n.node_id, macro_id, EdgeType.SUPPORTS, weight=0.6)
            logs.append(OperatorLog("compress", f"created macro node {macro_id}"))
        return logs
