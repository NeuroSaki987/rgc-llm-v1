from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch import nn

from rgc_llm.config import ModelConfig
from rgc_llm.core.graph import DynamicGraph
from rgc_llm.modules.decoder import DualDecoder
from rgc_llm.modules.event_encoder import EventEncoder
from rgc_llm.modules.goal_field import AutotelicGoalField
from rgc_llm.modules.memory import HierarchicalMemoryField
from rgc_llm.modules.operators import GraphCalculusOperators, OperatorLog
from rgc_llm.modules.resonance import ResonantPropagationEngine
from rgc_llm.modules.scheduler import MetaScheduler


@dataclass
class ForwardOutput:
    text: str
    graph_summary: Dict[str, Any]
    operator_logs: List[str]
    goals: List[Dict[str, Any]]
    raw: Dict[str, Any]


class RGCLLM(nn.Module):
    def __init__(self, config: ModelConfig, device: str = "cpu") -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.encoder = EventEncoder(config.hidden_dim, config.phase_dim)
        self.resonance = ResonantPropagationEngine(config.hidden_dim, config.phase_dim)
        self.operators = GraphCalculusOperators(config.hidden_dim, config.operator_threshold)
        self.scheduler = MetaScheduler(config.hidden_dim + config.phase_dim, config.scheduler_actions)
        self.memory = HierarchicalMemoryField()
        self.goal_field = AutotelicGoalField()
        self.decoder = DualDecoder(config.hidden_dim + config.phase_dim)

    def initialize_graph(self, text: str) -> DynamicGraph:
        graph = DynamicGraph(self.config.hidden_dim, self.config.phase_dim, device=self.device)
        graph = self.encoder.build_graph(text, graph)
        return graph

    def _apply_action(self, graph: DynamicGraph, action: str) -> List[OperatorLog]:
        if action == "merge":
            return self.operators.merge(graph)
        if action == "split":
            return self.operators.split(graph)
        if action == "infer":
            return self.operators.infer(graph)
        if action == "resolve":
            return self.operators.resolve(graph)
        if action == "compress":
            return self.operators.compress(graph)
        return []

    def forward(self, text: str, deep: bool = True, urgency: float = 0.4) -> ForwardOutput:
        graph = self.initialize_graph(text)
        operator_logs: List[str] = []

        self.resonance(graph, steps=self.config.resonance_steps)
        action, _ = self.scheduler(graph, urgency=urgency)
        operator_logs.extend([f"scheduler->{action}"])
        operator_logs.extend(log.detail for log in self._apply_action(graph, action))

        if deep:
            for _ in range(self.config.deep_steps):
                self.resonance(graph, steps=1)
                d_action, _ = self.scheduler(graph, urgency=urgency * 0.8)
                if d_action == "halt":
                    operator_logs.append("scheduler->halt")
                    break
                operator_logs.extend(log.detail for log in self._apply_action(graph, d_action))

        self.memory.ingest_graph(graph)
        goals = self.goal_field.spawn(graph)

        uncertainty = 1.0 - sum(n.state.confidence for n in graph.nodes.values()) / max(1, len(graph.nodes))
        conflict = float(graph.conflict_energy().item())
        decoded = self.decoder(graph, goals, urgency=urgency, uncertainty=uncertainty, conflict=conflict)
        output = ForwardOutput(
            text=decoded["final_text"],
            graph_summary=graph.summary(),
            operator_logs=operator_logs,
            goals=[goal.__dict__ for goal in goals[:5]],
            raw=decoded,
        )
        return output
