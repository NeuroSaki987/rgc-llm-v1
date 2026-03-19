from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rgc_llm.core.graph import DynamicGraph, NodeType


@dataclass
class SpawnedGoal:
    target_node: int
    objective_type: str
    budget: int
    deadline: int
    drive: float


class AutotelicGoalField:
    def __init__(self, drive_threshold: float = 1.8) -> None:
        self.drive_threshold = drive_threshold

    def spawn(self, graph: DynamicGraph) -> List[SpawnedGoal]:
        goals: List[SpawnedGoal] = []
        for nid, node in graph.nodes.items():
            delta = node.state.activity * node.state.utility
            drive = (
                0.8 * node.state.utility
                + 0.9 * (1.0 - node.state.confidence)
                + 0.7 * node.state.conflict
                + 0.5 * node.state.novelty
                + 0.9 * delta
            )
            if drive > self.drive_threshold:
                objective = "verify" if node.state.conflict > 0.2 else "expand"
                if node.node_type == NodeType.GOAL:
                    objective = "plan"
                goals.append(SpawnedGoal(nid, objective, budget=3, deadline=2, drive=drive))
        return sorted(goals, key=lambda g: g.drive, reverse=True)
