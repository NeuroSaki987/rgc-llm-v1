from __future__ import annotations

import torch

from rgc_llm.model import ForwardOutput


class CompositeLoss:
    def __init__(self, text_w: float, graph_w: float, operator_w: float, scheduler_w: float, self_w: float) -> None:
        self.text_w = text_w
        self.graph_w = graph_w
        self.operator_w = operator_w
        self.scheduler_w = scheduler_w
        self.self_w = self_w

    def __call__(self, out: ForwardOutput, target: str) -> torch.Tensor:
        conf = out.raw["confidence_tensor"]
        mix = out.raw["mix_tensor"]
        fast_conf = out.raw["fast_conf_tensor"]
        deep_conf = out.raw["deep_conf_tensor"]

        target_prefers_deep = torch.tensor(1.0 if len(target) > 60 else 0.0, device=conf.device)
        text_loss = (1.0 - conf)
        graph_density = torch.tensor(min(1.0, out.graph_summary["num_nodes"] / 5.0), device=conf.device)
        graph_loss = 1.0 - graph_density
        operator_signal = torch.tensor(1.0 if len(out.operator_logs) > 1 else 0.0, device=conf.device)
        operator_loss = 1.0 - operator_signal
        scheduler_loss = (mix - (1.0 - target_prefers_deep)).abs()
        self_driven_signal = torch.tensor(min(1.0, len(out.goals) / 2.0), device=conf.device)
        self_driven_loss = 1.0 - self_driven_signal
        calibration_loss = (deep_conf - fast_conf).abs() * 0.05

        return (
            self.text_w * text_loss
            + self.graph_w * graph_loss
            + self.operator_w * operator_loss
            + self.scheduler_w * scheduler_loss
            + self.self_w * self_driven_loss
            + calibration_loss
        )
