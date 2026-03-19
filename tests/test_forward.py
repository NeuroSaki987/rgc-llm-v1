from rgc_llm.config import AppConfig
from rgc_llm.model import RGCLLM


def test_forward_returns_output():
    cfg = AppConfig.from_yaml("configs/default.yaml")
    model = RGCLLM(cfg.model)
    out = model("Compare solar and wind power under storage constraints.", deep=True)
    assert isinstance(out.text, str)
    assert "num_nodes" in out.graph_summary
    assert isinstance(out.goals, list)
