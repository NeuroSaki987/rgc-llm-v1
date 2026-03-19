from rgc_llm.config import AppConfig
from rgc_llm.model import RGCLLM


def test_graph_initialization_has_nodes():
    cfg = AppConfig.from_yaml("configs/default.yaml")
    model = RGCLLM(cfg.model)
    graph = model.initialize_graph("If latency rises, inspect the server, but the cause is uncertain.")
    assert len(graph.nodes) >= 2
    assert len(graph.edges) >= 1
