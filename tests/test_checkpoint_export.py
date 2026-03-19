import torch

from rgc_llm.config import AppConfig
from rgc_llm.model import RGCLLM
from rgc_llm.utils.checkpointing import load_checkpoint_into_model, save_checkpoint


def test_save_and_load_checkpoint(tmp_path):
    cfg = AppConfig.from_yaml("configs/default.yaml")
    model = RGCLLM(cfg.model)
    ckpt = tmp_path / "model.pt"
    save_checkpoint(model, cfg, ckpt, epoch=1)
    other = RGCLLM(cfg.model)
    payload = load_checkpoint_into_model(other, ckpt)
    assert payload["epoch"] == 1
    assert set(model.state_dict().keys()) == set(other.state_dict().keys())
    for key, value in other.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)
