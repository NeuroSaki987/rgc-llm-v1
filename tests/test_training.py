from rgc_llm.config import AppConfig
from rgc_llm.training.trainer import Trainer


def test_trainer_runs_one_fit_cycle():
    cfg = AppConfig.from_yaml("configs/default.yaml")
    cfg.training.epochs = 1
    trainer = Trainer(cfg)
    model = trainer.fit()
    assert model is not None
