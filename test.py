import hydra
import lightning as L
import rootutils
from omegaconf import DictConfig

from REAPS import utils
from REAPS.utils import task_wrapper

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@task_wrapper
def run_test(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    callbacks = utils.instantiate_callbacks(cfg.get("callbacks"))
    logger = utils.instantiate_loggers(cfg.get("logger"))

    trainer_cfg = {k: v for k, v in cfg.trainer.items() if k != "_target_"}
    trainer = L.Trainer(**trainer_cfg, callbacks=callbacks, logger=logger)

    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    test_metrics = trainer.callback_metrics
    return test_metrics, {"cfg": cfg, "datamodule": datamodule, "model": model, "trainer": trainer}


@hydra.main(config_path="configs", config_name="test", version_base=None)
def main(cfg: DictConfig):
    assert cfg.ckpt_path, "ckpt_path parameters must be provided!"
    run_test(cfg)


if __name__ == "__main__":
    main()
