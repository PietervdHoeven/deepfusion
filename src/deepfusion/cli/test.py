# src/cli/test.py
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import Trainer, seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="test")
def main(cfg: DictConfig) -> None:
    """
    Minimal testing entrypoint:
      - Seeds
      - Instantiates logger (optional), datamodule, model, and trainer via Hydra
      - Runs trainer.test() with optional ckpt_path

    Expected config keys (instantiable via Hydra):
      cfg.logger      -> (optional) Lightning logger
      cfg.datamodule  -> LightningDataModule
      cfg.model       -> LightningModule
      cfg.trainer     -> Trainer kwargs
      cfg.seed        -> int
      cfg.ckpt_path   -> (optional) path to checkpoint for evaluation
    """
    print(OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed), workers=True)

    # Optional logger (allow omission in config)
    logger = instantiate(cfg.logger)

    # Core components
    datamodule = instantiate(cfg.datamodule)   # -> LightningDataModule
    model = instantiate(cfg.model)             # -> LightningModule
    trainer: Trainer = instantiate(cfg.trainer, logger=logger)

    # Test (Hydra config can pass ckpt_path=None or a path)
    trainer.test(model=model, dataloaders=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()