# src/cli/train.py
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.profilers import AdvancedProfiler
import torch
# torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """
    Minimal training entrypoint:
      - Seeds everything
      - Instantiates logger, callbacks, datamodule, model, and trainer from Hydra config
      - Fits model
      - Optionally tests on best checkpoint (if cfg.test=true)

    Expected config keys (all instantiable via Hydra):
      cfg.logger          -> Lightning logger (e.g., TensorBoard/W&B)
      cfg.early_stopping  -> EarlyStopping callback
      cfg.checkpoint      -> ModelCheckpoint callback
      cfg.datamodule      -> LightningDataModule
      cfg.model           -> LightningModule
      cfg.trainer         -> Trainer kwargs
      cfg.seed            -> int
      cfg.test            -> bool (optional): run test after fit on best ckpt
    """
    print(OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed), workers=True)

    # Instantiate components
    logger = instantiate(cfg.logger)                        # e.g., TensorBoardLogger
    early_stopping = instantiate(cfg.early_stopping)        # EarlyStopping
    checkpoint_cb = instantiate(cfg.checkpoint)             # ModelCheckpoint
    callbacks = [early_stopping, checkpoint_cb]
    # profiler = AdvancedProfiler(dirpath="profiling", filename="train_profiler.txt")

    datamodule = instantiate(cfg.datamodule)                # LightningDataModule
    model = instantiate(cfg.model)                          # LightningModule

    # Single Trainer instantiation from config
    trainer: Trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks, profiler="simple")

    # Fit
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
