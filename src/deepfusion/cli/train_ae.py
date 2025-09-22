# debug.py
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# swap these lines to test different modules
from deepfusion.models.autoencoder import Autoencoder, AE7D
from deepfusion.datamodules.ae_datamodule import AEDataModule
from deepfusion.utils.losses import weighted_l1, weighted_l2, masked_l1, masked_l2, weighted_charbonnier, masked_charbonnier
from torch.nn import L1Loss, MSELoss



def main():
    pl.seed_everything(42)

    # init datamodule + model
    datamodule = AEDataModule(
        data_dir="data",
        use_sampler=True,
        batch_size=1,
        num_workers=10,
        pin_memory=True,
        prefetch_factor=2
    )
    # optional channels:
    # lite:     [32, 48, 64, 80, 112, 160, 224, 320]
    # large:    [32, 64, 96, 128, 160, 224, 320, 448]

    model_params = {
        "in_channels": 1,
        "channels": [24, 32, 48, 64, 96, 160, 256, 384],
        "residual": True
    }

    backbone = AE7D(**model_params)

    learning_params = {
        "masked_pretraining": False,
        "loss_fn": weighted_charbonnier,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "betas": (0.9, 0.95),
    }
    model = Autoencoder(model=backbone, **learning_params)
    
    # logger (logs to logs/debug/version_x)
    logger = TensorBoardLogger(
        save_dir="logs", name="AutoEncoder", 
        version=f"{model_params['channels']}_{model_params['residual']}")
    
    # Checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=logger.log_dir,
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )

    # trainer (no checkpoint, no early stopping)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",       # change to "16-mixed" if you want AMP
        log_every_n_steps=10,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        max_epochs=-1,
        accumulate_grad_batches=16,  # simulate larger batch size
        gradient_clip_val=1.0,
        overfit_batches=0,        # for debugging, set to 0 or remove for full training
    )

    # fit loop
    ckpt_path = "/home/spieterman/projects/deepfusion/logs/AutoEncoder/[24, 32, 48, 64, 96, 160, 256, 384]_True/best-checkpoint.ckpt"
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)  # or "path/to/checkpoint.ckpt"

    # optional test after training
    # trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()