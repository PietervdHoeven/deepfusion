# debug.py
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# swap these lines to test different modules
from deepfusion.models.autoencoder import AE5D, Autoencoder, AE7D
from deepfusion.datamodules.ae_datamodule import AEDataModule
from deepfusion.utils.losses import weighted_l1, weighted_l2, masked_l1, masked_l2, weighted_charbonnier, masked_charbonnier
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader
#dl = DataLoader()

def main():
    pl.seed_everything(42)

    # init datamodule + model
    datamodule = AEDataModule(
        data_dir="data",
        use_sampler=True,
        batch_size=2,
        num_workers=14,
        pin_memory=True,
        prefetch_factor=4
    )
    # optional channels:
    # lite:     [32, 48, 64, 80, 112, 160, 224, 320]
    # large:    [32, 64, 96, 128, 160, 224, 320, 448]

    model_params = {
        "in_channels": 1,
        "channels": [32, 64, 128, 256, 384, 512],
        "residual": True,
        "depthwise": True
    }

    backbone = AE5D()

    training_params = {
        "masked_pretraining": False,
        "loss_fn": MSELoss(),
        "lr": 1e-3,
        "weight_decay": 1e-5,
        #"betas": (0.9, 0.95),
    }
    model = Autoencoder(model=backbone, **training_params)
    
    # logger (logs to logs/debug/version_x)
    logger = TensorBoardLogger(
        save_dir="logs", name="AutoEncoder", 
        version=f"ch-{model_params['channels']}_res-{model_params['residual']}_dw-{model_params['depthwise']}_mask-{training_params['masked_pretraining']}_loss-{training_params['loss_fn'].__class__.__name__}")
    
    # Checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_mse",
        dirpath=logger.log_dir,
        filename="epoch{epoch:02d}-val_loss{val_mse:.4f}-checkpoint",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_mse",
        patience=12,
        mode="min",
        min_delta=0.0005,
    )

    # trainer (no checkpoint, no early stopping)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",       # change to "16-mixed" if you want AMP
        log_every_n_steps=10,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=1000,
        accumulate_grad_batches=64,  # simulate larger batch size
        gradient_clip_val=1.0,
        overfit_batches=0        # for debugging, set to 0 or remove for full training

    )

    # fit loop
    ckpt_path = "/home/spieterman/projects/deepfusion/logs/AutoEncoder/ch-[32, 64, 128, 256, 384, 512]_res-True_dw-True_mask-False_loss-MSELoss/last.ckpt"
    # ckpt_path = None
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)  # or "path/to/checkpoint.ckpt"

    # optional test after training
    # trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()