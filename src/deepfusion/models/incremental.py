from deepfusion.models.autoencoder import ConvBlock, conv3x3x3, proj1x1x1, up3x3x3
import pytorch_lightning as pl
import torch
import torch.nn as nn
from deepfusion.utils.losses import masked_l1, masked_l2, weighted_l1, weighted_l2
from deepfusion.utils.visualisation import plot_recons

class Autoencoder_0(pl.LightningModule):
    def __init__(self, in_channels=1, latent_dim=128, residual=False, loss_fn=weighted_l1, lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.999), warmup_epochs=5):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.residual = residual


        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_epochs = warmup_epochs

        # Encoder
        self.enc1 = ConvBlock(in_channels, 16, sample="down", residual=residual)  # [B, 16, 64, 64, 64]
        # Decoder
        self.dec1 = ConvBlock(16, in_channels, sample="up", residual=residual)      # [B, 64, 64, 64, 64]


    def forward(self, x):
        # Encoder
        x = self.enc1(x)

        # Decoder
        x = self.dec1(x)

        return x
    
    def _step(self, batch, batch_idx, stage="train"):
        x, mask, patient_ids, session_ids = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x, mask)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss, x, x_hat
    
    def training_step(self, batch, batch_idx):
        loss, x, x_hat = self._step(batch, batch_idx, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, x, x_hat = self._step(batch, batch_idx, stage="val")

        mse = weighted_l2(x_hat, x, batch[1])  # Standard L2 loss with weighted mask
        self.log("val_mse", mse, prog_bar=True, on_step=False, on_epoch=True)

        if batch_idx == 0:
            plot_recons(x, x_hat, fname=f"val_reconstructions_epoch{self.current_epoch}.png")


    def configure_optimizers(self):
        # split params into decay / no-decay
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad: 
                continue
            if p.ndim == 1 or name.endswith(".bias"):
                no_decay.append(p)   # biases & norm (1D) → no decay
            else:
                decay.append(p)      # conv/linear weights → decay

        param_groups = [
            {"params": decay,    "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=self.betas)

        # --- 1) Warmup schedule (linear from 0 -> base LR in N epochs)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8 / self.lr,  # basically "start at ~0"
            end_factor=1.0,
            total_iters=self.warmup_epochs,  # number of warmup epochs
        )

        # --- 2) Your plateau scheduler
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=3,
            min_lr=1e-6,
        )

        # --- 3) Sequential wrapper
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, plateau_scheduler],
            milestones=[self.warmup_epochs],  # switch from warmup -> plateau after self.warmup_epochs epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",   # Lightning needs this for ReduceLROnPlateau
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
