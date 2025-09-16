import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

from deepfusion.models.blocksv2 import ConvBlock3D, Downsample3D, Upsample3D
from deepfusion.utils.losses import masked_mae
from deepfusion.utils.evaluation import plot_slices

class Encoder(nn.Module):
    """
    (B, in_ch, 128,128,128) -> (B, d_enc)
    Channels: in_ch→16→32→64→128→256, then head to d_enc.
    """
    def __init__(self, in_ch=1, base=16, downsample: str = "learned", dropout: bool = True):
        super().__init__()
        self.conv1 = ConvBlock3D(in_ch, base)                                       # (16, 128³)
        self.down1 = Downsample3D(base, base, downsample)                           # (16, 64³)
        self.conv2 = ConvBlock3D(base, base*2)                                      # (32, 64³)
        self.down2 = Downsample3D(base*2, base*2, downsample)                       # (32, 32³)
        self.conv3 = ConvBlock3D(base*2, base*4, dropout=0.2 if dropout else 0.0)   # (64, 32³)
        self.down3 = Downsample3D(base*4, base*4, downsample)                       # (64, 16³)
        self.conv4 = ConvBlock3D(base*4, base*8, dropout=0.2 if dropout else 0.0)   # (128, 16³)
        self.down4 = Downsample3D(base*8, base*8, downsample)                       # (128, 8³)
        self.conv5 = ConvBlock3D(base*8, base*16, dropout=0.3 if dropout else 0.0)  # (256, 8³)
        self.down5 = Downsample3D(base*16, base*16, downsample)                     # (256, 4³)
        self.conv6 = ConvBlock3D(base*16, base*32, dropout=0.3 if dropout else 0.0) # (256, 4³)


    def forward(self, x):                               # x: [B, 1, 128,128,128]
        x = self.conv1(x)                               # [B, base, 128,128,128]
        x = self.down1(x)                               # [B, base*2, 64,64,64]
        x = self.conv2(x)                               # [B, base*2, 64,64,64]
        x = self.down2(x)                               # [B, base*4, 32,32,32]
        x = self.conv3(x)                               # [B, base*4, 32,32,32]
        x = self.down3(x)                               # [B, base*8, 16,16,16]
        x = self.conv4(x)                               # [B, base*8, 16,16,16]
        x = self.down4(x)                               # [B, base*16, 8,8,8]
        x = self.conv5(x)                               # [B, base*16, 8,8,8]
        x = self.down5(x)                               # [B, base*16, 4,4,4]
        x = self.conv6(x)                               # [B, base*16, 4,4,4]                           
        return x                                         

class Decoder(nn.Module):
    """
    (B, d_enc) -> (B, out_ch, 128,128,128)
    Mirrors encoder channels: 256→128→64→32→16→8, then head to out_ch.
    """
    def __init__(self, out_ch=1, base=16, upsample: str = "learned"):
        super().__init__()
        self.conv1 = ConvBlock3D(base*32, base*16)                  # (256, 4³)
        self.up1   = Upsample3D(base*16, base*16, upsample)         # (256, 8³)
        self.conv2 = ConvBlock3D(base*16, base*8)                   # (128, 8³)
        self.up2   = Upsample3D(base*8, base*8, upsample)           # (128, 16³)
        self.conv3 = ConvBlock3D(base*8, base*4)                    # (64, 16³)
        self.up3   = Upsample3D(base*4, base*4, upsample)           # (64, 32³)
        self.conv4 = ConvBlock3D(base*4, base*2)                    # (32, 32³)
        self.up4   = Upsample3D(base*2, base*2, upsample)           # (32, 64³)
        self.conv5 = ConvBlock3D(base*2, base)                      # (16, 64³)
        self.up5   = Upsample3D(base, base, upsample)               # (16, 128³)
        self.head  = nn.Conv3d(base, out_ch, kernel_size=3, padding=1)   # (out_ch, 128³)

    def forward(self, z):                   # z: [B, base*16, 4, 4, 4]
        z = self.conv1(z)                   # [B, 256, 4, 4, 4]
        z = self.up1(z)                     # [B, 256, 8, 8, 8]
        z = self.conv2(z)                   # [B, 128, 8, 8, 8]
        z = self.up2(z)                     # [B, 128, 16, 16, 16]
        z = self.conv3(z)                   # [B, 64, 16, 16, 16]
        z = self.up3(z)                     # [B, 64, 32, 32, 32]
        z = self.conv4(z)                   # [B, 32, 32, 32, 32]
        z = self.up4(z)                     # [B, 32, 64, 64, 64]
        z = self.conv5(z)                   # [B, 16, 64, 64, 64]
        z = self.up5(z)                     # [B, 16, 128, 128, 128]
        x_pred = self.head(z)
        return x_pred                             # [B, 1, 128,128,128]
    
    
class AutoEncoder3D(pl.LightningModule):
    def __init__(self, in_ch=1, base=8, downsample: str = "max", upsample: str = "trilinear", dropout: bool = True, lr: float = 1e-3, weight_decay: float = 1e-3):
        """
        """
        super().__init__()
        self.encoder = Encoder(in_ch=in_ch, base=base, downsample=downsample, dropout=dropout)
        self.decoder = Decoder(out_ch=in_ch, base=base, upsample=upsample)
        self.lr = lr
        self.weight_decay = weight_decay
        self.mean_img = np.load("data/deepfusion/volumes/test/mean_image_testset.npy")  # shape (D,H,W)
        self.save_hyperparameters()

    def forward(self, x):                           # x: [B, 1, 128,128,128]
        z = self.encoder(x)                         # [B, 256, 4,4,4]
        x_pred = self.decoder(z)                    # [B, 1, 128,128,128]
        return x_pred
    
    def training_step(self, batch, batch_idx):
        x, mask, patient_ids, session_ids = batch                                   # [B,1,128,128,128]
        x_hat = self(x) * mask                        # [B,1,128,128,128]  (zero background)
        loss = masked_mae(x_hat, x, mask)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, patient_ids, session_ids = batch
        x_hat = self(x)
        loss = masked_mae(x_hat, x, mask)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        x, mask, patient_ids, session_ids = batch

        x_hat = self(x)
        zero_pred = torch.zeros_like(x_hat)
        mean_pred = torch.from_numpy(self.mean_img)[None, None, ...].to(device=x.device, dtype=x.dtype).expand_as(x_hat)

        # masked MAE for model, zero, mean
        loss_zero = masked_mae(zero_pred, x, mask)
        loss_mean = masked_mae(mean_pred, x, mask)  
        loss_model = masked_mae(x_hat, x, mask)

        self.log("mae_model", loss_model, prog_bar=True, on_step=False, on_epoch=True)
        self.log("mae_zero", loss_zero, prog_bar=False, on_step=False, on_epoch=True)
        self.log("mae_mean", loss_mean, prog_bar=False, on_step=False, on_epoch=True)


    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        if batch_idx < 5: # log first 5 samples
            x, mask, patient_ids, session_ids = batch
            x_hat = self(x)
            mean_img = torch.from_numpy(self.mean_img)[None, None, ...].to(device=x.device, dtype=x.dtype).expand_as(x_hat)

            images = plot_slices(x[0], x_hat[0], mean_img[0], mask[0]) # [5, 1, H, W]
            self.logger.experiment.add_images(f"sample_{batch_idx}", images, global_step=self.global_step, dataformats="NCHW")


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

        optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=4,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# # test forward pass
# if __name__ == "__main__":
#     model = Autoencoder3D(in_ch=1, base=16, downsample="learned", upsample="learned")
#     x = torch.randn(50, 1, 128, 128, 128)  # batch of 2 samples
#     x_pred = model(x)
#     print(x_pred.shape)  # should be [2, 1, 128, 128, 128]
