import pytorch_lightning as pl
import torch
import torch.nn as nn
from deepfusion.utils.losses import masked_l1, masked_l2, weighted_l1, weighted_l2
from deepfusion.utils.visualisation import plot_recons
from deepfusion.utils.masking import patch_mask
from torchmetrics import StructuralSimilarityIndexMeasure
import os


def proj1x1x1(in_channels, out_channels, sample: str | None = None):
    if sample == "down":
        # AvgPool → 1x1x1 conv (padding=0)
        return nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )
    elif sample == "up":
        # Upsample → 1x1x1 conv (padding=0)
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )
    else:
        return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

def conv3x3x3(in_channels, out_channels, type = "normal", sample: str | None = None, bias=False):
    if type == "normal":
        return nrml_conv3x3x3(in_channels, out_channels, sample=sample, bias=bias)
    elif type == "depthwise":
        return dw_conv3x3x3(in_channels, out_channels, sample=sample, bias=bias)
    else:
        return grpd_conv3x3x3(in_channels, out_channels, sample=sample, bias=bias)

def nrml_conv3x3x3(in_channels, out_channels, sample: str | None = None, bias=False):
    if sample == "down":
        return nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        )
    elif sample == "up":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        )
    else:
        return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

def dw_conv3x3x3(in_channels, out_channels, sample: str | None = None, bias=False):
    if sample == "down":
        return nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=bias),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        )
    elif sample == "up":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=bias),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        )
    else:
        return nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=bias),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        )
    
def grpd_conv3x3x3(in_channels, out_channels, sample: str | None = None, bias=False, groups=8):
    if sample == "down":
        return nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=bias)
        )
    elif sample == "up":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=bias)
        )
    else:
        return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sample: str | None = None, residual=True, type="normal"):
        super().__init__()
        self.residual = residual
        self.conv1 = conv3x3x3(in_channels, out_channels, type=type, sample=sample)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3x3(out_channels, out_channels, type=type)
        self.norm2 = nn.GroupNorm(8, out_channels)
        if (sample != None or in_channels != out_channels) and residual:
            self.proj = proj1x1x1(in_channels, out_channels, sample=sample)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.residual:
            x = self.proj(x)
            y += x
        y = self.relu(y)
        return y

class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_fn=masked_l1,
        lr=1e-4,
        weight_decay=0,
        betas=(0.9, 0.999),
        masked_pretraining=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.masked_pretraining = masked_pretraining
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        if self.masked_pretraining:
            x, patch_mask_tensor = patch_mask(x, mask_ratio=0.5, ps=8)
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log(f"train_mse", loss, prog_bar=True, on_step=False, on_epoch=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)

        if batch_idx < 4:
            save_dir = f"logs/{self.logger.name}/{self.logger.version}/plots"
            os.makedirs(save_dir, exist_ok=True)
            plot_recons(x, x_hat, fname=f"{save_dir}/train_reconstructions_epoch{self.current_epoch}_batch{batch_idx}.png")
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        mae = torch.nn.functional.l1_loss(x_hat, x)


        self.log("val_mse", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", mae, prog_bar=True, on_step=False, on_epoch=True)


        if batch_idx < 4:
            save_dir = f"logs/{self.logger.name}/{self.logger.version}/plots"
            os.makedirs(save_dir, exist_ok=True)
            plot_recons(x, x_hat, fname=f"{save_dir}/val_reconstructions_epoch{self.current_epoch}_batch{batch_idx}.png")
    
    def test_step(self, batch, batch_idx):
        x, mask, patient_ids, session_ids = batch
        x_hat = self(x)
        w_mae = weighted_l1(x_hat, x, mask)
        w_mse = weighted_l2(x_hat, x, mask)
        m_mae = masked_l1(x_hat, x, mask)
        m_mse = masked_l2(x_hat, x, mask)
        mae = torch.nn.functional.l1_loss(x_hat, x)
        mse = torch.nn.functional.mse_loss(x_hat, x)

        ssim = self.ssim(x, x_hat)

        self.log("test_w_mae", w_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_w_mse", w_mse, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test_m_mae", m_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_m_mse", m_mse, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_mse", mse, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test_ssim", ssim, prog_bar=True, on_step=False, on_epoch=True)

        if batch_idx == 0:
            save_dir = f"logs/{self.logger.name}/version_{self.logger.version}/plots"
            os.makedirs(save_dir, exist_ok=True)
            plot_recons(x, x_hat, fname=f"{save_dir}/test_reconstructions_epoch{self.current_epoch}.png")


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=self.weight_decay, lr=self.lr, betas=self.betas)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            min_lr=1e-6,
            threshold=0.001,
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "monitor": "val_mse",   # Lightning needs this for ReduceLROnPlateau
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    

class AE7D(nn.Module):  # 7 downsamples
    def __init__(self, in_channels=1, channels=[32, 64, 128, 256, 512, 1024, 2048, 4096], residual=True, depthwise=False):
        super().__init__()
        self.initial_conv = nn.Sequential(
            dw_conv3x3x3(in_channels, channels[0]),
            nn.GroupNorm(8, channels[0]),
            nn.LeakyReLU(0.1),
        )
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, channels[0], sample="down", residual=residual, depthwise=depthwise),
            ConvBlock(channels[0], channels[1], sample="down", residual=residual, depthwise=depthwise),
            ConvBlock(channels[1], channels[2], sample="down", residual=residual, depthwise=depthwise),
            ConvBlock(channels[2], channels[3], sample="down", residual=residual, depthwise=depthwise),
            ConvBlock(channels[3], channels[4], sample="down", residual=residual, depthwise=depthwise),
            ConvBlock(channels[4], channels[5], sample="down", residual=residual, depthwise=depthwise),
            ConvBlock(channels[5], channels[6], sample="down", residual=residual, depthwise=depthwise),
            ConvBlock(channels[6], channels[7], sample="down", residual=residual, depthwise=depthwise),
        )

        self.decoder = nn.Sequential(
            ConvBlock(channels[7], channels[6], sample="up", residual=residual, depthwise=depthwise),
            ConvBlock(channels[6], channels[5], sample="up", residual=residual, depthwise=depthwise),
            ConvBlock(channels[5], channels[4], sample="up", residual=residual, depthwise=depthwise),
            ConvBlock(channels[4], channels[3], sample="up", residual=residual, depthwise=depthwise),
            ConvBlock(channels[3], channels[2], sample="up", residual=residual, depthwise=depthwise),
            ConvBlock(channels[2], channels[1], sample="up", residual=residual, depthwise=depthwise),
            ConvBlock(channels[1], channels[0], sample="up", residual=residual, depthwise=depthwise),
        )

        self.final_conv = dw_conv3x3x3(channels[0], in_channels, bias=True)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return self.final_conv(x)
    
class AE5D(nn.Module):
    def __init__(self, in_channels=1, channels=[16, 32, 64, 128, 256, 384], residual=True, depthwise=False):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, channels[0]),
            nn.LeakyReLU(0.1),
        )

        self.encoder = nn.Sequential(
            ConvBlock(channels[0], channels[1], sample="down", residual=residual, type="normal"),
            ConvBlock(channels[1], channels[2], sample="down", residual=residual, type="normal"),
            ConvBlock(channels[2], channels[3], sample="down", residual=residual, type="normal"),
            ConvBlock(channels[3], channels[4], sample="down", residual=residual, type="depthwise"),
            ConvBlock(channels[4], channels[5], sample="down", residual=residual, type="depthwise"),
        )

        self.decoder = nn.Sequential(
            ConvBlock(channels[5], channels[4], sample="up", residual=residual, type="depthwise"),
            ConvBlock(channels[4], channels[3], sample="up", residual=residual, type="depthwise"),
            ConvBlock(channels[3], channels[2], sample="up", residual=residual, type="normal"),
            ConvBlock(channels[2], channels[1], sample="up", residual=residual, type="normal"),
            ConvBlock(channels[1], channels[0], sample="up", residual=residual, type="normal"),
        )

        self.head = nn.Conv3d(channels[0], 1, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return self.head(x)