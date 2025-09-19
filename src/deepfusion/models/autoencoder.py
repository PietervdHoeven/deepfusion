import torch
import torch.nn as nn
import pytorch_lightning as pl

from deepfusion.utils.losses import masked_l1, masked_l2, weighted_l1, weighted_l2
from deepfusion.utils.visualisation import plot_recons

def conv3x3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def up3x3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    )

def proj1x1x1(in_channels, out_channels, sample: str | None = None):
    if sample == "down":
        return nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )
    elif sample == "up":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )
    else:
        return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sample: str | None = None, residual=True):
        super().__init__()
        self.residual = residual
        self.sample = sample
        if sample == "down":
            self.conv1 = conv3x3x3(in_channels, out_channels, stride=2)
        elif sample == "up":
            self.conv1 = up3x3x3(in_channels, out_channels)
        else:
            self.conv1 = conv3x3x3(in_channels, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3x3(out_channels, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        if (sample != None or in_channels != out_channels) and residual:
            self.proj = proj1x1x1(in_channels, out_channels, sample=sample)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x):
        y = self.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.residual:
            x = self.proj(x)
            y += x
        y = self.gelu(y)
        return y


class Encoder(nn.Module):
    def __init__(
            self, 
            in_ch=1,
            channels=[16, 32, 64, 128, 192, 384, 512],
            residual=True,
            ):
        super().__init__()
        self.in_ch = in_ch
        self.channels = channels
        self.residual = residual

        # Initial convolution
        self.conv_in = conv3x3x3(in_ch, channels[0])    # [B, C_0, 128, 128, 128]
        self.norm_in = nn.GroupNorm(8, channels[0])
        self.gelu = nn.GELU()

        # Downsampling layers
        self.layer1 = ConvBlock(channels[0], channels[1], residual=self.residual, sample="down")  # [B, C_1, 64, 64, 64]
        self.layer2 = ConvBlock(channels[1], channels[2], residual=self.residual, sample="down")  # [B, C_2, 32, 32, 32]
        self.layer3 = ConvBlock(channels[2], channels[3], residual=self.residual, sample="down")  # [B, C_3, 16, 16, 16]
        self.layer4 = ConvBlock(channels[3], channels[4], residual=self.residual, sample="down")  # [B, C_4, 8, 8, 8]
        self.layer5 = ConvBlock(channels[4], channels[5], residual=self.residual, sample="down")  # [B, C_5, 4, 4, 4]
        self.layer6 = ConvBlock(channels[5], channels[6], residual=self.residual, sample="down")  # [B, C_6, 2, 2, 2]
        self.dropout = nn.Dropout3d(p=0.1)

    def forward(self, x):
        x = self.gelu(self.norm_in(self.conv_in(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.dropout(x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            out_ch=1,
            channels=[16, 32, 64, 128, 192, 384, 512],
            residual=True,
            ):
        super().__init__()
        self.out_ch = out_ch
        self.channels = channels
        self.residual = residual

        # Upsampling layers
        self.layer1 = ConvBlock(channels[6], channels[5], residual=self.residual, sample="up")  # [B, C_6, 4, 4, 4]
        self.layer2 = ConvBlock(channels[5], channels[4], residual=self.residual, sample="up")  # [B, C_5, 8, 8, 8]
        self.layer3 = ConvBlock(channels[4], channels[3], residual=self.residual, sample="up")  # [B, C_4, 16, 16, 16]
        self.layer4 = ConvBlock(channels[3], channels[2], residual=self.residual, sample="up")  # [B, C_3, 32, 32, 32]
        self.layer5 = ConvBlock(channels[2], channels[1], residual=self.residual, sample="up")  # [B, C_2, 64, 64, 64]
        self.layer6 = ConvBlock(channels[1], channels[0], residual=self.residual, sample="up")  # [B, C_1, 128, 128, 128]

        # finalising convolution
        self.conv_out = nn.Conv3d(channels[0], out_ch, kernel_size=3, padding=1)  # [B, out_ch, 128, 128, 128]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.conv_out(x)
        return x


class Autoencoder(pl.LightningModule):
    def __init__(
            self,
            in_ch=1,
            out_ch=1,
            channels=[16, 32, 64, 128, 192, 384, 512],
            residual=True,
            lr=1e-3,
            weight_decay=1e-5,
            ):
        super().__init__()
        # training hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        # model
        self.encoder = Encoder(in_ch=in_ch, channels=channels, residual=residual)
        self.decoder = Decoder(out_ch=out_ch, channels=channels, residual=residual)

    def forward(self, x):                           # x: [B, 1, 128,128,128]
        z = self.encoder(x)                         # [B, 512, 2,2,2]
        x_pred = self.decoder(z)                    # [B, 1, 128,128,128]
        return x_pred
    
    def training_step(self, batch, batch_idx):
        x, mask, patient_ids, session_ids = batch
        x_hat = self(x)
        train_mse = weighted_l2(x_hat, x, mask)  # Standard L2 loss
        train_mae = weighted_l1(x_hat, x, mask)              # Masked MAE
        self.log("train_mse", train_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_mae", train_mae, prog_bar=True, on_step=True, on_epoch=True)
        self.log("lr", self.lr, prog_bar=True, on_step=False, on_epoch=True)
        return train_mse

    def validation_step(self, batch, batch_idx):
        x, mask, patient_ids, session_ids = batch
        x_hat = self(x)
        val_mse = weighted_l2(x_hat, x, mask)  # Standard L2 loss with weighted mask
        val_mae = weighted_l1(x_hat, x, mask)              # Masked MAE
        self.log("val_mse", val_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", val_mae, prog_bar=True, on_step=True, on_epoch=True)

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
            patience=3,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_mse",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
# if __name__ == "__main__":
#     model = AutoEncoder(in_ch=1, out_ch=1)
#     x = torch.randn(2, 1, 128, 128, 128)  # Example input tensor with batch size 2
#     x_recon = model(x)
#     print(x_recon.shape)  # Should output: torch.Size([2, 1, 128, 128, 128])