import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, TensorDataset


class Vanilla3DCNN(pl.LightningModule):
    def __init__(self, num_classes=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Conventional 3D CNN backbone
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),   # input: [B,1,128,128,128]
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # [B,16,64,64,64]

            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # [B,32,32,32,32]

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # [B,64,16,16,16]

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # [B,128,8,8,8]
        )

        # Global average pooling + linear classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # [B,128,1,1,1]
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)