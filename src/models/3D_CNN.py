import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchmetrics import AUROC


class Vanilla3DCNN(pl.LightningModule):
    def __init__(self, task: str = "tri_cdr", lr=1e-3):
        super().__init__()
        self.num_classes = 3 if task == "tri_cdr" else 2
        self.save_hyperparameters()

        # Conventional 3D CNN backbone
        self.features = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, stride=1, padding=1),   # input: [B,4,128,128,128]
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

            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(64, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # [B,256,4,4,4]
        )

        # Global average pooling + linear classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # [B,256,1,1,1]
            nn.Flatten(),
            nn.Linear(256, self.num_classes)
        )

        # metrics
        if self.num_classes == 2:
            # binary AUROC
            self.train_auroc = AUROC(task="binary")
            self.val_auroc   = AUROC(task="binary")
        else:
            # multiclass Macro-AUROC (OvR)
            self.train_auroc = AUROC(task="multiclass", num_classes=self.num_classes, average="macro")
            self.val_auroc   = AUROC(task="multiclass", num_classes=self.num_classes, average="macro")


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _step(self, batch, stage: str):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        auroc = self.train_auroc if stage == "train" else self.val_auroc
        log_args = dict(on_step=False, on_epoch=True, prog_bar=True)

        auroc.update(logits, y)
        self.log(f"{stage}_loss", loss, **log_args)
        self.log(f"{stage}_acc", acc, **log_args)
        self.log(f"{stage}_auroc", auroc, **log_args)

        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")
    
    def validation_step(self, batch, batch_idx):
        self._step(batch, stage="val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)