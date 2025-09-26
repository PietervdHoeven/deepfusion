import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torchmetrics import MeanAbsoluteError, R2Score, ConfusionMatrix, MetricCollection
from torchmetrics.classification import AUROC, Recall  # macro recall == balanced accuracy

# use the shared ConvBlock for simple baselines
from .blocks import ConvBlock


class Baseline(pl.LightningModule):
    """
    Minimal Lightning module for: 'age
    Subclasses implement: forward(x) -> [B, feat_dim]   (already pooled)
    """
    def __init__(
            self, 
            task: str, 
            feat_dim: int, 
            lr: float = 1e-4, 
            weight_decay: float = 1e-4,
            patience: int = 10,
            ):
        super().__init__()
        self.task = task  # "bin_cdr", "tri_cdr", "gender", "handedness", "age"
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        out_dim = 1 if task != "tri_cdr" else 3

        self.head = nn.Linear(feat_dim, out_dim)

        if task == "age":
            self.metrics = nn.ModuleDict({
                "train_metrics": MetricCollection({"mae": MeanAbsoluteError()}),
                "val_metrics":   MetricCollection({"mae": MeanAbsoluteError(), "r2": R2Score()}),
                "test_metrics":  MetricCollection({"mae": MeanAbsoluteError(), "r2": R2Score()}),
            })
        elif task in {"bin_cdr", "gender", "handedness"}:
            self.metrics = nn.ModuleDict({
                "train_metrics": MetricCollection({"auroc": AUROC(task="binary")}),
                "val_metrics":   MetricCollection({"auroc": AUROC(task="binary"),
                          "balacc": Recall(task="multiclass", num_classes=2, average="macro")}),
                "test_metrics":  MetricCollection({"auroc": AUROC(task="binary"),
                          "balacc": Recall(task="multiclass", num_classes=2, average="macro")}),
            })
        else:  # multiclass (tri_cdr)
            self.metrics = nn.ModuleDict({
                "train_metrics": MetricCollection({"auroc": AUROC(task="multiclass", num_classes=3, average="macro")}),
                "val_metrics":   MetricCollection({"auroc": AUROC(task="multiclass", num_classes=3, average="macro"),
                          "balacc": Recall(task="multiclass", num_classes=3, average="macro")}),
                "test_metrics":  MetricCollection({"auroc": AUROC(task="multiclass", num_classes=3, average="macro"),
                          "balacc": Recall(task="multiclass", num_classes=3, average="macro")}),
            })

        # buffers for test-time confusion matrix
        self._test_preds, self._test_targets = [], []

    # --- internal ---
    def _logits(self, x):
        return self.head(self(x))  # subclass forward -> features; then linear head

    def _step(self, batch, stage):
        x, y = batch

        if self.task == "age":
            pred = self._logits(x).squeeze(-1).float()
            y = y.float()
            loss = F.mse_loss(pred, y)
            m = self.metrics[f"{stage}_metrics"]
            m["mae"].update(pred, y)
            if "r2" in m: m["r2"].update(pred, y)
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
            return loss

        if self.task in {"bin_cdr", "gender", "handedness"}:
            logits = self._logits(x).squeeze(-1)
            probs  = torch.sigmoid(logits)
            loss   = F.binary_cross_entropy_with_logits(logits, y.float())
            m = self.metrics[f"{stage}_metrics"]
            m["auroc"].update(probs, y.int())
            if stage != "train":
                preds = (probs > 0.5).long()
                m["balacc"].update(preds, y.long())
            if stage == "test":
                preds = (probs > 0.5).long()
                self._test_preds.append(preds.cpu())
                self._test_targets.append(y.long().cpu())
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
            return loss

        # multiclass (3)
        logits = self._logits(x)                    # [B, 3]
        probs  = torch.softmax(logits, dim=1)
        loss   = F.cross_entropy(logits, y.long())
        m = self.metrics[f"{stage}_metrics"]
        m["auroc"].update(probs, y.long())
        if stage != "train":
            preds = torch.argmax(probs, dim=1)
            m["balacc"].update(preds, y.long())
        if stage == "test":
            preds = torch.argmax(probs, dim=1)
            self._test_preds.append(preds.cpu())
            self._test_targets.append(y.long().cpu())
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    # --- lightning hooks ---
    def training_step(self, batch, _):  
        return self._step(batch, "train")
    def validation_step(self, batch, _): 
        self._step(batch, "val")
    def test_step(self, batch, _):       
        self._step(batch, "test")

    def _finish(self, stage):
        for name, metric in self.metrics[f"{stage}_metrics"].items():
            self.log(f"{stage}_{name}", metric.compute(), prog_bar=(stage != "test"))
            metric.reset()

    def on_train_epoch_end(self):      
        self._finish("train")
    def on_validation_epoch_end(self): 
        self._finish("val")

    def on_test_start(self):
        self._test_preds.clear()
        self._test_targets.clear()

    def on_test_epoch_end(self):
        self._finish("test")
        if self.task in {"bin_cdr", "tri_cdr", "gender", "handedness"} and self._test_preds:
            preds = torch.cat(self._test_preds, dim=0)
            targets = torch.cat(self._test_targets, dim=0)
            num_classes = 3 if self.task == "tri_cdr" else 2
            cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)(preds, targets).cpu().numpy()
            self._save_confmat_image(cm, num_classes)
            self._test_preds.clear()
            self._test_targets.clear()

    def _save_confmat_image(self, cm, num_classes: int):
        log_dir = getattr(self.logger, "log_dir", None)
        if log_dir is None:
            return
        out_dir = os.path.join(log_dir, "confusion_matrices")
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation="nearest")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=range(num_classes),
            yticks=range(num_classes),
            xlabel="Predicted label",
            ylabel="True label",
            title=f"Confusion Matrix (epoch {self.current_epoch})",
        )
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        fig.tight_layout()

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = f"cm_epoch{self.current_epoch}_{self.task}_{num_classes}c_{ts}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close(fig)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=self.patience, min_lr=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}




# ---- Example encoder that plugs into Baseline and returns [B, feat_dim] ----
class EncoderOnly(Baseline):
    def __init__(
            self, 
            task="multiclass", 
            in_ch=4, 
            channels=(16,32,64,128,256,384), 
            residual=True, 
            lr=1e-4, 
            weight_decay=1e-4
            ):
        super().__init__(task=task, feat_dim=channels[-1], lr=lr, weight_decay=weight_decay)

        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, channels[0], 3, 1, 1, bias=False),
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
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        x = self.gap(x).flatten(1)   # -> [B, feat_dim]
        return x


class ResNet10(Baseline):
    def __init__(
            self, 
            task="multiclass", 
            in_ch=4,
            channels=(64,128,256,512),
            lr=1e-4, 
            weight_decay=1e-4
            ):
        super().__init__(task=task, feat_dim=channels[-1], lr=lr, weight_decay=weight_decay)

        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, channels[0]),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.encoder = nn.Sequential(
            ConvBlock(channels[0], channels[0], sample=None,   residual=True, type="normal"),
            ConvBlock(channels[0], channels[1], sample="down", residual=True, type="normal"),
            ConvBlock(channels[1], channels[2], sample="down", residual=True, type="normal"),
            ConvBlock(channels[2], channels[3], sample="down", residual=True, type="normal"),
        )

        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        x = self.gap(x).flatten(1)   # -> [B, feat_dim]
        return x
    

class ResNet18(Baseline):
    def __init__(
            self, 
            task="multiclass", 
            in_ch=4,
            channels=(64,128,256,512),
            lr=1e-4, 
            weight_decay=1e-4
            ):
        super().__init__(task=task, feat_dim=channels[-1], lr=lr, weight_decay=weight_decay)

        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, channels[0]),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.encoder = nn.Sequential(
            ConvBlock(channels[0], channels[0], sample=None,   residual=True, type="normal"),
            ConvBlock(channels[0], channels[0], sample=None,   residual=True, type="normal"),
            ConvBlock(channels[0], channels[1], sample="down", residual=True, type="normal"),
            ConvBlock(channels[1], channels[1], sample=None,   residual=True, type="normal"),
            ConvBlock(channels[1], channels[2], sample="down", residual=True, type="normal"),
            ConvBlock(channels[2], channels[2], sample=None,   residual=True, type="normal"),
            ConvBlock(channels[2], channels[3], sample="down", residual=True, type="normal"),
            ConvBlock(channels[3], channels[3], sample=None,   residual=True, type="normal"),
        )

        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        x = self.gap(x).flatten(1)   # -> [B, feat_dim]
        return x
