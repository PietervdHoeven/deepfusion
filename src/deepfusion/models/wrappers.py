import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchmetrics import MetricCollection, MeanAbsoluteError, R2Score, AUROC, Recall, ConfusionMatrix
from deepfusion.models.autoencoder import AE5D
from deepfusion.utils.losses import masked_l1, weighted_l1, weighted_l2, masked_l2
from deepfusion.utils.visualisation import plot_recons
from deepfusion.utils.masking import patch_mask
from torchmetrics import StructuralSimilarityIndexMeasure

from deepfusion.models.transformer import AxialMaskedModellingTransformer, AxialPredictingTransformer
from deepfusion.utils.losses import masked_recon_loss
from deepfusion.utils.finetuner import load_pretrained_backbone


class AutoencoderPretrainer(pl.LightningModule):
    """
    PyTorch Lightning module for 3D convolutional autoencoder pretraining.
    Args:
        model (nn.Module): Autoencoder model with `forward(x) -> x_recon`.
        loss_fn (callable): Loss function for reconstruction (default: masked_l1).
        lr (float, optional): Learning rate for optimizer. Default is 1e-4.
        weight_decay (float, optional): Weight decay for optimizer. Default is 0.
        betas (Tuple[float, float], optional): AdamW optimizer betas. Default is (0.9, 0.999).
        masked_pretraining (bool, optional): Whether to apply random patch masking during training. Default is False.
    Methods:
        forward(x): Forward pass through the autoencoder.
        training_step(batch, batch_idx): Training step logic.
        validation_step(batch, batch_idx): Validation step logic.
        test_step(batch, batch_idx): Test step logic.
        configure_optimizers(): Configures optimizers and learning rate schedulers.
    Example:
        >>> model = AutoencoderPretrainer(model=AE5D(), loss_fn=masked_l1)
    """
    def __init__(
        self,
        model=AE5D(),
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


    

class TransformerPretrainer(pl.LightningModule):
    """
    PyTorch Lightning module for masked latent reconstruction using AxialMaskedModellingTransformer.
    Args:
        in_channels (int, optional): Input channel dimension. Default is 384.
        embed_dim (int, optional): Transformer hidden dimension. Default is 384.
        num_heads (int, optional): Number of attention heads. Default is 6.
        num_spatials (int, optional): Sequence length. Default is 36.
        num_layers (int, optional): Number of transformer layers. Default is 6.
        attn_dropout (float, optional): Dropout rate for attention layers. Default is 0.1.
        ffn_dropout (float, optional): Dropout rate for feed-forward layers. Default is 0.1.
        betas (Tuple[float, float], optional): AdamW optimizer betas. Default is (0.9, 0.95).
        lr (float, optional): Learning rate for optimizer. Default is 1e-4.
        weight_decay (float, optional): Weight decay for optimizer. Default is 5e-2.
    Methods:
        forward(X, G, Q_mask, pad_mask): Forward pass through the model.
        training_step(batch, batch_idx): Training step logic.
        validation_step(batch, batch_idx): Validation step logic.
        test_step(batch, batch_idx): Test step logic.
        configure_optimizers(): Configures optimizers and learning rate schedulers.
    Example:
        >>> model = TransformerPretrainer(in_channels=384, embed_dim=384, num_heads=6, num_spatials=36, num_layers=6)
    """
    def __init__(
        self,
        in_channels=384,
        embed_dim=384,
        num_heads=6,
        num_spatials=36,
        num_layers=6,
        attn_dropout=0.02,
        ffn_dropout=0.1,
        betas=(0.9, 0.95),
        lr=1e-4,
        weight_decay=5e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AxialMaskedModellingTransformer(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_spatials=num_spatials,
            num_layers=num_layers,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
        )

    def forward(self, x, g, Q_mask, pad_mask):
        # x: (B,Q,S,C), g: (B,Q,4), q_mask: (B,Q), pad_mask: (B,Q)
        return self.model(x, g, Q_mask=Q_mask, padding_mask=pad_mask)
    
    def _step(self, batch, stage: str):
        x, g, Q_mask, pad_mask = batch
        x_hat, _ = self(x, g, Q_mask, pad_mask)
        loss = masked_recon_loss(x_hat, x, Q_mask)
        m = Q_mask.unsqueeze(-1).unsqueeze(-1).expand_as(x_hat)  # (B,Q,1,1) -> (B,Q,S,C)
        mae = F.l1_loss(x_hat[m], x[m])
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, on_step=False, on_epoch=True)
        return self._step(batch, stage="train")
    
    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, stage="val")
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._step(batch, stage="test")
        return loss

    def configure_optimizers(self):
        # --- param groups: no weight decay on bias/LayerNorm ---
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or n.endswith(".bias"):   # biases + LayerNorm/BatchNorm weights
                no_decay.append(p)
            else:
                decay.append(p)

        optim_groups = [
            {"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        # --- AdamW optimizer ---
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            eps=1e-6
        )

        # --- Schedulers: linear warmup + cosine decay ---
        total_steps = self.trainer.estimated_stepping_batches
        print(f"Total training steps (est): {total_steps}")
        warmup_steps = max(1, int(0.05 * total_steps))
        cosine_steps = max(1, total_steps - warmup_steps)

        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=self.hparams.lr * 0.01
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # step-level updates
                "frequency": 1,
            }
        }


class ClassifierRegressorTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training classification and regression models.
    This wrapper supports binary, multiclass, and regression tasks, and can be used with any backbone
    `nn.Module` that outputs pooled feature vectors. It automatically configures appropriate heads,
    metrics, and logging for the following tasks:
        - "bin_cdr": Binary classification (e.g., healthy vs. MCI/AD)
        - "tri_cdr": Multiclass classification (3 classes: HC, MCI, AD)
        - "gender": Binary classification (male/female)
        - "handedness": Binary classification (left/right)
        - "age": Regression
    Metrics are tracked per stage (train/val/test) and confusion matrices are saved for classification tasks.
    The optimizer is AdamW with ReduceLROnPlateau scheduler.
    Args:
        model (nn.Module): Backbone model with `forward(x) -> [B, feat_dim]`.
        task (str): Task type, one of {"bin_cdr", "tri_cdr", "gender", "handedness", "age"}.
        feat_dim (int): Feature dimension output by the backbone.
        lr (float, optional): Learning rate. Default is 1e-4.
        weight_decay (float, optional): Weight decay for optimizer. Default is 1e-4.
        patience (int, optional): Patience for LR scheduler. Default is 10.
    Methods:
        training_step: Training loop step.
        validation_step: Validation loop step.
        test_step: Test loop step.
        configure_optimizers: Returns optimizer and scheduler.
        on_train_epoch_end: Logs metrics at end of training epoch.
        on_validation_epoch_end: Logs metrics at end of validation epoch.
        on_test_start: Clears test buffers.
        on_test_epoch_end: Logs metrics and saves confusion matrix at end of test epoch.
    """
    def __init__(
            self,
            model: nn.Module,  # any nn.Module with forward(x) -> [B, feat_dim]
            task: str, 
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

        self.model = model

        hidden = 2 * self.model.embed_dim
        self.head = nn.Sequential(
            nn.Linear(self.model.embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

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

    def _logits(self, x):
        # x may be a tuple/list: (X, G, Q_mask, pad_mask)
        feats = self.model(*x) if isinstance(x, (tuple, list)) else self.model(x)
        return self.head(feats)  # subclass forward -> features; then linear head

    def _step(self, batch, stage):
        x, y = batch    

        if self.task == "age":
            pred = self._logits(x).squeeze(-1).float()
            y = y.float()
            loss = F.mse_loss(pred, y)
            m = self.metrics[f"{stage}_metrics"]
            m["mae"].update(pred, y)
            if "r2" in m: m["r2"].update(pred, y)
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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

        if self.task == "tri_cdr":
            class_names = ["HC", "MCI", "AD"]
        elif self.task == "bin_cdr":
            class_names = ["HC", "MCI/AD"]
        elif self.task == "handedness":
            class_names = ["Left", "Right"]
        else:   # gender
            class_names = ["Male", "Female"]
        ax.set(
            xticks=range(len(class_names)),
            yticks=range(len(class_names)),
            xlabel="Predicted label",
            ylabel="True label",
        )

        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
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
    


class TransformerFinetuner(ClassifierRegressorTrainer):
    """
    Finetunes with a fixed transformer backbone and trainable attention pool + head.
    Loads weights from a TransformerPretrainer checkpoint if provided.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        attn_pool: bool = False,
        task: str = "tri_cdr",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 10,
        pretrain_ckpt: str | None = None,   # <-- add this
    ):
        # Hardcoded backbone+pool inside the features wrapper
        features = AxialPredictingTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_pool=attn_pool,
        )

        # Load pretrained backbone
        if pretrain_ckpt:
            load_pretrained_backbone(features.transformer, pretrain_ckpt)

        super().__init__(
            model=features,
            task=task,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )

        # Freeze backbone, train pool+head
        for p in self.model.transformer.parameters():
            p.requires_grad = False
        for p in self.model.pool.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

        self.model.transformer.eval()