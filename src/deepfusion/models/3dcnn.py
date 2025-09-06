import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import AUROC, MeanAbsoluteError

class Vanilla3DCNN(pl.LightningModule):
    def __init__(self, task: str = "tri_cdr", lr=1e-4):
        """
        Modular 3D CNN for both classification (binary/multiclass) and regression (age).
        The output head and metrics are selected based on the task.
        Args:
            task: str, one of "tri_cdr" (3-class), "age" (regression), or any binary task.
            lr: float, learning rate for Adam optimizer.
        """
        super().__init__()
        self.task = task
        self.lr = lr
        self.save_hyperparameters()

        # --- 3D CNN Backbone ---
        # Applies a series of 3D convolutions, group normalization, ReLU, and max pooling.
        # Input: [B, 4, 128, 128, 128] (4 channels: e.g., DTI metrics)
        self.features = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, stride=1, padding=1),   # [B,16,128,128,128]
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

        # --- Output Head and Metrics ---
        # Selects the output layer and evaluation metrics based on the task.
        if self.task == "tri_cdr":
            # 3-class classification (e.g., CDR: 0, 0.5, 1+)
            self.num_classes = 3
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),  # Global average pooling to [B,256,1,1,1]
                nn.Flatten(),             # [B,256]
                nn.Linear(256, self.num_classes)  # [B,3]
            )
            # Macro-averaged multiclass AUROC for evaluation
            self.train_auroc = AUROC(task="multiclass", num_classes=3, average="macro")
            self.val_auroc = AUROC(task="multiclass", num_classes=3, average="macro")
        elif self.task == "age":
            # Regression (predicting age)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(256, 1)  # Single output for regression
            )
            # Mean Absolute Error (MAE) for evaluation
            self.train_mae = MeanAbsoluteError()
            self.val_mae = MeanAbsoluteError()
        else:
            # Binary classification (e.g., bin_cdr, gender, handedness)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(256, 1)  # Single logit for BCEWithLogitsLoss
            )
            # AUROC for binary classification
            self.train_auroc = AUROC(task="binary")
            self.val_auroc = AUROC(task="binary")

    def forward(self, x):
        """
        Forward pass through the CNN backbone and output head.
        Args:
            x: torch.Tensor, shape [B, 4, 128, 128, 128]
        Returns:
            Output logits or regression value, depending on task.
        """
        x = self.features(x)
        x = self.head(x)
        return x
    
    def _step(self, batch, stage: str):
        """
        Shared step for training/validation.
        Calculates loss, accuracy, and metrics based on the task.
        Args:
            batch: tuple (x, y)
            stage: "train" or "val"
        Returns:
            Loss tensor (for backprop in training)
        """
        x, y = batch
        log_args = dict(on_step=False, on_epoch=True, prog_bar=True)

        if self.task == "tri_cdr":
            # --- Multiclass classification ---
            logits = self.forward(x)  # [B, 3]
            loss = F.cross_entropy(logits, y)  # Cross-entropy loss
            preds = torch.argmax(logits, dim=1)  # Predicted class indices
            acc = (preds == y).float().mean()    # Accuracy
            auroc = self.train_auroc if stage == "train" else self.val_auroc
            auroc.update(logits, y)              # Update multiclass AUROC
            # Logging
            self.log(f"{stage}_loss", loss, **log_args)
            self.log(f"{stage}_acc", acc, **log_args)
            self.log(f"{stage}_auroc", auroc, **log_args)
            return loss

        elif self.task == "age":
            # --- Regression ---
            pred = self.forward(x).squeeze(-1)   # [B]
            loss = F.mse_loss(pred, y.float())   # Mean squared error loss
            mae = self.train_mae if stage == "train" else self.val_mae
            mae.update(pred, y.float())          # Update MAE metric
            # Logging
            self.log(f"{stage}_loss", loss, **log_args)
            self.log(f"{stage}_mae", mae, **log_args)
            return loss

        else:
            # --- Binary classification ---
            logits = self.forward(x).squeeze(-1)  # [B]
            loss = F.binary_cross_entropy_with_logits(logits, y.float())  # BCE loss
            probs = torch.sigmoid(logits)  # Predicted probabilities
            preds = (probs > 0.5).long()  # Predicted class (0 or 1)
            acc = (preds == y).float().mean()             # Accuracy
            auroc = self.train_auroc if stage == "train" else self.val_auroc
            auroc.update(probs, y)        # Update binary AUROC
            # Logging
            self.log(f"{stage}_loss", loss, **log_args)
            self.log(f"{stage}_acc", acc, **log_args)
            self.log(f"{stage}_auroc", auroc, **log_args)
            return loss
    
    def training_step(self, batch, batch_idx):
        """
        Training step: calls _step with 'train' stage.
        """
        return self._step(batch, stage="train")
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step: calls _step with 'val' stage.
        """
        self._step(batch, stage="val")

    def configure_optimizers(self):
        """
        Adam optimizer with learning rate from hyperparameters.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

# test run

# if __name__ == "__main__":
#     model = Vanilla3DCNN(task="tri_cdr", lr=1e-3)
#     x = torch.randn(2, 4, 128, 128, 128)
#     logits = model(x)
#     print(logits.shape)  # should be [2, num_classes]