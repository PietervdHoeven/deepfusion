import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import AUROC, MeanAbsoluteError

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet3D_18(pl.LightningModule):
    def __init__(self, task="tri_cdr", lr=1e-3, in_channels=4):
        super().__init__()
        self.task = task
        self.save_hyperparameters()

        # Initial convolution
        self.in_planes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ResNet-18 layers
        self.layer1 = self._make_layer(64, 1)
        self.layer2 = self._make_layer(128, 1, stride=2)
        self.layer3 = self._make_layer(256, 1, stride=2)
        self.layer4 = self._make_layer(512, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # Output head and metrics
        if self.task == "tri_cdr":
            self.num_classes = 3
            self.fc = nn.Linear(512, self.num_classes)
            self.train_auroc = AUROC(task="multiclass", num_classes=3, average="macro")
            self.val_auroc = AUROC(task="multiclass", num_classes=3, average="macro")
        elif self.task == "age":
            self.fc = nn.Linear(512, 1)
            self.train_mae = MeanAbsoluteError()
            self.val_mae = MeanAbsoluteError()
        else:
            self.fc = nn.Linear(512, 1)
            self.train_auroc = AUROC(task="binary")
            self.val_auroc = AUROC(task="binary")

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        layers = [BasicBlock3D(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _step(self, batch, stage: str):
        x, y = batch
        log_args = dict(on_step=False, on_epoch=True, prog_bar=True)

        if self.task == "tri_cdr":
            logits = self.forward(x)
            loss = F.cross_entropy(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            auroc = self.train_auroc if stage == "train" else self.val_auroc
            auroc.update(logits, y)
            self.log(f"{stage}_loss", loss, **log_args)
            self.log(f"{stage}_acc", acc, **log_args)
            self.log(f"{stage}_auroc", auroc, **log_args)
            return loss

        elif self.task == "age":
            pred = self.forward(x).squeeze(-1)
            loss = F.mse_loss(pred, y.float())
            mae = self.train_mae if stage == "train" else self.val_mae
            mae.update(pred, y.float())
            self.log(f"{stage}_loss", loss, **log_args)
            self.log(f"{stage}_mae", mae, **log_args)
            return loss

        else:
            logits = self.forward(x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
            preds = (torch.sigmoid(logits) > 0.5).long()
            acc = (preds == y).float().mean()
            auroc = self.train_auroc if stage == "train" else self.val_auroc
            auroc.update(torch.sigmoid(logits), y)
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