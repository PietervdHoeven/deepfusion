import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfusion.models.blocks import ResBlock3D, BasicBlock3D, ReconBlock3D

class BasicEncoder(nn.Module):
    def __init__(self, in_ch=1, base=16, out_dim=256, downsample: str | None = "learned"):
        super().__init__()
        self.body = nn.Sequential(
            BasicBlock3D(in_ch,   base,    downsample=downsample),  # 128->64
            BasicBlock3D(base,    base*2,  downsample=downsample),  # 64->32
            BasicBlock3D(base*2,  base*4,  downsample=downsample),  # 32->16
            BasicBlock3D(base*4,  base*8,  downsample=downsample),  # 16->8
            BasicBlock3D(base*8,  base*16, downsample=downsample),  # 8->4
        )
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.head = nn.Conv3d(base*16, out_dim, 1, bias=False)  # 1x1x1 → d_enc

    def forward(self, x):           # x: [B, 1, D, H, W]
        z = self.body(x)            # [B, C, 4,4,4]
        z = self.pool(z)            # [B, C, 1,1,1]
        z = self.head(z).flatten(1) # [B, d_enc]
        return z
    

class ResEncoder(nn.Module):
    def __init__(self, in_ch=1, base=16, out_dim=256):
        super().__init__()
        self.body = nn.Sequential(
            ResBlock3D(in_ch,   base,    stride=2),  # 128->64
            ResBlock3D(base,    base*2,  stride=2),  # 64->32
            ResBlock3D(base*2,  base*4,  stride=2),  # 32->16
            ResBlock3D(base*4,  base*8,  stride=2),  # 16->8
            ResBlock3D(base*8,  base*16, stride=2),  # 8->4
        )
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.head = nn.Conv3d(base*16, out_dim, 1, bias=False)  # 1x1x1 → d_enc

    def forward(self, x):           # x: [B*V, 1, D, H, W]
        z = self.body(x)            # [B*V, C, 4,4,4]
        z = self.pool(z)            # [B*V, C, 1,1,1]
        z = self.head(z).flatten(1) # [B*V, d_enc]
        return z


# okay its time to setup the transformer. We'll hardcode layer dimensions for now. No need to gridsearch yet. We want the following architecture:

# encoder takes a series of 1x128x128x128 volumes. We might get multiple series per batch but for now we'll just focus on a batch_size=1. So we have an encoder that loops over all the volumes and encodes them

