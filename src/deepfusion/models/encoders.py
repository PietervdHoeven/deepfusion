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
    

class ZMapCNNEncoder(nn.Module):
    """
    Input:  z ∈ [B, 256, 4, 4, 4]
    Output: token ∈ [B, d_token]
    """
    def __init__(self, c_in: int = 256, d_token: int = 512):
        super().__init__()
        self.conv1 = nn.Conv3d(c_in, 384, kernel_size=3, stride=2, padding=1)
        # 2 → 1
        self.conv2 = nn.Conv3d(384, 512, kernel_size=3, stride=2, padding=1)
        self.proj  = nn.Linear(512, d_token)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(z))          # [B,384,2,2,2]
        x = F.relu(self.conv2(x))          # [B,512,1,1,1]
        x = x.flatten(1)                    # [B,512]
        token = self.proj(x)                # [B,d_token]
        return token


