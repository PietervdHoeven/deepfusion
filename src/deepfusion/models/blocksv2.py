import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))
    
class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample: str = "learned"):
        super().__init__()
        if downsample == "max":
            self.downsample = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        elif downsample == "avg":
            self.downsample = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        elif downsample == "learned":
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError("downsample must be one of {'max', 'avg', 'learned'}")

    def forward(self, x):
        x = self.downsample(x)
        return x

class Upsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, upsample: str = "learned"):
        super().__init__()
        if upsample == "nearest":
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        elif upsample == "trilinear":
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        elif upsample == "learned":
            self.upsample = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError("upsample must be one of {'learned', 'nearest', 'trilinear'}")

    def forward(self, x):
        x = self.upsample(x)
        return x
