import torch.nn as nn

def proj1x1x1(in_channels, out_channels, sample: str | None = None):
    if sample == "down":
        # AvgPool → 1x1x1 conv (padding=0)
        return nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )
    elif sample == "up":
        # Upsample → 1x1x1 conv (padding=0)
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )
    else:
        return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

def nrml_conv3x3x3(in_channels, out_channels, sample: str | None = None, bias=False):
    if sample == "down":
        return nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        )
    elif sample == "up":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        )
    else:
        return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

def dw_conv3x3x3(in_channels, out_channels, sample: str | None = None, bias=False):
    if sample == "down":
        return nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=bias),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        )
    elif sample == "up":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=bias),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        )
    else:
        return nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=bias),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        )

def grpd_conv3x3x3(in_channels, out_channels, sample: str | None = None, bias=False, groups=8):
    if sample == "down":
        return nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=bias)
        )
    elif sample == "up":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=bias)
        )
    else:
        return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=bias)

def conv3x3x3(in_channels, out_channels, type="normal", sample: str | None = None, bias=False):
    if type == "normal":
        return nrml_conv3x3x3(in_channels, out_channels, sample=sample, bias=bias)
    elif type == "depthwise":
        return dw_conv3x3x3(in_channels, out_channels, sample=sample, bias=bias)
    else:
        return grpd_conv3x3x3(in_channels, out_channels, sample=sample, bias=bias)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sample: str | None = None, residual=True, type="normal"):
        super().__init__()
        self.residual = residual
        self.conv1 = conv3x3x3(in_channels, out_channels, type=type, sample=sample)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3x3(out_channels, out_channels, type=type)
        self.norm2 = nn.GroupNorm(8, out_channels)
        if (sample != None or in_channels != out_channels) and residual:
            self.proj = proj1x1x1(in_channels, out_channels, sample=sample)
        else:
            self.proj = nn.Identity()
    
        # forward kept identical to the original
    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.residual:
            x = self.proj(x)
            y += x
        y = self.relu(y)
        return y
