import torch.nn as nn
from deepfusion.models.blocks.convolution import ConvBlock

class EncoderOnly(nn.Module):
    def __init__(
            self,  
            in_ch=4, 
            channels=(16,32,64,128,256,384), 
            residual=True, 
            ):
        super().__init__()
        self.d = channels[-1]

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


class ResNet10(nn.Module):
    def __init__(
            self, 
            task="multiclass", 
            in_ch=4,
            channels=(64,128,256,512)
            ):
        super().__init__()
        self.d = channels[-1]

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
    

class ResNet18(nn.Module):
    def __init__(
            self, 
            in_ch=4,
            channels=(64,128,256,512),
            ):
        super().__init__()
        self.d = channels[-1]

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
    
