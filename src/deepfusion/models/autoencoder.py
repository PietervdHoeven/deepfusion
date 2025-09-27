from deepfusion.models.blocks.convolution import ConvBlock
import torch.nn as nn

class AE5D(nn.Module):
    def __init__(self, in_channels=1, channels=[16, 32, 64, 128, 256, 384], residual=True):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
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

        self.decoder = nn.Sequential(
            ConvBlock(channels[5], channels[4], sample="up", residual=residual, type="depthwise"),
            ConvBlock(channels[4], channels[3], sample="up", residual=residual, type="depthwise"),
            ConvBlock(channels[3], channels[2], sample="up", residual=residual, type="normal"),
            ConvBlock(channels[2], channels[1], sample="up", residual=residual, type="normal"),
            ConvBlock(channels[1], channels[0], sample="up", residual=residual, type="normal"),
        )

        self.head = nn.Conv3d(channels[0], 1, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return self.head(x)