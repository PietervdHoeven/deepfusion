import torch
import torch.nn as nn

from deepfusion.models.blocksv2 import ConvBlock3D, Downsample3D, Upsample3D

class Encoder(nn.Module):
    """
    (B, in_ch, 128,128,128) -> (B, d_enc)
    Channels: in_ch→16→32→64→128→256, then head to d_enc.
    """
    def __init__(self, in_ch=1, base=16, downsample: str = "learned"):
        super().__init__()
        self.conv1 = ConvBlock3D(in_ch, base)                       # (16, 128³)
        self.down1 = Downsample3D(base, base, downsample)           # (16, 64³)
        self.conv2 = ConvBlock3D(base, base*2)                      # (32, 64³)
        self.down2 = Downsample3D(base*2, base*2, downsample)       # (32, 32³)
        self.conv3 = ConvBlock3D(base*2, base*4)                    # (64, 32³)
        self.down3 = Downsample3D(base*4, base*4, downsample)       # (64, 16³)
        self.conv4 = ConvBlock3D(base*4, base*8)                    # (128, 16³)
        self.down4 = Downsample3D(base*8, base*8, downsample)       # (128, 8³)
        self.conv5 = ConvBlock3D(base*8, base*16)                   # (256, 8³)
        self.down5 = Downsample3D(base*16, base*16, downsample)     # (256, 4³)
        self.conv6 = ConvBlock3D(base*16, base*16)                  # (256, 4³)


    def forward(self, x):                               # x: [B, 1, 128,128,128]
        x = self.conv1(x)                               # [B, base, 128,128,128]
        x = self.down1(x)                               # [B, base*2, 64,64,64]
        x = self.conv2(x)                               # [B, base*2, 64,64,64]
        x = self.down2(x)                               # [B, base*4, 32,32,32]
        x = self.conv3(x)                               # [B, base*4, 32,32,32]
        x = self.down3(x)                               # [B, base*8, 16,16,16]
        x = self.conv4(x)                               # [B, base*8, 16,16,16]
        x = self.down4(x)                               # [B, base*16, 8,8,8]
        x = self.conv5(x)                               # [B, base*16, 8,8,8]
        x = self.down5(x)                               # [B, base*16, 4,4,4]
        x = self.conv6(x)                               # [B, base*16, 4,4,4]                           
        return x                                         

class Decoder(nn.Module):
    """
    (B, d_enc) -> (B, out_ch, 128,128,128)
    Mirrors encoder channels: 256→128→64→32→16→8, then head to out_ch.
    """
    def __init__(self, out_ch=1, base=16, upsample: str = "learned"):
        super().__init__()
        self.conv1 = ConvBlock3D(base*16, base*16)                  # (256, 4³)
        self.up1   = Upsample3D(base*16, base*16, upsample)         # (256, 8³)
        self.conv2 = ConvBlock3D(base*16, base*8)                   # (128, 8³)
        self.up2   = Upsample3D(base*8, base*8, upsample)           # (128, 16³)
        self.conv3 = ConvBlock3D(base*8, base*4)                    # (64, 16³)
        self.up3   = Upsample3D(base*4, base*4, upsample)           # (64, 32³)
        self.conv4 = ConvBlock3D(base*4, base*2)                    # (32, 32³)
        self.up4   = Upsample3D(base*2, base*2, upsample)           # (32, 64³)
        self.conv5 = ConvBlock3D(base*2, base)                      # (16, 64³)
        self.up5   = Upsample3D(base, base, upsample)               # (16, 128³)
        self.head  = nn.Conv3d(base, out_ch, kernel_size=3, padding=1)   # (out_ch, 128³)

    def forward(self, z):                   # z: [B, base*16, 4, 4, 4]
        z = self.conv1(z)                   # [B, 256, 4, 4, 4]
        z = self.up1(z)                     # [B, 256, 8, 8, 8]
        z = self.conv2(z)                   # [B, 128, 8, 8, 8]
        z = self.up2(z)                     # [B, 128, 16, 16, 16]
        z = self.conv3(z)                   # [B, 64, 16, 16, 16]
        z = self.up3(z)                     # [B, 64, 32, 32, 32]
        z = self.conv4(z)                   # [B, 32, 32, 32, 32]
        z = self.up4(z)                     # [B, 32, 64, 64, 64]
        z = self.conv5(z)                   # [B, 16, 64, 64, 64]
        z = self.up5(z)                     # [B, 16, 128, 128, 128]
        x_pred = self.head(z)
        return x_pred                             # [B, 1, 128,128,128]
    
class Autoencoder3D(nn.Module):
    def __init__(self, in_ch=1, base=16, downsample: str = "learned", upsample: str = "learned"):
        super().__init__()
        self.encoder = Encoder(in_ch=in_ch, base=base, downsample=downsample)
        self.decoder = Decoder(out_ch=in_ch, base=base, upsample=upsample)

    def forward(self, x):                       # x: [B, 1, 128,128,128]
        z = self.encoder(x)                     # [B, 256, 4,4,4]
        x_pred = self.decoder(z)                 # [B, 1, 128,128,128]
        return x_pred
    

# test forward pass
if __name__ == "__main__":
    model = Autoencoder3D(in_ch=1, base=16, downsample="learned", upsample="learned")
    x = torch.randn(50, 1, 128, 128, 128)  # batch of 2 samples
    x_pred = model(x)
    print(x_pred.shape)  # should be [2, 1, 128, 128, 128]