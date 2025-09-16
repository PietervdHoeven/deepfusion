import torch
import torch.nn as nn
from deepfusion.models.blocks import ReconBlock3D

class Decoder(nn.Module):
    """
    (B, d_enc) -> (B, out_ch, 128,128,128)
    Mirrors encoder channels: 256→128→64→32→16→8, then head to out_ch.
    """
    def __init__(self, d_enc=256, base=16, out_ch=1):
        super().__init__()
        in_ch = base * 16  # matches last encoder stage (e.g., 256 when base=16)

        # Initial linear layer to reshape latent vector to 4x4x4 feature map
        self.seed = nn.Sequential(
            nn.Linear(d_enc, in_ch * 4 * 4 * 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (in_ch, 4, 4, 4)),
        )
        # Upsampling ladder (each ReconBlock3D does x2 upsample via your deconv)
        self.up1 = ReconBlock3D(in_ch,    base*8)   # 4 -> 8
        self.up2 = ReconBlock3D(base*8,   base*4)   # 8 -> 16
        self.up3 = ReconBlock3D(base*4,   base*2)   # 16 -> 32
        self.up4 = ReconBlock3D(base*2,   base)     # 32 -> 64
        self.up5 = ReconBlock3D(base,     base//2)  # 64 -> 128

        self.head = nn.Conv3d(base//2, out_ch, kernel_size=3, padding=1)

    def forward(self, z_vec):                   # z_vec: [B, d_enc]
        x = self.seed(z_vec)                    # [B, C4, 4, 4, 4]
        x = self.up1(x)                         # 8
        x = self.up2(x)                         # 16
        x = self.up3(x)                         # 32
        x = self.up4(x)                         # 64
        x = self.up5(x)                         # 128
        return self.head(x)                     # [B, out_ch, 128,128,128]