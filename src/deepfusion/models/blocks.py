import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def deconv3x3x3(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3x3(in_channels, out_channels, stride)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(out_channels, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    
class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample: str | None = None):
        super().__init__()
        self.conv = conv3x3x3(in_channels, out_channels, stride)
        self.norm = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)
        if downsample == "max":
            self.downsample = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        elif downsample == "avg":
            self.downsample = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        elif downsample == "learned":
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
        else:
            self.downsample = None

    def forward(self, x):
        out = self.relu(self.norm(self.conv(x)))
        if self.downsample is not None:
            out = self.downsample(out)
        return out

class ReconBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = deconv3x3x3(in_channels, out_channels)
        self.norm = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.norm(self.deconv(x)))
        return out
    

class Embedder(nn.Module):
    def __init__(self, d_enc=256, d_model=384):
        super().__init__()
        self.proj_z = nn.Linear(d_enc, d_model, bias=False)
        self.proj_g = nn.Linear(4,     d_model, bias=True)
        self.z_mask = nn.Parameter(torch.zeros(d_model))  # [d_model]
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, z, g, mdm_mask=None):
        """
        z: [B, L, d_enc]
        g: [B, L, 4]
        mdm_mask: [B, L] bool or None
        """
        z_emb = self.proj_z(z)        # [B, L, d_model]
        g_emb = self.proj_g(g)        # [B, L, d_model]

        if mdm_mask is not None:
            mask = mdm_mask.unsqueeze(-1)         # [B, L, 1]
            z_emb = torch.where(
                mask,                             # condition
                self.z_mask.view(1, 1, -1),       # replacement [1,1,d_model] -> broadcast
                z_emb                             # original
            )                                    # [B, L, d_model]

        token = z_emb + g_emb                     # [B, L, d_model]
        return self.norm(token)
   
    
class Unembedder(nn.Module):
    """
    Acts like nn.Linear(d_model -> d_enc, bias=False)
    """
    def __init__(self, d_model=384, d_enc=256):
        super().__init__()
        self.proj = nn.Linear(d_model, d_enc, bias=False)
    
    def forward(self, x):
        return self.proj(x)  # [B, L, d_enc]