import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Dummy 3D volumes
B, C, D, H, W = 2, 1, 128, 128, 128
x_true = torch.rand(B, C, D, H, W)           # target in [0,1]
x_pred = x_true * 0.9 + 0.05 * torch.rand(B, C, D, H, W)  # slightly corrupted version

# Compute SSIM (higher = more similar)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
ssim_val = ssim(x_pred, x_true)  # scalar in [0,1]
print("SSIM:", ssim_val.item())