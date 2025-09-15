# --- utils/losses.py ---
import torch

def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B,C,D,H,W)
    mask:       (B,1,D,H,W) bool; True=brain
    """
    mask = mask.to(dtype=pred.dtype, device=pred.device)           # convert to float mask
    diff = (pred - target).abs() * mask                            # zero background
    return diff.sum() / mask.sum()