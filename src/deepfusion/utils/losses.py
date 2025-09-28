# --- utils/losses.py ---
import torch
import torch.nn.functional as F

def masked_l1(pred, target, mask, eps=1e-8):
    num = (mask * (pred - target).abs()).sum()
    den = mask.sum().clamp_min(eps)
    return num / den

def masked_l2(pred, target, mask, eps=1e-8):
    num = (mask * (pred - target).pow(2)).sum()
    den = mask.sum().clamp_min(eps)
    return num / den

def weighted_l1(pred, target, mask, w_fg=0.0, w_bg=1.0, eps=1e-8):
    w = w_fg*mask + w_bg*(1-mask)
    num = (w * (pred - target).abs()).sum()
    den = w.sum().clamp_min(eps)
    return num / den

def weighted_l2(pred, target, mask, w_fg=0.0, w_bg=1.0, eps=1e-8):
    w = w_fg*mask + w_bg*(1-mask)
    num = (w * (pred - target).pow(2)).sum()
    den = w.sum().clamp_min(eps)
    return num / den

def masked_charbonnier(pred, target, mask, eps=1e-3, eps_denom=1e-8):
    """
    Charbonnier (smooth L1) over foreground only.
    eps ~1e-3 is good for [0,1] or z-scored inputs.
    """
    diff = pred - target
    rho  = torch.sqrt(diff * diff + eps * eps)         # smooth |diff|
    num  = (mask * rho).sum()
    den  = mask.sum().clamp_min(eps_denom)
    return num / den

def weighted_charbonnier(pred, target, mask, w_fg=1.0, w_bg=0.01, eps=1e-3, eps_denom=1e-8):
    """
    Charbonnier with foreground/background weights.
    """
    w  = w_fg * mask + w_bg * (1.0 - mask)
    diff = pred - target
    rho  = torch.sqrt(diff * diff + eps * eps)
    num  = (w * rho).sum()
    den  = w.sum().clamp_min(eps_denom)
    return num / den

# ============================================================
# Masked reconstruction loss (apply only on masked directions)
# ============================================================
def masked_recon_loss(
    X_hat: torch.Tensor,          # (B, Q, S, C)
    X: torch.Tensor,              # (B, Q, S, C)
    q_space_mask: torch.Tensor,   # (B, Q) bool; True = masked (compute loss here)
    alpha: float = 1.0,           # weight for MSE term
    beta: float = 0.1             # weight for cosine term
) -> torch.Tensor:
    B, Q, S, C = X.shape
    # Broadcast mask over S and C: (B,Q) -> (B,Q,S,C)
    m = q_space_mask.view(B, Q, 1, 1).expand(B, Q, S, C)   # True where masked

    # Select only masked positions
    X_masked     = X[m].view(-1, C)       # (N, C), where N = num_masked * S
    X_hat_masked = X_hat[m].view(-1, C)   # (N, C)

    # --- 1) MSE term ---
    mse = F.mse_loss(X_hat_masked, X_masked)

    # --- 2) Cosine alignment term ---
    if X_masked.numel() > 0 and beta > 0:  # avoid NaNs if mask empty
        cos = F.cosine_embedding_loss(
            X_hat_masked,
            X_masked,
            torch.ones(X_masked.size(0), device=X.device),
            reduction="mean"
        )
    else:
        cos = torch.tensor(0.0, device=X.device)

    # --- Combine ---
    return alpha * mse + beta * cos