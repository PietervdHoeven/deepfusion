# --- utils/losses.py ---

def masked_l1(pred, target, mask, eps=1e-8):
    num = (mask * (pred - target).abs()).sum()
    den = mask.sum().clamp_min(eps)
    return num / den

def masked_l2(pred, target, mask, eps=1e-8):
    num = (mask * (pred - target).pow(2)).sum()
    den = mask.sum().clamp_min(eps)
    return num / den

def weighted_l1(pred, target, mask, w_fg=1.0, w_bg=0.05, eps=1e-8):
    w = w_fg*mask + w_bg*(1-mask)
    num = (w * (pred - target).abs()).sum()
    den = w.sum().clamp_min(eps)
    return num / den

def weighted_l2(pred, target, mask, w_fg=1.0, w_bg=0.05, eps=1e-8):
    w = w_fg*mask + w_bg*(1-mask)
    num = (w * (pred - target).pow(2)).sum()
    den = w.sum().clamp_min(eps)
    return num / den