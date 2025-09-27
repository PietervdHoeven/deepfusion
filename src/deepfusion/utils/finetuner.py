import os
import torch
import torch.nn as nn

def load_pretrained_backbone(backbone: nn.Module, ckpt_path: str, prefix: str = "model."):
    """
    Load pretrained weights into a backbone from a Lightning checkpoint.

    Args:
        backbone: nn.Module that should receive the weights.
        ckpt_path: path to the checkpoint file (.ckpt).
        prefix: prefix in the state dict that should be stripped (default "model.").

    Returns:
        (missing_keys, unexpected_keys) from load_state_dict
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    # keep only parameters that start with prefix and strip it
    filtered = {k.replace(prefix, ""): v for k, v in state.items() if k.startswith(prefix)}

    missing, unexpected = backbone.load_state_dict(filtered, strict=False)
    print(f"[load] {os.path.basename(ckpt_path)} -> "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    return missing, unexpected