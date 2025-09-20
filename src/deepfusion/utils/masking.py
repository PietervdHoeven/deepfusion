import torch

def patch_mask(x, mask_ratio=0.5, ps=8):
    """
    Apply simple cubic patch masking to a 128x128x128 volume.
    - Input: x of shape (B, C, 128, 128, 128)
    - Output: x with ~mask_ratio fraction of 8x8x8 patches set to 0
    """
    B, C, D, H, W = x.shape
    ps = 8  # patch size
    gd, gh, gw = D // ps, H // ps, W // ps  # number of patches per dimension

    # Random binary mask on the patch grid (1=keep, 0=mask)
    patch_mask = (torch.rand(B, 1, gd, gh, gw, device=x.device) > mask_ratio).float()

    # Inflate patch mask to full voxel size
    voxel_mask = patch_mask.repeat_interleave(ps, dim=2)\
                            .repeat_interleave(ps, dim=3)\
                            .repeat_interleave(ps, dim=4)

    # Apply voxel mask (masked voxels -> 0)
    x_masked = x * voxel_mask
    return x_masked, voxel_mask