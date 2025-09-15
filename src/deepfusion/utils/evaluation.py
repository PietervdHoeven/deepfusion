import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import io
from PIL import Image

def compute_testset_mean_image(test_root: Path) -> np.ndarray:
    """
    Compute the per-voxel mean image across all normalized DWI sessions in the test set.
    Each session folder contains:
      - *_normalised-dwi.npy  (shape [G, X, Y, Z], float16/float32, background=0)
      - *_brain-mask.npy      (shape [X, Y, Z], bool)

    The mean is computed only over voxels where the session's mask is True,
    counting all G gradients per session equally.

    Parameters
    ----------
    test_root : Path
        e.g. Path('~/projects/deepfusion/data/deepfusion/volumes/test').expanduser()

    Returns
    -------
    mean_img : np.ndarray
        Array [X, Y, Z] float32: per-voxel mean over all sessions & gradients.
        Voxels never covered by any mask will be 0.
    """
    test_root = Path(test_root)

    # Find all session files
    norm_files = sorted(test_root.glob("sub-*/ses-*/*_normalised-dwi.npy"))
    if not norm_files:
        raise FileNotFoundError(f"No *_normalised-dwi.npy files found under {test_root}")

    sum_img = None   # float64 accumulator for numerical stability
    cnt_img = None   # int64 voxel-wise counts

    for norm_path in tqdm(norm_files, desc="Processing sessions"):
        # Derive matching mask path
        mask_path = norm_path.with_name(norm_path.name.replace("_normalised-dwi.npy", "_brain-mask.npy"))

        # Memory-map large arrays to keep RAM usage down
        dwi = np.load(norm_path, mmap_mode="r")       # shape [G, X, Y, Z]
        mask = np.load(mask_path).astype(bool)        # shape [X, Y, Z]

        G = dwi.shape[0]

        # Lazy-init accumulators with the first session's spatial shape
        if sum_img is None:
            sum_img = np.zeros(mask.shape, dtype=np.float64)
            cnt_img = np.zeros(mask.shape, dtype=np.int64)

        # Zero-out background per session, then sum over gradients
        # (Multiply once to avoid masking after the sum.)
        dwi_masked = dwi * mask[None, ...]            # broadcast mask to [G,X,Y,Z]
        session_sum = dwi_masked.sum(axis=0)          # [X,Y,Z]

        # Accumulate voxel-wise sums and counts
        sum_img += session_sum                         # add the per-voxel sum over G
        cnt_img += (mask.astype(np.int64) * G)         # each True voxel gets +G samples

    # Final mean: safe where count > 0, else 0
    mean_img = np.zeros_like(sum_img, dtype=np.float32)
    valid = cnt_img > 0
    mean_img[valid] = (sum_img[valid] / cnt_img[valid]).astype(np.float32)

    return mean_img


def plot_slices(x, x_hat, mean_image, mask, z_idx=None):
    """
    Create a side-by-side plot for a single sample at slice z_idx.
    x, x_hat, mean_image: [G, X, Y, Z]
    mask: [X, Y, Z]
    """
    # Make sure everything is on CPU and detached (no gradients)
    x = x.detach().cpu()
    x_hat = x_hat.detach().cpu()
    mean_image = mean_image.detach().cpu()
    mask = mask.detach().cpu()

    if z_idx is None:
        z_idx = x.shape[-1] // 2  # middle slice

    mask.to(device="cpu")
    x.to(device="cpu")
    x_hat.to(device="cpu")
    mean_image.to(device="cpu")

    print(f"mask shape: {mask.shape}, x shape: {x.shape}, x_hat shape: {x_hat.shape}, mean_image shape: {mean_image.shape}")

    # Pick the first gradient for visualization
    m   = mask[0, :, :, z_idx]                                # [1, H,W]
    x0  = x[0, :, :, z_idx]   * m                          # [1, H,W]
    xh  = x_hat[0, :, :, z_idx]* m
    xm  = mean_image[0, :, :, z_idx]* m
    r_model  = (x0 - xh).abs()
    r_mean  = (x0 - xm).abs()

    def prep(img, mask, clip=None):
        if clip is None:
            # For z-scored images: robust clip to [-2,2] then scale to [0,1]
            img = img.clamp(-2, 2)
            img = (img + 2.0) / 4.0
        else:
            # For residuals: clip to [0, clip] then scale to [0,1]
            img = (img * mask).clamp(0, clip) / max(clip, 1e-6)
        return img.unsqueeze(0)                           # [1,1,H,W]

    panel_input  = prep(x0, m)
    panel_ae     = prep(xh, m)
    panel_mean   = prep(xm, m)
    panel_r_ae   = prep(r_model, m, clip=2.0)
    panel_r_mean = prep(r_mean, m, clip=2.0)

    # Concatenate horizontally: [5, 1, H, W]
    grid = torch.cat([panel_input, panel_ae, panel_mean, panel_r_ae, panel_r_mean], dim=0).float().unsqueeze(1)
    print(grid.shape)

    return grid  # CHW for TensorBoard add_image

# # --- Example usage ---
# if __name__ == "__main__":
#     root = Path("~/projects/deepfusion/data/deepfusion/volumes/test").expanduser()
#     mean_img = compute_testset_mean_image(root)

#     z_idx = mean_img.shape[2] // 2
#     plt.imshow(mean_img[:, :, z_idx].T, cmap="gray", origin="lower")
#     plt.title(f"Axial slice z={z_idx}")
#     plt.axis("off")
#     plt.show()

#     # Optional: save for your baseline
#     out_path = root / "mean_image_testset.npy"
#     np.save(out_path, mean_img)
#     plt.savefig(root / "mean_image_slice.png")
#     print(f"Saved mean image to: {out_path}")
#     print("Mean/Std of mean image (inside non-zero voxels):",
#           float(mean_img[mean_img != 0].mean()),
#           float(mean_img[mean_img != 0].std()))

