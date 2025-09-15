import matplotlib.pyplot as plt
import torch

from deepfusion.datamodules.ae_datamodule import AE_DataModule

def test_mask_alignment():
    # init your datamodule (adjust paths in configs if needed)
    dm = AE_DataModule()
    dm.setup(stage="test")  # or "fit" if you want training set

    ds = dm.test_dataloader().dataset  # get underlying dataset

    # grab a few samples
    for i in range(3):
        x, mask = ds[i]  # x: [G, X, Y, Z], mask: [X, Y, Z]
        print(f"Sample {i}: x={tuple(x.shape)}, mask={tuple(mask.shape)}")

        z = 64
        # first gradient for visualization
        vol_slice = x[0, :, :, z].cpu()
        mask_slice = mask[0, :, :, z].cpu()

        print(f"shape: vol_slice={vol_slice.shape}, mask_slice={mask_slice.shape}")

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(vol_slice, cmap="gray", origin="lower")
        axs[0].set_title(f"Volume slice z={z}")
        axs[0].axis("off")

        axs[1].imshow(mask_slice, cmap="gray", origin="lower")
        axs[1].set_title(f"Mask slice z={z}")
        axs[1].axis("off")

        plt.tight_layout()
        plt.savefig(f"sample_{i}_z{z}.png", dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    test_mask_alignment()