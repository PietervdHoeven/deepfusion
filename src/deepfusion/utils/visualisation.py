from pyexpat import model
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, no Tk involved
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml

# from deepfusion.datasets.ae_dataset import AEDataset
# from deepfusion.models.autoencoders import AutoEncoder


def plot_recons(x, x_pred, fname="reconstructions.png"):
    # x, x_pred: (B,1,D,H,W) or (B,D,H,W)
    x = x.detach().cpu().numpy()
    y = x_pred.detach().cpu().numpy()
    if x.ndim == 5 and x.shape[1] == 1: x = x[:,0]
    if y.ndim == 5 and y.shape[1] == 1: y = y[:,0]

    n = x.shape[0]
    fig, axes = plt.subplots(n, 6, figsize=(12, 2*n))
    if n == 1: axes = axes[None, :]
    titles = ["Axial-Input","Axial-Reconstructed","Coronal-Input","Coronal-Reconstructed","Sagittal-Input","Sagittal-Reconstructed"]
    for c,t in enumerate(titles): axes[0,c].set_title(t, fontsize=9)

    for r in range(n):
        inp, rec = x[r], y[r]            # (D,H,W)
        D,H,W = inp.shape
        z,yc,xc = D//2, H//2, W//2
        imgs = [inp[z], rec[z], inp[:,yc,:], rec[:,yc,:], inp[:,:,xc], rec[:,:,xc]]
        for c in range(6):
            axes[r,c].imshow(imgs[c], cmap="gray", origin="lower"); axes[r,c].axis("off")

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)