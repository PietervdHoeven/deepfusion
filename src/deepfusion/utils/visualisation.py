from pyexpat import model
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml

# from deepfusion.datasets.ae_dataset import AEDataset
# from deepfusion.models.autoencoders import AutoEncoder


def plot_reconstructions(batch, model):
    # dataset = AEDataset(data_dir="data", stage="test")
    # vols = []
    # for i in indices:
    #     v, mask, patient_id, session_id = dataset[i]
    #     print(f"volume shape: {v.shape}, dtype: {v.dtype}, patient: {patient_id}, session: {session_id}")
    #     vols.append((v, mask))  # expect (D,H,W)

    # with open("/home/spieterman/projects/deepfusion/logs/recon-AutoEncoder3D/train/version_20/hparams.yaml", "r") as f:
    #     hparams = yaml.safe_load(f)

    # model = AutoEncoder(**hparams).to("cuda")
    # model.load_state_dict(
    #     torch.load(
    #         "/home/spieterman/projects/deepfusion/logs/recon-AutoEncoder3D/train/version_20/checkpoints/best-epoch:66-val_loss:0.2663.ckpt"
    #         )["state_dict"]
    #         )
    # model.eval()
    
    y = []
    with torch.no_grad():
        for v, mask, patient_id, session_id in batch:
            x = torch.tensor(v.unsqueeze(0), dtype=torch.float32).to("cuda")  # (1,1,D,H,W)
            rec = model(x)  # (1,1,D,H,W)
            rec = rec.squeeze(0).cpu() * mask  # apply mask
            rec = rec.detach().numpy()
            y.append(np.squeeze(rec))  # expect (D,H,W)
            print(np.squeeze(rec).shape)

    n = len(batch)
    fig, axes = plt.subplots(n, 6, figsize=(12, 2*n))
    if n == 1: axes = axes[None, :]
    titles = ["Axial-Input","Axial-Reconstructed","Coronal-Input","Coronal-Reconstructed","Sagittal-Input","Sagittal-Reconstructed"]
    for c,t in enumerate(titles): axes[0,c].set_title(t, fontsize=9)

    for r in range(n):
        inp = batch[r][0].numpy()  # (1,D,H,W)
        rec = y[r]                # (D,H,W)
        inp = np.squeeze(inp[0])  # (D,H,W)
        print(inp.shape, rec.shape)
        D,H,W = inp.shape
        z,yc,xc = D//2, H//2, W//2
        imgs = [inp[z], rec[z], inp[:,yc,:], rec[:,yc,:], inp[:,:,xc], rec[:,:,xc]]
        for c in range(6):
            axes[r,c].imshow(imgs[c], cmap="gray", origin="lower"); axes[r,c].axis("off")

    plt.tight_layout()
    plt.savefig("reconstructions.png")

plot_reconstructions()