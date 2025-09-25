from deepfusion.models.autoencoder import Autoencoder
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os

model = Autoencoder.load_from_checkpoint(
    "/home/spieterman/projects/deepfusion/checkpoints/Autoencoder.ckpt"
)
print(model)
manifest = pd.read_csv("data/deepfusion/volumes/manifest.csv")

def forward(model, x):
    model.eval()
    with torch.no_grad():
        x = model.model.stem(x)
        z = model.model.encoder(x)
    return z

for group_keys, group_df in tqdm(manifest.groupby(["patient_id", "session_id"])):
    # group info
    patient_id, session_id = group_keys
    num_g = group_df["g_idx"].max() + 1  # g_idx starts from 0
    stage = group_df["stage"].iloc[0]

    # load data
    dwi_data = np.load(
        f"data/deepfusion/volumes/{stage}/{patient_id}/{session_id}/{patient_id}_{session_id}_normalised-dwi.npy",
        mmap_mode="r",
    )  # shape [N,D,H,W]
    grads = np.load(
        f"data/deepfusion/volumes/{stage}/{patient_id}/{session_id}/{patient_id}_{session_id}_grads.npy",
        mmap_mode="r",
    )  # shape [N,4]

    # encode in batches
    z_list = []
    BATCH_SIZE = 35
    for i in range(0, num_g, BATCH_SIZE):
        j = min(i + BATCH_SIZE, num_g)

        dwi_batch = dwi_data[i:j]  # shape [B,D,H,W]

        x = torch.from_numpy(dwi_batch).unsqueeze(1).to(dtype=torch.float32).cuda()  # [B,1,D,H,W]

        z_batch = forward(model, x)  # [B,1,D,H,W] (bottleneck maps)

        z_arr = z_batch.squeeze(1).cpu().numpy()  # [B,D,H,W]
        z_list.append(z_arr)

    z_all = np.concatenate(z_list, axis=0)  # shape [N,D,H,W]

    # save maps and grads
    maps_path = f"data/deepfusion/latents/{stage}/{patient_id}/{session_id}/{patient_id}_{session_id}_latent-maps.npy"
    os.makedirs(os.path.dirname(maps_path), exist_ok=True)
    np.save(maps_path, z_all)

    grads_path = f"data/deepfusion/latents/{stage}/{patient_id}/{session_id}/{patient_id}_{session_id}_grads.npy"
    os.makedirs(os.path.dirname(grads_path), exist_ok=True)
    np.save(grads_path, grads)
