from torch.utils.data import Dataset
import torch
import numpy as np
from pathlib import Path
import pandas as pd

from deepfusion.utils.labels import map_label

class DeepFusionDataset(Dataset):
    """
    Minimal Dataset for 3D volumes saved as a single .npy per session.

    - Reads meta_data.csv for session metadata.
    - Filters by stage (train/val/test).
    - Loads .npy files:
        latent maps: {data_dir}/deepfusion/latent-maps/{stage}/{patient_id}/{session_id}/{patient_id}_{session_id}_latent-maps.npy
        gradients:   {data_dir}/deepfusion/latent-maps/{stage}/{patient_id}/{session_id}/{patient_id}_{session_id}_grads.npy
    - Returns (X, G) as torch.float32, and label if task != "mdm".
    """

    def __init__(self, data_dir: str, stage: str = "data", task: str = "mdm"):
        self.data_dir = Path(data_dir)
        self.stage = stage
        self.task = task
        self.metadata = pd.read_csv(self.data_dir / "meta_data.csv")
        self.files = sorted(
            (self.data_dir / "deepfusion" / "latent-maps" / self.stage).glob("**/*_latent-maps.npy")
        )
        if self.task != "pretraining":
            # Precompute label mapping for fast lookup
            label_col = "cdr" if self.task.endswith("cdr") else self.task
            self.label_map = {
                (str(row["patient_id"]), str(row["session_id"])): map_label(self.task, row[label_col])
                for _, row in self.metadata.iterrows()
            }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        latent_path = self.files[idx]
        parts = latent_path.stem.split("_")
        patient_id, session_id = parts[0], parts[1]

        # Load latent maps and gradients
        x = np.load(latent_path)  # shape: (L, C, D, H, W)
        grad_path = latent_path.parent / f"{patient_id}_{session_id}_grads.npy"
        g = np.load(grad_path)    # shape: (L, 4)

        if self.task != "pretraining":
            # Fast label lookup
            num_label = self.label_map[(patient_id, session_id)]
            if isinstance(num_label, int):
                y = torch.tensor(num_label).long()   # classification
            else:
                y = torch.tensor(num_label).float()  # regression
            return torch.from_numpy(x).float(), torch.from_numpy(g).float(), y

        return torch.from_numpy(x).float(), torch.from_numpy(g).float()
    
# if __name__ == "__main__":
#     dataset = DeepFusionDataset(data_dir="data", stage="train")
#     print(f"Dataset size: {len(dataset)}")
#     x, g = dataset[0]
#     print(f"x: {x.shape}, g: {g.shape}")