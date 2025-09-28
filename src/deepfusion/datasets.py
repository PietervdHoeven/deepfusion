# src/datasets.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from deepfusion.utils.labels import map_label

import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np

class BaselineDataset(Dataset):
    """
    Minimal Dataset for DTI scalar maps (FA, MD, RD, AD) saved as a single .npz per session.

    Directory layout (example):
      repo_root/
        data/
          meta_data.csv
          dti_maps/
            train/sub-XXXX/ses-YYYY/sub-XXXX_ses-YYYY_dti-scalar-maps.npz
            val/  ...
            test/ ...

    Each NPZ contains:
      {"fa": (D,H,W), "md": (D,H,W), "rd": (D,H,W), "ad": (D,H,W),
       "patient_id": "sub-XXXX", "session_id": "ses-YYYY", "bvals": (3,N), "bvecs": (3,N)}

    The dataset stacks FA/MD/RD/AD → (4, D, H, W) and uses `task` to select the label column in metadata.
    """
    def __init__(
            self,
            data_dir: str,
            stage: str,
            task: str
            ):
        """
        Parameters
        ----------
        data_dir : str | Path
            Path to repo_root/data/
        split : str
            One of {"train","val","test"} — selects subdir under data/dti_maps/
        task : str
            Name of the label column in metadata (e.g., "cdr", "age", ...).
        """
        self.data_dir = Path(data_dir)
        self.stage = stage
        self.task = task
        self.metadata = pd.read_csv(self.data_dir / "meta_data.csv")
        self.files = sorted((self.data_dir / "baselines" / self.stage).glob("**/*.npz"))

        # Precompute label mapping for fast lookup
        label_col = "cdr" if self.task[-3:] == "cdr" else self.task
        self.label_map = {
            (str(row["patient_id"]), str(row["session_id"])): map_label(self.task, row[label_col])
            for _, row in self.metadata.iterrows()
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        # get the .npy file path
        file_path = self.files[idx]

        # Load the DTI maps and metadata
        with np.load(file_path) as data:
            # Stack FA/MD/RD/AD -> (4, D, H, W)
            dti_maps = data["dti"]  # shape (4, D, H, W) NOTE: arr[0]=FA, arr[1]=MD, arr[2]=RD, arr[3]=AD
            patient_id = str(data["patient_id"])
            session_id = str(data["session_id"])

        # Fast label lookup
        num_label = self.label_map[(patient_id, session_id)]
        if type(num_label) == int:
            y = torch.tensor(num_label).long()  # classification
        else:
            y = torch.tensor(num_label).float() # regression

        # Convert dti_maps to PyTorch tensors
        x = torch.from_numpy(dti_maps).to(torch.float16)

        return x, y
    

class AutoencoderDataset(Dataset):
    """
    Minimal Dataset for AutoEncoder training on preprocessed DWI data.

    Directory layout (example):
      repo_root/
        data/
          deepfusion/
            train/sub-XXXX/ses-YYYY/sub-XXXX_ses-YYYY_normalised-dwi.npy
            val/  ...
            test/ ...
    """

    def __init__(self, data_dir: Path, stage: str):
        self.data_dir = Path(data_dir)
        self.stage = stage
        self.manifest = pd.read_csv(self.data_dir / "deepfusion/volumes" / "manifest.csv")
        self.manifest = self.manifest[self.manifest["stage"] == self.stage]
        self._mm_cache = {}

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx: int):
        # get the volume path and index
        row = self.manifest.iloc[idx]
        dwi_path = row["dwi_path"]
        g = int(row["g_idx"])

        # memmap the session file and read one volume
        dwi = np.load(dwi_path, mmap_mode='r')      # shape [N,D,H,W]
        vol = dwi[g]                       # view [D,H,W], zero-copy on CPU

        x = torch.from_numpy(vol).unsqueeze(0)      # [1,D,H,W]

        return x
    

class TransformerDataset(Dataset):
    """
    Minimal Dataset for 3D volumes saved as a single .npy per session.

    - Reads meta_data.csv for session metadata.
    - Filters by stage (train/val/test).
    - Loads .npy files:
        latent maps: {data_dir}/deepfusion/latents/{stage}/{patient_id}/{session_id}/{patient_id}_{session_id}_latent-maps.npy
        gradients:   {data_dir}/deepfusion/latents/{stage}/{patient_id}/{session_id}/{patient_id}_{session_id}_grads.npy
    - Returns (X, G) as torch.float32, and label if task != "pretraining".
    """

    def __init__(self, data_dir: str = "data", stage: str = "train", task: str = "pretraining"):
        self.data_dir = Path(data_dir)
        self.stage = stage
        self.task = task
        self.metadata = pd.read_csv(self.data_dir / "meta_data.csv")
        self.files = sorted(
            (self.data_dir / "deepfusion" / "latents" / self.stage).glob("**/*_latent-maps.npy")
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
        x = np.load(latent_path)  # shape: (Q, C, D, H, W)
        grad_path = latent_path.parent / f"{patient_id}_{session_id}_grads.npy"
        g = np.load(grad_path)    # shape: (Q, 4)

        if self.task != "pretraining":
            # Fast label lookup
            num_label = self.label_map[(patient_id, session_id)]
            if isinstance(num_label, int):
                y = torch.tensor(num_label).long()   # classification
            else:
                y = torch.tensor(num_label).float()  # regression
            return torch.from_numpy(x).float(), torch.from_numpy(g).float(), y

        return torch.from_numpy(x).float(), torch.from_numpy(g).float()
    
# print the mean and std of a dataset
def print_dataset_mean_std(data_dir="data", stage="train"):
    root = Path(data_dir) / "deepfusion" / "latents" / stage
    files = sorted(root.glob("**/*_latent-maps.npy"))
    if not files:
        raise FileNotFoundError(f"No latent .npy files under {root}")

    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0

    for f in files:
        x = np.load(f, mmap_mode="r")  # shape: (Q, C, D, H, W)
        if x.ndim != 5:
            raise ValueError(f"Unexpected shape in {f}: {x.shape}")
        total_sum   += np.sum(x, dtype=np.float64)
        total_sumsq += np.sum(np.square(x, dtype=np.float64), dtype=np.float64)
        total_count += x.size

    mean = float(total_sum / total_count)
    var  = float(total_sumsq / total_count - mean**2)
    std  = float(np.sqrt(max(var, 0.0)))

    print(f"[Latents:{stage}] mean={mean:.8f}, std={std:.8f}, n={total_count}")

if __name__ == "__main__":
    print_dataset_mean_std(data_dir="data", stage="train")