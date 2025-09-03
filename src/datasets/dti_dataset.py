# src/datasets/dti_dataset.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.labels import map_label

class DTI_Dataset(Dataset):
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
       "patient_id": "sub-XXXX", "session_id": "ses-YYYY", "bvals": ..., "bvecs": ...}

    The dataset stacks FA/MD/RD/AD → (4, D, H, W) and uses `task` to select the label column in metadata.
    """
    def __init__(
            self,
            data_dir: str,
            split: str,
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
        self.split = split
        self.task = task
        self.metadata = pd.read_csv(self.data_dir / "meta_data.csv")
        self.files = list((self.data_dir / "dti_maps" / self.split).glob("**/*.npz"))
        self.data = self.load_data()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        # get the .npz file path
        npz_file_path = self.files[idx]

        # Load the DTI maps and metadata
        with np.load(npz_file_path) as data:
            # Stack FA/MD/RD/AD -> (4, D, H, W)
            arr = np.stack([data["fa"], data["md"], data["rd"], data["ad"]], axis=0)

            # IDs for label lookup
            pid = str(data["patient_id"])
            sid = str(data["session_id"])

            # lookup label in metadata
            label = self.metadata.loc[
                (self.metadata["patient_id"] == pid) &
                (self.metadata["session_id"] == sid),
                self.task
            ].values

        # Cast labels to tensors
        num_label = map_label(self.task, label)
        if type(num_label) == int:
            y = torch.tensor(num_label).long()  # classification
        else:
            y = torch.tensor(num_label).float() # regression

        # Convert dti_maps to PyTorch tensors
        x = torch.from_numpy(arr).float()

        return x, y 
