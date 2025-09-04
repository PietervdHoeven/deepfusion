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
        self.files = sorted((self.data_dir / "dti_maps" / self.stage).glob("**/*.npz"))

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
            pid = str(data["patient"])
            sid = str(data["session"])

            # lookup label in metadata
            label_col = "cdr" if self.task[-3:] == "cdr" else self.task
            label = self.metadata.loc[
                (self.metadata["patient"] == pid) &
                (self.metadata["session"] == sid),
                label_col
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
