import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np

class AEDataset(Dataset):
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
        self.manifest = pd.read_csv(self.data_dir / "deepfusion" / "manifest.csv")
        self.manifest = self.manifest[self.manifest["stage"] == self.stage]

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx: int):
        # get the volume path and index
        row = self.manifest.iloc[idx]
        dwi_path = row["dwi_path"]
        g = int(row["g_idx"])

        # memmap the session file and read one volume
        arr = np.load(dwi_path, mmap_mode="r")      # shape [N,D,H,W]
        vol = arr[g]                                # view [D,H,W], zero-copy on CPU

        x = torch.from_numpy(vol).unsqueeze(0)      # [1,D,H,W], dtype matches on-disk (likely float16)

        return x


# Test run

# if __name__ == "__main__":
#     dataset = AEDataset(data_dir="data", stage="train")
#     print(f"Dataset length: {len(dataset)}")
#     sample = dataset[0]
#     print(f"Sample shape: {sample.shape}, dtype: {sample.dtype}")