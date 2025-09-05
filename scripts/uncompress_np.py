from pathlib import Path
import numpy as np
from tqdm import tqdm

root = Path("data/")  # change to your root directory
print(root)

for npz_path in tqdm(root.rglob("*.npz")):
    print("Rewriting:", npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        # unpack dict into savez
        np.savez(npz_path, **{k: data[k] for k in data.files})
