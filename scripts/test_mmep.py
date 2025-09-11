import numpy as np
from pathlib import Path

# point this to one of your processed sessions
session_dir = Path("data/normalised_dwi_session/train/sub-OAS30001/ses-d2430")

dwi = np.load(session_dir / "sub-OAS30001_ses-d2430_normalised-dwi.npy", mmap_mode="r")     # [N,D,H,W]
grads = np.load(session_dir / "sub-OAS30001_ses-d2430_grads.npy", mmap_mode="r") # [N,4]

print("DWI shape:", dwi.shape)
print("Grads shape:", grads.shape)

# Check correspondence for a random volume
g = 3
vol = dwi[g]                # [D,H,W]
bval, bx, by, bz = grads[g] # scalars

print(f"Volume {g}: shape={vol.shape}")
print(f"  bval={bval:.1f}, bvec=({bx:.3f}, {by:.3f}, {bz:.3f})")
print(f" file_size={vol.nbytes/1e6:.1f} MB, mean={vol.mean():.3f}, std={vol.std():.3f}")

# A couple of sanity checks
print("Number of volumes matches grads:", dwi.shape[0] == grads.shape[0])
print("Unique bvals (rounded):", np.unique(np.round(grads[:,0], -2)))