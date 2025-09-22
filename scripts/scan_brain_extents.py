# scan_brain_extents.py
#
#  • Uses every CPU core (n_jobs = -1   ⇒ all cores)
#  • Loads ⇢ computes extents ⇢ returns a (len_x,len_y,len_z) triple
#  • Aggregates all triples to find the max per axis

import os, sys, glob
import numpy as np
import nibabel as nib
from joblib import Parallel, delayed
from tqdm import tqdm

def brain_extents(nii_path: str):
    """Return (len_x, len_y, len_z) for a single NIfTI file."""
    vol  = nib.load(nii_path).get_fdata(dtype=np.float32)      # lazy mmap read
    if vol.ndim > 3:
        vol = vol[..., 0]
    mask = vol > 0
    if not mask.any():
        return (0, 0, 0)

    coords  = np.argwhere(mask)
    mins    = coords.min(axis=0)
    maxs    = coords.max(axis=0)
    return tuple((maxs - mins + 1).tolist())

def main(data_dir, n_jobs=-1):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*allruns.nii.gz"),
                             recursive=True))
    if not files:
        sys.exit(f"No NIfTI files found in {data_dir}")

    # Parallel computation with live progress bar
    extents = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(brain_extents)(f)
        for f in tqdm(files, desc="Scanning volumes", unit="vol")
    )

    # extents is a list of (x,y,z); convert to array for easy max()
    extents_arr              = np.asarray(extents, dtype=np.int32)
    print("\nPer-volume brain extents (len_x, len_y, len_z):")
    print(extents_arr)
    max_len_x, max_len_y, max_len_z = extents_arr.max(axis=0)

    print("\nLargest per-axis brain extents:")
    print(f"  X (left-right)      : {max_len_x} voxels")
    print(f"  Y (ant.-post.)      : {max_len_y} voxels")
    print(f"  Z (inf.-sup.)       : {max_len_z} voxels")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python scan_brain_extents_mp.py /path/to/cleaned_dmri [n_jobs]")
    data_root = sys.argv[1]
    n_jobs    = int(sys.argv[2]) if len(sys.argv) == 3 else -1   # -1 → all cores
    main(data_root, n_jobs)
