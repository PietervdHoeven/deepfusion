# --- Minimal, readable universal split generator ---
# Assumptions:
# - Your CSV has at least: patient, session, cdr, gender, handedness, age
# - No NaNs, types are already clean
# - We want ONE robust split for both classification & regression tasks
# - Strategy: Stratify on a COMPOSITE label (e.g., "cdr|gender|age_bin")


import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
import os


from dipy.io.gradients import read_bvals_bvecs # type: ignore
from dipy.core.gradients import gradient_table # type: ignore
from dipy.reconst.dti import TensorModel # type: ignore


def split_and_save(
    csv_path,
    group_key="patient",
    folds=10,
    val_folds=1,
    test_folds=1,
    seed=42,
    class_cols=["cdr", "gender"],
    n_reg_bins=5,
    map_out="data/patient_split_map.csv"
):
    df = pd.read_csv(csv_path)

    # Build composite stratification label
    parts = [df[col].astype(str) for col in class_cols]
    binned = pd.qcut(df["age"].astype(float), q=n_reg_bins, duplicates="drop")
    parts.append(binned.astype(str))
    composite = pd.Series(["|".join(t) for t in zip(*parts)], index=df.index)
    strat_y, _ = pd.factorize(composite, sort=True)

    groups = df[group_key].astype(str).values
    X_dummy = np.zeros(len(df))

    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    split_iterator = sgkf.split(X_dummy, strat_y, groups)

    # Get val/test indices
    val_indices = []
    test_indices = []
    for i in range(val_folds):
        _, idx = next(split_iterator)
        val_indices.extend(idx)
    for i in range(test_folds):
        _, idx = next(split_iterator)
        test_indices.extend(idx)

    split = np.full(len(df), "train", dtype=object)
    split[val_indices] = "val"
    split[test_indices] = "test"
    df["split"] = split

    # Save to same CSV
    df.to_csv(csv_path, index=False)

    # Patient -> split map
    patient_map = df[[group_key, "split"]].drop_duplicates().sort_values(group_key)
    patient_map.to_csv(map_out, index=False)

    print("Done.")
    print("Wrote:", csv_path)
    print("Wrote:", map_out)
    print(df["split"].value_counts())
    print("Unique patients per split:")
    print(df.groupby("split")[group_key].nunique())


def process_session(
        dwi_path: str,
        bval_path: str,
        bvec_path: str,
        metadata: pd.DataFrame
):
    """
    Load once, normalize DWI, save per-session NPZ,
    compute DTI maps from the same arrays, save NIfTIs.
    Places outputs under {root}/{split}/{patient}/{session}/...
    """
    # Infer patient/session from filename (or pass them in explicitly if you prefer)
    base = Path(dwi_path).with_suffix("").with_suffix("")  # strip .nii.gz
    name = base.name

    # Expect names like sub-XXXX_ses-YY_*; tweak to your convention
    parts = name.split("_")
    sub = next((p for p in parts if p.startswith("sub-")), None)
    ses = next((p for p in parts if p.startswith("ses-")), None)
    patient_id = sub
    session_id = ses

    row = metadata[(metadata["patient"] == patient_id) & (metadata["session"] == session_id)]
    split = row.iloc[0]["split"]

    dwi, bvals, bvecs, affine, header = load_session_data(dwi_path, bval_path, bvec_path)

    dwi_normalised = normalise_dwi(dwi, bvals)

    dwi_norm_path = f"data/normalised_dwi/{split}/{patient_id}/{session_id}/{patient_id}_{session_id}_normalised-dwi.npz"
    os.makedirs(os.path.dirname(dwi_norm_path), exist_ok=True)
    np.savez_compressed(
        dwi_norm_path,
        dwi=dwi_normalised,
        patient_id=patient_id,
        session_id=session_id,
        bvals=bvals,
        bvecs=bvecs
    )

    dti_scalar_maps = compute_dti_metrics(dwi, bvals, bvecs)

    dti_scalar_maps_path = f"data/dti_maps/{split}/{patient_id}/{session_id}/{patient_id}_{session_id}_dti-scalar-maps.npz"
    os.makedirs(os.path.dirname(dti_scalar_maps_path), exist_ok=True)
    np.savez_compressed(
        dti_scalar_maps_path,
        **dti_scalar_maps,
        patient_id=patient_id,
        session_id=session_id,
        bvals=bvals,
        bvecs=bvecs
    )


def load_session_data(dwi_path: str, bval_path: str, bvec_path: str):
    """
    Load the session's 4D DWI + gradients once.

    Returns
    -------
    dwi : np.ndarray, shape (X, Y, Z, N), float32
    bvals : np.ndarray, shape (N,), float32
    bvecs : np.ndarray, shape (3, N), float32
    affine : np.ndarray, shape (4, 4), float32
    header : nib.Nifti1Header
    """
    dwi_img = nib.load(dwi_path)
    dwi = dwi_img.get_fdata(dtype=np.float32)           # (X,Y,Z,N)
    affine = dwi_img.affine.astype(np.float32)
    header = dwi_img.header.copy()

    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    bvals = np.asarray(bvals, dtype=np.float32)         # (N,)
    bvecs = np.asarray(bvecs, dtype=np.float32)         # (3,N)

    return dwi, bvals, bvecs, affine, header


def normalise_dwi(dwi_data: np.ndarray, bvals: np.ndarray) -> np.ndarray:
    """
    Load a 4-D DWI series and apply robust, session-level gain and z-score normalisation.

    Parameters
    ----------
    dwi_path : Path
        Path to the cleaned 4-D DWI NIfTI (shape [X,Y,Z,N]).
    bval_path : Path
        Path to the matching .bval file (one row of N b-values).

    Returns
    -------
    dwi_norm : ndarray (float32)
        The fully normalised DWI data (same shape as input).
    """

    # 1) Identify all b0 volumes (b-value == 0)
    b0_indices  = np.where(bvals == 0)[0]         # e.g. array([0, 10, 20])

    # 2) Extract those b0 volumes and form a union-mask of nonzero voxels
    #    (handles slight mis-alignments: if any run has signal, we treat it as brain)
    b0_volumes = np.take(dwi_data, b0_indices, axis=3)  # shape (X, Y, Z, N_b0)
    mask       = np.any(b0_volumes > 0, axis=3)         # boolean mask [X,Y,Z]

    # 3) Remove any stray zero-intensity voxels inside the union mask
    #    (e.g. holes due to warping) before computing the gain
    b0_values  = b0_volumes[mask].ravel()
    b0_values  = b0_values[b0_values > 0]               # drop zeros

    # 4) SESSION-GAIN NORMALISATION:
    #    Anchor the median of all b0 tissue intensities to 1.0,
    #    removing scanner-/coil-level scale differences across sessions
    gain       = 1.0 / (np.median(b0_values) + 1e-12)
    dwi_scaled = dwi_data * gain

    # 5) SESSION-WIDE Z-SCORE NORMALISATION:
    #    Compute mean/std across all brain voxels in all volumes,
    #    giving zero-mean/unit-variance inputs for the autoencoder,
    #    yet preserving relative shell attenuation patterns.
    dwi_values = dwi_scaled[mask, ...].ravel()
    dwi_values = dwi_values[dwi_values > 0]              # drop any sneaky zeros
    mean       = dwi_values.mean()
    std        = dwi_values.std() + 1e-6
    dwi_norm   = (dwi_scaled - mean) / std

    # 6) Return the normalised data
    return dwi_norm.astype(np.float32)


def compute_dti_metrics(
        dwi_data: np.ndarray,
        bvals: np.ndarray,
        bvecs: np.ndarray
): 
    # DIPY expects gradient table from bvals/bvecs
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)

    # lightweight mask to skip background
    mask = np.sum(dwi_data, axis=-1) > 0

    model = TensorModel(gtab)
    fit = model.fit(dwi_data, mask=mask)

    fa = np.nan_to_num(fit.fa.astype(np.float32))
    md = np.nan_to_num(fit.md.astype(np.float32))
    rd = np.nan_to_num(fit.rd.astype(np.float32))
    ad = np.nan_to_num(fit.ad.astype(np.float32))
    fa = np.clip(fa, 0, 1)

    out = {"fa": fa, "md": md, "rd": rd, "ad": ad}

    #plot_dti(out)

    return out


import matplotlib.pyplot as plt
import numpy as np

def plot_dti(out, z: int = 64):
    """
    Plot one axial slice (z index) from each DTI map in `out`.
    out = {"fa": 3D array, "md": 3D array, "rd": 3D array, "ad": 3D array}
    """
    keys = ["fa", "md", "rd", "ad"]
    fig, axes = plt.subplots(1, len(keys), figsize=(12, 3))

    for ax, k in zip(axes, keys):
        if k not in out:
            ax.axis("off")
            continue
        vol = out[k]
        slc = np.rot90(vol[:, :, z])  # rotate for display
        ax.imshow(slc, cmap="gray")
        ax.set_title(k.upper())
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    fig.savefig("dti.png")



def main():
    parser = argparse.ArgumentParser(description="Universal split generator for cleaned DWI dataset.")
    parser.add_argument('--metadata-csv', type=str, default="data/meta_data.csv", help='Path to meta_data.csv (in/out)')
    parser.add_argument('--group-key', type=str, default="patient", help='Column to group by (default: patient)')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds (default: 10)')
    parser.add_argument('--val-folds', type=int, default=1, help='Number of folds for validation (default: 1)')
    parser.add_argument('--test-folds', type=int, default=1, help='Number of folds for test (default: 1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--class-cols', nargs='+', default=["cdr", "gender", "handedness"], help='Columns for composite stratification label (default: cdr gender handedness)')
    parser.add_argument('--n-reg-bins', type=int, default=5, help='Number of quantile bins for regression column (default: 5)')
    parser.add_argument('--map-out', type=str, default="data/patient_split_map.csv", help='Path to patient split map output (default: data/patient_split_map.csv)')
    parser.add_argument('--skip-split', action='store_true', help='Skip splitting if split column already exists')

    args = parser.parse_args()
    if not args.skip_split:
        split_and_save(
            csv_path=args.metadata_csv,
            group_key=args.group_key,
            folds=args.folds,
            val_folds=args.val_folds,
            test_folds=args.test_folds,
            seed=args.seed,
            class_cols=args.class_cols,
            n_reg_bins=args.n_reg_bins,
            map_out=args.map_out
        )
    # Load metadata with splits
    metadata = pd.read_csv(args.metadata_csv)

    # loop over the metadata and process each patients data
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing sessions"):
        patient_id = row["patient"]
        session_id = row["session"]
        
        dwi_path = f"data/cleaned_dwi/{patient_id}/{session_id}/{patient_id}_{session_id}_dwi_allruns.nii.gz"
        bval_path = f"data/cleaned_dwi/{patient_id}/{session_id}/{patient_id}_{session_id}_dwi_allruns.bval"
        bvec_path = f"data/cleaned_dwi/{patient_id}/{session_id}/{patient_id}_{session_id}_dwi_allruns.bvec"

        try:
            process_session(
            dwi_path=dwi_path,
            bval_path=bval_path,
            bvec_path=bvec_path,
            metadata=metadata
            )
        except Exception as e:
            print(f"Error processing {patient_id}_{session_id}: {e}")
            # Move to next session (continue loop)
            continue

    



if __name__ == "__main__":
    main()
