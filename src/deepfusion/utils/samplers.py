from pathlib import Path
from deepfusion.utils.labels import map_label
from collections import Counter
import pandas as pd
import numpy as np

def compute_classifier_sampler_weights(dataset):
    """
    Returns weights aligned to dataset.files order using inverse class frequency.
    Assumes meta_data.csv has columns: ['patient','session','stage', <label_col>].
    Assumes dataset.files filenames are '../p_id/s_id/{p_id}_{s_id}_{filetype}.npz'.
    """
    meta_df = pd.read_csv(Path(dataset.data_dir) / "meta_data.csv")
    meta_df = meta_df[meta_df["stage"] == "train"]

    label_col = "cdr" if dataset.task.endswith("cdr") else dataset.task

    # Build fast lookup: (patient, session) -> raw_label
    idx = {(r.patient_id, r.session_id): r[label_col] for _, r in meta_df.iterrows()}

    # Collect numeric labels in dataset order
    raw_labels = []
    for file in dataset.files:
        file_name = Path(file).name  # get only the filename
        patient_id, session_id = file_name.split("_", 2)[:2]  # 'sub-..', 'ses-..'
        raw_label = idx.get((patient_id, session_id))
        raw_labels.append(raw_label)

    num_labels = [map_label(dataset.task, label) for label in raw_labels]


    if dataset.task == "age":
        # Bin ages into 5 bins
        num_labels_arr = np.array(num_labels)
        bins = np.linspace(num_labels_arr.min(), num_labels_arr.max(), 6)
        bin_indices = np.digitize(num_labels_arr, bins, right=False) - 1  # bins: 0-4
        counts = Counter(bin_indices)
        weights = [1.0 / counts[bin_idx] for bin_idx in bin_indices]
    else:
        counts = Counter(num_labels)
        weights = [1.0 / counts[label] for label in num_labels]


    print(f"{len(weights)} weights computed for task {dataset.task} (counts={dict(counts)})")
    return weights


# --- Bucketing: map a scalar b-value -> one of 6 buckets (int labels) ---
# Buckets chosen from your distribution:
#   0) b0                 : exactly 0
#   1) low                : (0, 400]
#   2) mid                : (400, 900]
#   3) b1000 (dominant)   : (900, 1100]
#   4) high               : (1100, 1500]
#   5) very_high          : (1500, +inf)
# We encode them as integers 0..5 so we can count quickly.
def bucketize_bvals(bvals: pd.Series) -> pd.Series:
    b = bvals.to_numpy()
    out = np.empty_like(b, dtype=np.int64)

    # exact b0
    mask_b0 = (b == 0)
    out[mask_b0] = 0

    # the rest by ranges
    mask_low       = (b >    0) & (b <=  400)
    mask_mid       = (b >  400) & (b <=  900)
    mask_b1000     = (b >  900) & (b <= 1100)
    mask_high      = (b > 1100) & (b <= 1500)
    mask_veryhigh  = (b > 1500)

    out[mask_low]      = 1
    out[mask_mid]      = 2
    out[mask_b1000]    = 3
    out[mask_high]     = 4
    out[mask_veryhigh] = 5

    return pd.Series(out, index=bvals.index)


def compute_qspace_sampler_weights(
        dataset,
        alpha: float = 0.5,  # shell tempering exponent     (Don't want to overfit on rare shells too much)
        beta:  float = 1.0,  # patient balancing exponent
        gamma: float = 0.1,  # session  balancing exponent  (sessions with just 1 volume explode weights if gamma=1.0)
        ) -> np.ndarray:
    """
    Build a WeightedRandomSampler using the dataset.manifest (one row per volume).

    Math (per-sample weight w_i):
        Let:
          p(i)      = patient for sample i
          s(i)      = session for sample i
          b(i)      = bucket for sample i  (from bucketize_bvals)
          |S_p|     = #sessions for patient p
          |V_p,s|   = #volumes in (patient p, session s)
          N_b       = #volumes in bucket b (global)

        We set:
            w_i = ( |S_{p(i)}| )^{-beta}        # downweight patients with many sessions
                * ( |V_{p(i), s(i)}| )^{-gamma} # downweight long sessions
                * ( N_{b(i)} )^{-alpha}         # downweight frequent b-shells (tempered)
    """
    df = dataset.manifest.reset_index(drop=True).copy()

    # Add bucket column (0..5) following the scheme above
    df["b_bucket"] = bucketize_bvals(df["bval"])

    # Precompute counts used in the weight formula
    # |S_p| : number of sessions per patient
    sess_per_patient = df.groupby("patient_id")["session_id"].nunique() # series: index=patient_id, value=#sessions
    # |V_{p,s}| : volumes per (patient, session)
    vols_per_session = df.groupby(["patient_id", "session_id"]).size()  # series: index=(patient_id,session_id), value=#volumes
    # N_b : volumes per bucket (global)
    vols_per_bucket  = df["b_bucket"].value_counts()                    # series: index=b_bucket, value=#volumes

    # Map counts to each sample in df order
    Sp = sess_per_patient.reindex(df["patient_id"]).to_numpy(dtype=np.float64)  # [num_samples] containing |S_p(i)|
    Vps = vols_per_session.reindex(pd.MultiIndex.from_frame(df[["patient_id","session_id"]])).to_numpy(dtype=np.float64)    # [num_samples] containing |V_{p(i), s(i)}|
    Nb = vols_per_bucket.reindex(df["b_bucket"]).to_numpy(dtype=np.float64)     # [num_samples] containing |N_{b(i)}|

    # Apply the exponents (note: x**(-k) == 1 / (x**k))
    weights = (Sp ** (-beta)) * (Vps ** (-gamma)) * (Nb ** (-alpha))

    weights = weights / weights.mean()  # normalize to mean=1

    return weights


def compute_patient_sampler_weights(dataset):
    meta_df = dataset.metadata
    meta_df = meta_df[meta_df["stage"] == "train"]

    # Compute class counts
    counts = Counter(meta_df["patient_id"])

    weights = []
    for file in dataset.files:
        file_name = Path(file).name  # get only the filename
        patient_id = file_name.split("_", 1)[0]  # 'sub-..'
        weights.append(1.0 / counts[patient_id])

    return weights