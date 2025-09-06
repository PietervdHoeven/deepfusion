from pathlib import Path
from torch.utils.data import Dataset
from deepfusion.utils.labels import map_label
from collections import Counter
import pandas as pd
import numpy as np

def compute_sample_weights(dataset):
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
