from pathlib import Path
from torch.utils.data import Dataset
from src.utils.labels import map_label
from collections import Counter
import pandas as pd

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
    idx = {(r.patient, r.session): r[label_col] for _, r in meta_df.iterrows()}

    # Collect numeric labels in dataset order
    raw_labels = []
    for file in dataset.files:
        file_name = Path(file).name  # get only the filename
        p_id, s_id = file_name.split("_", 2)[:2]  # 'sub-..', 'ses-..'
        raw_label = idx.get((p_id, s_id))
        raw_labels.append(raw_label)

    num_labels = [map_label(dataset.task, label) for label in raw_labels]
    counts = Counter(num_labels)
    weights = [1.0 / counts[label] for label in num_labels]

    print(f"{len(weights)} weights computed for task {dataset.task} (counts={dict(counts)})")
    return weights
