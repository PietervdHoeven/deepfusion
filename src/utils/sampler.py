from pathlib import Path
from src.utils.labels import map_label
from collections import Counter
import pandas as pd

def compute_sample_weights(
        data_dir: str,
        task: str
        ):
    # load the metadata and drop anything that isn training data
    meta_df = pd.read_csv(Path(data_dir) / "meta_data.csv")
    meta_df = meta_df[meta_df["stage"] == "train"].reset_index(drop=True)

    # remove cdr prefix
    label_col = "cdr" if task[-3:] == "cdr" else task

    # map labels to numerical labels
    raw_labels = meta_df[label_col].values
    num_labels = [map_label(task, label) for label in raw_labels]

    # count labels
    label_counts = Counter(num_labels)

    # Build weights
    weights = [1.0 / label_counts[label] for label in num_labels]

    print(len(weights), "weights computed for task", task)

    return weights