# src/utils/labels.py
MAPPERS = {
    "gender":     {"male": 0, "female": 1},
    "handedness": {"right": 0, "left": 1},

    # CDR variants
    "bin_cdr": {0.0: 0, 0.5: 1, 1.0: 1, 2.0: 1},
    "tri_cdr": {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 2},
    "ord_cdr": {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3},

    # Regression
    "age": None,
}

def map_label(task: str, label):
    mapper = MAPPERS[task]
    if mapper is None:
        num_label = float(label)
    else:
        num_label = int(mapper[label])
    return num_label
