import numpy as np

npz_path = "data/normalised_dwi/train/sub-OAS31000/ses-d0072/sub-OAS31000_ses-d0072_normalised-dwi.npz"

with np.load(npz_path) as data:
    print("Keys in the npz file:")
    print(list(data.keys()))
    print("\nShapes and dtypes of arrays:")
    for key in data:
        arr = data[key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")