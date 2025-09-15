import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
from typing import Optional
from deepfusion.datasets.ae_dataset import AEDataset
from deepfusion.utils.sampler import compute_AE_sampler_weights

class AE_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        use_sampler: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        use_subset: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_sampler = use_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.use_subset = use_subset

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_sampler = None  # only for train

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "validate"):
            self.train_dataset = AEDataset(data_dir=self.data_dir, stage="train")
            self.val_dataset   = AEDataset(data_dir=self.data_dir, stage="val")

            # NOTE: This is fragile! Only use for quick testing and you are sure that meta_data.csv and files are a perfect match.
            if self.use_subset:
                self.train_dataset = Subset(self.train_dataset, np.arange(0, 1000))      
                self.train_dataset.manifest = self.train_dataset.dataset.manifest[:1000]
                self.train_dataset.data_dir = self.train_dataset.dataset.data_dir
                self.val_dataset = Subset(self.val_dataset, np.arange(0, 200))
                self.val_dataset.manifest = self.val_dataset.dataset.manifest[:200]
                self.val_dataset.data_dir = self.val_dataset.dataset.data_dir

            if self.use_sampler:
                weights = compute_AE_sampler_weights(self.train_dataset)
                self.train_sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=True,
                )
            else:
                self.train_sampler = None

        if stage in (None, "test"):
            self.test_dataset = AEDataset(data_dir=self.data_dir, stage="test")

    def _dl(self, dataset, *, sampler=None, shuffle=False) -> DataLoader:
        # If sampler is provided, shuffle must be False.
        if sampler is not None:
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=(self.num_workers > 0),
        )

    def train_dataloader(self) -> DataLoader:
        # shuffle only when no sampler
        return self._dl(self.train_dataset, sampler=self.train_sampler, shuffle=(self.train_sampler is None))

    def val_dataloader(self) -> DataLoader:
        return self._dl(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dl(self.test_dataset, shuffle=False)
    

# test run
# import pandas as pd
# import numpy as np
# from deepfusion.utils.sampler import bucketize_bvals
# def check_sampler_distribution(dataset, weights, num_samples=20000):
#     """
#     Draw samples from a WeightedRandomSampler and compare empirical vs. expected
#     distributions for patients and b-value buckets.
#     """
#     from torch.utils.data import WeightedRandomSampler
#     import torch

#     # Build sampler
#     sampler = WeightedRandomSampler(
#         weights=torch.as_tensor(weights, dtype=torch.double),
#         num_samples=num_samples,
#         replacement=True,
#     )

#     # Draw indices
#     idxs = np.fromiter(iter(sampler), dtype=int, count=num_samples)
#     df = dataset.manifest.reset_index(drop=True)
#     df["b_bucket"] = bucketize_bvals(df["bval"])

#     # Empirical proportions
#     patient_emp = pd.Series(df.loc[idxs, "patient_id"]).value_counts(normalize=True)
#     shell_emp   = pd.Series(df.loc[idxs, "b_bucket"]).value_counts(normalize=True)

#     # Theoretical proportions (weights normalized by group)
#     df["_w"] = weights
#     patient_theo = df.groupby("patient_id")["_w"].sum()
#     patient_theo /= patient_theo.sum()
#     shell_theo = df.groupby("b_bucket")["_w"].sum()
#     shell_theo /= shell_theo.sum()

#     print("\n--- Patient distribution ---")
#     out = pd.DataFrame({
#         "empirical": patient_emp,
#         "theoretical": patient_theo
#     }).fillna(0).sort_values("theoretical", ascending=False)
#     print(out.head(20))  # print top 20 patients

#     print("\n--- Shell distribution ---")
#     out = pd.DataFrame({
#         "empirical": shell_emp,
#         "theoretical": shell_theo
#     }).fillna(0).sort_index()
#     print(out)

#     return out

# Slap under computing sampler weights in the datamodule for more testing:
# --- Easy checks ---
                # w = np.array(weights, dtype=np.float64)
                # print("\n[Sampler weights sanity check]")
                # print(f"Length: {len(w)} (should equal #train samples)")
                # print(f"Min: {w.min():.4e}, Max: {w.max():.4e}")
                # print(f"Mean: {w.mean():.4e}, Std: {w.std():.4e}")
                # print(f"10 smallest: {np.sort(w)[:10]}")
                # print(f"10 largest : {np.sort(w)[-10:]}")

                # # Rough check: are heavier weights associated with rarer b-buckets?
                # if hasattr(self.train_dataset, "manifest") and "b_bucket" in self.train_dataset.manifest:
                #     df = self.train_dataset.manifest.copy()
                #     df["_w"] = w
                #     bucket_means = df.groupby("b_bucket")["_w"].mean().sort_index()
                #     bucket_counts = df["b_bucket"].value_counts().sort_index()
                #     print("\nMean weight per bucket vs counts:")
                #     for b in bucket_means.index:
                #         print(f"Bucket {b}: count={bucket_counts[b]}, mean_w={bucket_means[b]:.4e}")

                # # Optional: quick histogram
                # try:
                #     import matplotlib.pyplot as plt
                #     plt.hist(w, bins=50)
                #     plt.title("Sampler weight distribution")
                #     plt.savefig("ae_sampler_weights_hist.png")
                #     plt.show()
                # except ImportError:
                #     pass

# if __name__ == "__main__":
#     dm = DTI_DataModule(data_dir="data", batch_size=8, use_sampler=True)
#     dm.setup("fit")
#     train_loader = dm.train_dataloader()
#     val_loader = dm.val_dataloader()

#     print(f"Train batches: {len(train_loader)}")
#     for batch in train_loader:
#         x = batch
#         print(f"Train batch x shape: {x.shape}, dtype: {x.dtype}")
#         break

#     print(f"Val batches: {len(val_loader)}")
#     for batch in val_loader:
#         x = batch
#         print(f"Val batch x shape: {x.shape}, dtype: {x.dtype}")
#         break

#     weights = compute_AE_sampler_weights(dm.train_dataset)
#     check_sampler_distribution(dm.train_dataset, weights, num_samples=20000)
