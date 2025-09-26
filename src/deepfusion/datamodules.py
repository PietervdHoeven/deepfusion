# src/datamodules.py
from typing import Optional
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# swap these to your actual imports
from deepfusion.datasets import (
    BaselineDataset,
    AutoencoderDataset,
    TransformerDataset,
)
from deepfusion.utils.samplers import (
    compute_classifier_sampler_weights,
    compute_qspace_sampler_weights,
    compute_patient_sampler_weights,
)
# only used by TransformerDataModule
from deepfusion.utils.collate import collate_transformer


# -----------------------
# Base: shared utilities
# -----------------------
class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        data_dir: str = "data",
        batch_size: int = 32,
        use_sampler: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
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

    def _dl(self, dataset, *, sampler=None, shuffle=False, collate_fn=None) -> DataLoader:
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
            collate_fn=collate_fn,
        )

    # Default loaders (modules may override to inject collate_fn)
    def train_dataloader(self) -> DataLoader:
        return self._dl(self.train_dataset, sampler=self.train_sampler, shuffle=(self.train_sampler is None))

    def val_dataloader(self) -> DataLoader:
        return self._dl(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dl(self.test_dataset, shuffle=False)


# -----------------------
# Baseline (scalar maps)
# -----------------------
class BaselineDataModule(DataModule):
    def __init__(
        self,
        data_dir: str = "data",
        task: str = "tri_cdr",
        batch_size: int = 32,
        use_sampler: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int | None = None,
        use_subset: bool = False,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            use_sampler=use_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            use_subset=use_subset,
        )
        self.task = task

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "validate"):
            self.train_dataset = BaselineDataset(data_dir=self.data_dir, stage="train", task=self.task)
            self.val_dataset   = BaselineDataset(data_dir=self.data_dir, stage="val",   task=self.task)

            if self.use_sampler:
                weights = compute_classifier_sampler_weights(self.train_dataset)
                self.train_sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights, num_samples=len(weights), replacement=True
                )
            else:
                self.train_sampler = None

        if stage in (None, "test"):
            self.test_dataset = BaselineDataset(data_dir=self.data_dir, stage="test", task=self.task)


# -----------------------
# Autoencoder (latents I/O)
# -----------------------
class AutoencoderDataModule(DataModule):
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
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            use_sampler=use_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            use_subset=use_subset,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "validate"):
            self.train_dataset = AutoencoderDataset(data_dir=self.data_dir, stage="train")
            self.val_dataset   = AutoencoderDataset(data_dir=self.data_dir, stage="val")

            if self.use_sampler:
                weights = compute_qspace_sampler_weights(self.train_dataset)
                self.train_sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights, num_samples=len(weights), replacement=True
                )
            else:
                self.train_sampler = None

        if stage in (None, "test"):
            self.test_dataset = AutoencoderDataset(data_dir=self.data_dir, stage="test")


# -----------------------
# Transformer (latents + grads, with masking)
# -----------------------
class TransformerDataModule(DataModule):
    def __init__(
        self,
        data_dir: str = "data",
        task: str = "pretraining",       # "pretraining" or downstream, e.g. "age", "gender", "tri_cdr"
        batch_size: int = 32,
        use_sampler: bool = True,        # only meaningful for downstream classification
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int | None = None,
        use_subset: bool = False,

        # masking controls
        mask_ratio_train: float = 0.3,
        mask_ratio_val: float = 0.3,
        mask_ratio_test: float = 0.0,    # 0.0 for eval
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            use_sampler=use_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            use_subset=use_subset,
        )
        self.task = task
        self.mask_ratio_train = mask_ratio_train
        self.mask_ratio_val   = mask_ratio_val
        self.mask_ratio_test  = mask_ratio_test

        # built in setup
        self._collate_train = None
        self._collate_val = None
        self._collate_test = None

    def setup(self, stage: Optional[str] = None) -> None:
        finetune = (self.task != "pretraining")
        # if you changed collate to allow 0 masks, we can just pass 0.0 for finetune
        r_train = 0.0 if finetune else self.mask_ratio_train
        r_val   = 0.0 if finetune else self.mask_ratio_val
        r_test  = 0.0 if finetune else self.mask_ratio_test

        if stage in (None, "fit", "validate"):
            self.train_dataset = TransformerDataset(self.data_dir, stage="train", task=self.task)
            self.val_dataset   = TransformerDataset(self.data_dir, stage="val",   task=self.task)

            # downstream sampling by patient (plug your own as needed)
            if self.use_sampler and finetune:
                weights = compute_patient_sampler_weights(self.train_dataset)
                self.train_sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights, num_samples=len(weights), replacement=True
                )
            elif self.use_sampler and not finetune:
                weights = compute_classifier_sampler_weights(self.train_dataset)
                self.train_sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights, num_samples=len(weights), replacement=True
                )
            else:
                self.train_sampler = None

            # collate with chosen mask ratios
            self._collate_train = lambda b: collate_transformer(b, mask_ratio=r_train)
            self._collate_val   = lambda b: collate_transformer(b, mask_ratio=r_val)

        if stage in (None, "test"):
            self.test_dataset = TransformerDataset(self.data_dir, stage="test", task=self.task)
            self._collate_test = lambda b: collate_transformer(b, mask_ratio=r_test)

    # override to inject collate_fns
    def train_dataloader(self) -> DataLoader:
        return self._dl(
            self.train_dataset,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            collate_fn=self._collate_train,
        )

    def val_dataloader(self) -> DataLoader:
        return self._dl(self.val_dataset, shuffle=False, collate_fn=self._collate_val)

    def test_dataloader(self) -> DataLoader:
        return self._dl(self.test_dataset, shuffle=False, collate_fn=self._collate_test)


# test run transformer

if __name__ == "__main__":
    dm = TransformerDataModule(data_dir="data", task="pretraining", batch_size=4, use_sampler=False, num_workers=4)
    dm.setup("fit")
    print(f"Train dataset size: {len(dm.train_dataset)}")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    for batch in train_loader:
        x, g, dir_mask, pad_mask = batch
        print(f"x: {x.shape}, g: {g.shape}, dir_mask: {dir_mask.shape}, pad_mask: {pad_mask.shape}")
        break
    for batch in val_loader:
        x, g, dir_mask, pad_mask = batch
        print(f"x: {x.shape}, g: {g.shape}, dir_mask: {dir_mask.shape}, pad_mask: {pad_mask.shape}")
        break

