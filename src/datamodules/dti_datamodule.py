# src/datamodules/dti_datamodule.py
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from src.datasets.dti_dataset import DTI_Dataset
from src.utils.sampler import compute_sample_weights

class DTI_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        task: str = "tri_cdr",
        batch_size: int = 32,
        use_sampler: bool = True,
        num_workers: int = 12,
        pin_memory: bool = True,
        test_run: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.task = task
        self.batch_size = batch_size
        self.use_sampler = use_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test_run = test_run

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_sampler = None  # only for train

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "validate"):
            self.train_dataset = DTI_Dataset(data_dir=self.data_dir, split="train", task=self.task)
            self.val_dataset   = DTI_Dataset(data_dir=self.data_dir, split="val",   task=self.task)

            if self.test_run:
                self.train_dataset = Subset(self.train_dataset, np.arange(0, 100))

            if self.use_sampler:
                weights = compute_sample_weights(self.data_dir, self.task)  # aligned to train_dataset
                self.train_sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=True,
                )
            else:
                self.train_sampler = None

        if stage in (None, "test"):
            self.test_dataset = DTI_Dataset(data_dir=self.data_dir, split="test", task=self.task)

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
            persistent_workers=(self.num_workers > 0),
        )

    def train_dataloader(self) -> DataLoader:
        # shuffle only when no sampler
        return self._dl(self.train_dataset, sampler=self.train_sampler, shuffle=(self.train_sampler is None))

    def val_dataloader(self) -> DataLoader:
        return self._dl(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dl(self.test_dataset, shuffle=False)