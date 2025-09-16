# src/deepfusion/datamodules/deepfusion_datamodule.py
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from deepfusion.utils.samplers import compute_patient_sampler_weights, compute_classifier_sampler_weights
from deepfusion.utils.collate import collate_fn

from deepfusion.datasets.deepfusion_dataset import DeepFusionDataset

class DeepFusionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        task: str = "pretraining",
        batch_size: int = 32,
        use_sampler: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        use_subset: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.task = task
        self.batch_size = batch_size
        self.use_sampler = use_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.use_subset = use_subset

        if self.task == "pretraining":
            self.sampler_fn = compute_patient_sampler_weights
        else:
            self.sampler_fn = compute_classifier_sampler_weights

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_sampler = None  # only for train

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "validate"):
            self.train_dataset = DeepFusionDataset(data_dir=self.data_dir, stage="train", task=self.task)
            self.val_dataset = DeepFusionDataset(data_dir=self.data_dir, stage="val", task=self.task)

            # NOTE: This is fragile! Only use for quick testing and you are sure that meta_data.csv and files are a perfect match.
            if self.use_subset:
                self.train_dataset = Subset(self.train_dataset, np.arange(0, 100))      
                self.train_dataset.files = self.train_dataset.dataset.files[:100]
                self.train_dataset.data_dir = self.train_dataset.dataset.data_dir
                self.train_dataset.task = self.train_dataset.dataset.task
                self.train_dataset.metadata = self.train_dataset.dataset.metadata
                self.val_dataset = Subset(self.val_dataset, np.arange(0, 20))      
                self.val_dataset.files = self.val_dataset.dataset.files[:20]
                self.val_dataset.data_dir = self.val_dataset.dataset.data_dir
                self.val_dataset.task = self.val_dataset.dataset.task
                self.val_dataset.metadata = self.val_dataset.dataset.metadata


            if self.use_sampler:
                weights = self.sampler_fn(self.train_dataset)
                self.train_sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=True,
                )
            else:
                self.train_sampler = None

        if stage in (None, "test"):
            self.test_dataset = DeepFusionDataset(data_dir=self.data_dir, stage="test", task=self.task)

    def _dl(self, dataset, *, sampler=None, shuffle=False) -> DataLoader:
        # If sampler is provided, shuffle must be False.
        if sampler is not None:
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=(self.num_workers > 0)
        )

    def train_dataloader(self) -> DataLoader:
        # shuffle only when no sampler
        return self._dl(self.train_dataset, sampler=self.train_sampler, shuffle=(self.train_sampler is None))

    def val_dataloader(self) -> DataLoader:
        return self._dl(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dl(self.test_dataset, shuffle=False)