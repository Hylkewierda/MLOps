from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 2)

    # ---- Transforms ----
    # (PCAMDataset already normalizes; transforms kept minimal)
    train_transform = transforms.Compose([])
    val_transform = transforms.Compose([])

    # ---- Paths ----
    train_x = base_path / "camelyonpatch_level_2_split_train_x.h5"
    train_y = base_path / "camelyonpatch_level_2_split_train_y.h5"
    val_x = base_path / "camelyonpatch_level_2_split_valid_x.h5"
    val_y = base_path / "camelyonpatch_level_2_split_valid_y.h5"

    # ---- Datasets ----
    train_dataset = PCAMDataset(train_x, train_y)
    val_dataset = PCAMDataset(val_x, val_y)

    # ---- WeightedRandomSampler (class imbalance) ----
    with h5py.File(train_y, "r") as f:
        labels = f["y"][:].reshape(-1)

    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader

