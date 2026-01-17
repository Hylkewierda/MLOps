from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class PCAMDataset(Dataset):
    def __init__(self, x_path: str, y_path: str):
        self.x_path = x_path
        self.y_path = y_path

        # We openen hier NIET de bestanden (lazy loading)
        with h5py.File(self.x_path, "r") as f:
            self.length = f["x"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Lazy open per access (Snellius-safe)
        with h5py.File(self.x_path, "r") as fx:
            x = fx["x"][idx]  # (96, 96, 3), uint8

        with h5py.File(self.y_path, "r") as fy:
            y = fy["y"][idx]  # scalar

        # Convert to torch
        x = torch.from_numpy(x).float()

        # Normalize to [0, 1]
        x = x / 255.0
        x = torch.clamp(x, 0.0, 1.0)

        # Change shape to (C, H, W)
        x = x.permute(2, 0, 1)

        y = torch.tensor(y, dtype=torch.long).squeeze()


        return x, y

