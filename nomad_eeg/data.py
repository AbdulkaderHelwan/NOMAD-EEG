"""Dataset folder format read by the scripts.

    my_dataset/
      meta.json          {"channels": [...], "classes": [...], "fs": 200, "n_samples": T}
      train_X.npy        [N, E, T] preprocessed signals (see preprocessing.py)
      train_y.npy        [N] integer labels
      train_subjects.npy [N] subject ids (optional)
      val_*.npy, test_*.npy

scripts/prepare_dataset.py writes this folder from raw arrays.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test")


def load_meta(root: str | Path) -> dict:
    return json.loads((Path(root) / "meta.json").read_text())


def load_split(root: str | Path, split: str, mmap: bool = True):
    root = Path(root)
    X = np.load(root / f"{split}_X.npy", mmap_mode="r" if mmap else None)
    y = np.load(root / f"{split}_y.npy").astype(np.int64)
    return X, y


class EEGSplit(Dataset):
    def __init__(self, root: str | Path, split: str):
        self.X, self.y = load_split(root, split)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(self.X[i], dtype=np.float32)), int(self.y[i])
