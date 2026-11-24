"""PyTorch Dataset + DataLoader factory reading the processed split CSV."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from cxr.config import all_labels
from cxr.data.io import read_xray
from cxr.data.transforms import build_transforms


class ChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, labels: list[str], transform,
                 path_col: str = "cache_path"):
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.transform = transform
        # Fall back to the original 1024px file if the resized cache is absent.
        self.paths = [p if isinstance(p, str) and Path(p).exists() else o
                      for p, o in zip(self.df.get(path_col, self.df["path"]), self.df["path"])]
        self.targets = torch.tensor(self.df[labels].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict[str, Any]:
        img = self.transform(read_xray(self.paths[i]))
        return {"image": img, "target": self.targets[i],
                "image_id": self.df.at[i, "Image Index"]}


def make_loaders(cfg: dict[str, Any], splits_csv: str | Path | None = None,
                 subset: int | None = None) -> dict[str, DataLoader]:
    splits_csv = splits_csv or Path(cfg["paths"]["processed_dir"]) / "splits.csv"
    df = pd.read_csv(splits_csv)
    labels = all_labels(cfg)
    lcfg = cfg["loader"]
    loaders = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        if subset:
            part = part.sample(min(subset, len(part)), random_state=cfg["project"]["seed"])
        ds = ChestXrayDataset(part, labels, build_transforms(cfg, train=split == "train"))
        loaders[split] = DataLoader(
            ds, batch_size=lcfg["batch_size"], shuffle=split == "train",
            num_workers=lcfg["num_workers"], pin_memory=torch.cuda.is_available(),
            drop_last=split == "train",
        )
    return loaders
