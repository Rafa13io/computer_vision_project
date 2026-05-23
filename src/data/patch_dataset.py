from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Load grayscale tooth patches as 3-channel tensors in [-1, 1]."""

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path,
        split_filter: str | None = "train",
        rare_only: bool = False,
        exclude_holdout: bool = True,
    ) -> None:
        self.root = Path(project_root)
        self.rows: list[dict] = []
        with open(manifest_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if exclude_holdout and row["split"] == "holdout":
                    continue
                if split_filter and row["split"] != split_filter:
                    continue
                if rare_only and int(row["is_rare"]) != 1:
                    continue
                self.rows.append(row)

        if not self.rows:
            raise ValueError(
                f"No samples in manifest for split={split_filter!r} rare_only={rare_only}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        path = self.root / row["patch_path"]
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
        return {
            "image": x,
            "patch_id": int(row["patch_id"]),
            "is_rare": int(row["is_rare"]),
        }
