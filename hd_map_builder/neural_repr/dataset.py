"""Dataset utilities for sampling occupancy grid training pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from hd_map_builder.mapping import OccupancyGrid


@dataclass
class OccupancySampleConfig:
    samples: int = 2048
    semantics: bool = True
    seed: int = 13


class OccupancySampleDataset(Dataset):
    """Randomly samples occupancy grid cells for neural training."""

    def __init__(
        self,
        grid: OccupancyGrid,
        *,
        config: OccupancySampleConfig = OccupancySampleConfig(),
    ):
        self.config = config
        rng = np.random.default_rng(config.seed)
        probs = grid.occupancy_probabilities()
        idxs = np.stack(np.meshgrid(
            np.arange(grid.config.size[0]),
            np.arange(grid.config.size[1]),
            np.arange(grid.config.size[2]),
            indexing="ij",
        ), axis=-1).reshape(-1, 3)

        sampled_idx = rng.choice(
            len(idxs),
            size=config.samples,
            replace=True,
        )
        self.indices = idxs[sampled_idx]
        centers = grid.voxel_centers(self.indices)
        self.coords = torch.from_numpy(centers).float()
        self.occupancy = torch.from_numpy(
            probs[tuple(self.indices.T)]
        ).float().unsqueeze(-1)

        semantic_volume = grid.semantic_mode()
        if config.semantics and semantic_volume is not None:
            semantics = semantic_volume[tuple(self.indices.T)]
            self.semantics = torch.from_numpy(semantics.astype(np.int64))
        else:
            self.semantics = None

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, idx: int):
        item = {
            "coords": self.coords[idx],
            "occupancy": self.occupancy[idx],
        }
        if self.semantics is not None:
            item["semantics"] = self.semantics[idx]
        return item


class ArrayOccupancyDataset(Dataset):
    """Dataset sourced from numpy arrays (e.g., saved replay exports)."""

    def __init__(
        self,
        coords: np.ndarray,
        occupancy: np.ndarray,
        semantics: Optional[np.ndarray] = None,
    ):
        self.coords = torch.as_tensor(coords, dtype=torch.float32)
        self.occupancy = torch.as_tensor(occupancy, dtype=torch.float32)
        self.semantics = (
            torch.as_tensor(semantics, dtype=torch.long) if semantics is not None else None
        )

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, idx: int):
        sample = {"coords": self.coords[idx], "occupancy": self.occupancy[idx]}
        if self.semantics is not None:
            sample["semantics"] = self.semantics[idx]
        return sample
