"""3D occupancy grid representation with semantic fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class OccupancyGridConfig:
    resolution: float
    size: Tuple[int, int, int]
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    free_log_odds: float = -0.4
    occupied_log_odds: float = 0.85
    min_log_odds: float = -5.0
    max_log_odds: float = 5.0

    def __post_init__(self):
        if len(self.size) != 3:
            raise ValueError("size must be a 3-tuple for x/y/z cells.")
        if len(self.origin) != 3:
            raise ValueError("origin must be a 3-tuple.")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive.")


class OccupancyGrid:
    """Dense 3D occupancy grid that aggregates hits from multiple sensors."""

    def __init__(self, config: OccupancyGridConfig):
        self.config = config
        self.log_odds = np.zeros(config.size, dtype=float)
        self.semantic_counts: Dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self.log_odds.fill(0.0)
        self.semantic_counts.clear()

    def integrate_hits(
        self,
        points_vehicle: np.ndarray,
        *,
        semantics: Optional[np.ndarray] = None,
        weight: float = 1.0,
    ) -> int:
        """Accumulate log-odds for occupied cells."""
        pts = np.asarray(points_vehicle, dtype=float)
        if pts.size == 0:
            return 0
        if semantics is not None and len(semantics) != pts.shape[0]:
            raise ValueError("Semantics array must align with point cloud length.")
        idxs, mask = self._points_to_indices(pts)
        if not np.any(mask):
            return 0
        clamped_idxs = idxs[mask]
        increment = self.config.occupied_log_odds * weight
        np.add.at(self.log_odds, tuple(clamped_idxs.T), increment)
        self._clamp_log_odds()
        if semantics is not None:
            self._integrate_semantics(clamped_idxs, semantics[mask])
        return clamped_idxs.shape[0]

    def integrate_free(
        self,
        points_vehicle: np.ndarray,
        *,
        weight: float = 1.0,
    ) -> int:
        """Integrate free-space observations (e.g., along LiDAR rays)."""
        pts = np.asarray(points_vehicle, dtype=float)
        if pts.size == 0:
            return 0
        idxs, mask = self._points_to_indices(pts)
        clamped_idxs = idxs[mask]
        if clamped_idxs.size == 0:
            return 0
        decrement = self.config.free_log_odds * weight
        np.add.at(self.log_odds, tuple(clamped_idxs.T), decrement)
        self._clamp_log_odds()
        return clamped_idxs.shape[0]

    def occupancy_probabilities(self) -> np.ndarray:
        """Return occupancy probability volume."""
        odds = self.log_odds
        return 1 - 1 / (1 + np.exp(odds))

    def semantic_mode(self) -> Optional[np.ndarray]:
        """Return semantic labels per cell (argmax over counts)."""
        if not self.semantic_counts:
            return None
        class_ids = sorted(self.semantic_counts.keys())
        stacked = np.stack([self.semantic_counts[cid] for cid in class_ids], axis=0)
        argmax = np.argmax(stacked, axis=0)
        label_map = np.zeros(self.log_odds.shape, dtype=np.int32)
        for idx, cid in enumerate(class_ids):
            label_map[argmax == idx] = cid
        # Mark cells with no semantic evidence as -1
        zero_mask = np.sum(stacked, axis=0) == 0
        label_map[zero_mask] = -1
        return label_map

    def occupied_points(self, threshold: float = 0.5) -> np.ndarray:
        """Return world coordinates of cells exceeding probability threshold."""
        probs = self.occupancy_probabilities()
        mask = probs >= threshold
        if not np.any(mask):
            return np.empty((0, 3))
        idxs = np.argwhere(mask)
        centers = self._indices_to_points(idxs)
        return centers

    def _points_to_indices(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rel = (points - np.asarray(self.config.origin)) / self.config.resolution
        idxs = np.floor(rel).astype(int)
        size = np.array(self.config.size)
        in_bounds = np.all((idxs >= 0) & (idxs < size), axis=1)
        return idxs, in_bounds

    def _indices_to_points(self, idxs: np.ndarray) -> np.ndarray:
        centers = (idxs.astype(float) + 0.5) * self.config.resolution
        return centers + np.asarray(self.config.origin)

    def _clamp_log_odds(self) -> None:
        np.clip(
            self.log_odds,
            self.config.min_log_odds,
            self.config.max_log_odds,
            out=self.log_odds,
        )

    def _integrate_semantics(self, idxs: np.ndarray, semantics: np.ndarray) -> None:
        semantic_ids = semantics.astype(int)
        for class_id in np.unique(semantic_ids):
            if class_id not in self.semantic_counts:
                self.semantic_counts[class_id] = np.zeros_like(self.log_odds, dtype=np.int32)
            mask = semantic_ids == class_id
            np.add.at(self.semantic_counts[class_id], tuple(idxs[mask].T), 1)
