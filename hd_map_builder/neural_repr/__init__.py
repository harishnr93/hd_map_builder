"""Neural implicit map representation utilities."""

from __future__ import annotations

TORCH_AVAILABLE = True

try:
    from .implicit_map import ImplicitMapDecoder, ImplicitMapConfig
    from .dataset import (
        OccupancySampleDataset,
        OccupancySampleConfig,
        ArrayOccupancyDataset,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional torch
    if exc.name != "torch":
        raise
    TORCH_AVAILABLE = False
    ImplicitMapDecoder = None  # type: ignore
    ImplicitMapConfig = None  # type: ignore
    OccupancySampleDataset = None  # type: ignore
    OccupancySampleConfig = None  # type: ignore
    ArrayOccupancyDataset = None  # type: ignore

__all__ = [
    "TORCH_AVAILABLE",
    "ImplicitMapDecoder",
    "ImplicitMapConfig",
    "OccupancySampleDataset",
    "OccupancySampleConfig",
    "ArrayOccupancyDataset",
]
