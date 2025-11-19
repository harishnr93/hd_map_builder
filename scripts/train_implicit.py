#!/usr/bin/env python
"""Train neural implicit decoder from saved dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hd_map_builder.neural_repr import (
    ArrayOccupancyDataset,
    ImplicitMapConfig,
    ImplicitMapDecoder,
    TORCH_AVAILABLE,
)
from hd_map_builder.neural_repr.training import TrainingConfig, train_decoder


def _load_dataset(path: Path) -> ArrayOccupancyDataset:
    payload = np.load(path)
    coords = payload["coords"]
    occupancy = payload["occupancy"]
    semantics = payload["semantics"] if "semantics" in payload and payload["semantics"].size else None
    return ArrayOccupancyDataset(coords=coords, occupancy=occupancy, semantics=semantics)


def main() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training.")

    parser = argparse.ArgumentParser(description="Train implicit decoder on saved dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help=".npz dataset from replay CLI.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--semantic-classes", type=int, default=16)
    parser.add_argument("--device", type=str, help="Torch device override (cpu/cuda).")
    parser.add_argument("--checkpoint", type=Path, help="Optional path to save trained state dict.")
    args = parser.parse_args()

    dataset = _load_dataset(args.dataset)
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        semantic_weight=args.semantic_weight,
        semantic_classes=args.semantic_classes,
        device=args.device,
    )
    decoder = ImplicitMapDecoder(ImplicitMapConfig(semantic_classes=args.semantic_classes))
    result = train_decoder(dataset, config=config, decoder=decoder)

    last = result["history"][-1]
    print(f"Training complete. Final loss: {last['loss']:.4f}, semantic: {last['sem_loss']:.4f}")
    if args.checkpoint:
        torch.save(decoder.state_dict(), args.checkpoint)
        print(f"Checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
