"""Neural implicit training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .implicit_map import ImplicitMapConfig, ImplicitMapDecoder


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    semantic_weight: float = 1.0
    semantic_classes: int = 16
    device: Optional[str] = None


def train_decoder(
    dataset: Dataset,
    *,
    config: TrainingConfig = TrainingConfig(),
    decoder: Optional[ImplicitMapDecoder] = None,
) -> dict:
    """Train implicit decoder on provided dataset; returns metrics history."""
    if decoder is None:
        decoder = ImplicitMapDecoder(
            ImplicitMapConfig(semantic_classes=config.semantic_classes)
        )
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    decoder.to(device)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.lr)
    bce = torch.nn.BCEWithLogitsLoss()
    history = []

    for epoch in range(config.epochs):
        decoder.train()
        total_loss = 0.0
        total_batches = 0
        sem_loss_sum = 0.0
        sem_batches = 0
        for batch in dataloader:
            coords = batch["coords"].to(device)
            occupancy = batch["occupancy"].to(device)
            features = batch.get("features")
            if features is not None:
                features = features.to(device)

            outputs = decoder(coords, features)
            occ_logits = outputs["sdf"]
            occ_loss = bce(occ_logits, occupancy)

            loss = occ_loss
            semantics = batch.get("semantics")
            if semantics is not None:
                semantics = semantics.to(device)
                mask = semantics >= 0
                if mask.any():
                    sem_logits = outputs["semantics"][mask]
                    sem_targets = semantics[mask]
                    ce = torch.nn.functional.cross_entropy(sem_logits, sem_targets)
                    loss = loss + config.semantic_weight * ce
                    sem_loss_sum += float(ce.detach().cpu())
                    sem_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        avg_sem_loss = sem_loss_sum / max(sem_batches, 1) if sem_batches > 0 else 0.0
        history.append({"epoch": epoch + 1, "loss": avg_loss, "sem_loss": avg_sem_loss})

    return {"decoder": decoder, "history": history}
