"""Simple neural implicit decoder that predicts SDF and semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class ImplicitMapConfig:
    coord_dim: int = 3
    feature_dim: int = 0
    hidden_dim: int = 128
    num_layers: int = 4
    semantic_classes: int = 16


class ImplicitMapDecoder(nn.Module):
    """MLP that maps xyz (+features) to signed distance + semantic logits."""

    def __init__(self, config: ImplicitMapConfig):
        super().__init__()
        self.config = config
        input_dim = config.coord_dim + config.feature_dim
        layers = []
        for layer_idx in range(config.num_layers):
            in_dim = input_dim if layer_idx == 0 else config.hidden_dim
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        self.backbone = nn.Sequential(*layers)
        self.sdf_head = nn.Linear(config.hidden_dim, 1)
        self.semantic_head = nn.Linear(config.hidden_dim, config.semantic_classes)

    def forward(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> dict:
        if not torch.onnx.is_in_onnx_export():
            if coords.ndim != 2 or coords.shape[-1] != self.config.coord_dim:
                raise ValueError("coords must be of shape [N, coord_dim].")
            if features is not None and features.shape[0] != coords.shape[0]:
                raise ValueError("features must align with coords batch.")
        if features is None:
            features = torch.zeros(
                coords.shape[0],
                self.config.feature_dim,
                device=coords.device,
                dtype=coords.dtype,
            )
        elif features.shape[1] != self.config.feature_dim:
            raise ValueError("features second dimension must match config.feature_dim.")

        x = torch.cat([coords, features], dim=-1)
        latent = self.backbone(x)
        sdf = self.sdf_head(latent)
        semantics = self.semantic_head(latent)
        return {"sdf": sdf, "semantics": semantics}
