#!/usr/bin/env python
"""CLI to export ImplicitMapDecoder to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from hd_map_builder.neural_repr import ImplicitMapConfig, ImplicitMapDecoder, TORCH_AVAILABLE
from hd_map_builder.neural_repr.export import export_decoder_to_onnx


def main() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for ONNX export.")

    parser = argparse.ArgumentParser(description="Export implicit decoder to ONNX.")
    parser.add_argument("--output", type=Path, default=Path("logs/implicit_decoder.onnx"))
    parser.add_argument("--coord-dim", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--semantic-classes", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    config = ImplicitMapConfig(
        coord_dim=args.coord_dim,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        semantic_classes=args.semantic_classes,
    )
    decoder = ImplicitMapDecoder(config)
    export_decoder_to_onnx(decoder, args.output, sample_shape=(args.batch, args.coord_dim))
    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()
