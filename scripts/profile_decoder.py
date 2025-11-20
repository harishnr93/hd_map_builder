#!/usr/bin/env python
"""CLI to profile implicit decoder inference throughput."""

from __future__ import annotations

import argparse

from hd_map_builder.neural_repr import ImplicitMapConfig, ImplicitMapDecoder, TORCH_AVAILABLE
from hd_map_builder.neural_repr.export import profile_decoder


def main() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for profiling.")

    parser = argparse.ArgumentParser(description="Profile implicit decoder inference speed.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=str, help="torch device override (cpu/cuda)")
    parser.add_argument("--coord-dim", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--semantic-classes", type=int, default=16)
    args = parser.parse_args()

    config = ImplicitMapConfig(
        coord_dim=args.coord_dim,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        semantic_classes=args.semantic_classes,
    )
    decoder = ImplicitMapDecoder(config)
    metrics = profile_decoder(
        decoder,
        batch_size=args.batch_size,
        steps=args.steps,
        warmup=args.warmup,
        device=args.device,
    )
    print(
        f"Device: {metrics['device']} | Batch: {metrics['batch_size']} | "
        f"Avg ms: {metrics['avg_ms']:.3f} | Throughput: {metrics['throughput_sps']:.2f} samples/s"
    )


if __name__ == "__main__":
    main()
