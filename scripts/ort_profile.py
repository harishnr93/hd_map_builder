#!/usr/bin/env python
"""Run ONNX Runtime benchmark on exported decoder."""

from __future__ import annotations

import argparse
from pathlib import Path

from hd_map_builder.neural_repr.onnx_runtime import run_onnx_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark implicit decoder with ONNX Runtime.")
    parser.add_argument("--model", type=Path, default=Path("logs/implicit_decoder.onnx"))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--coord-dim", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    metrics = run_onnx_benchmark(
        args.model,
        batch_size=args.batch_size,
        coord_dim=args.coord_dim,
        feature_dim=args.feature_dim,
        steps=args.steps,
    )
    print(
        f"ONNX Runtime ({metrics['provider']}): batch={metrics['batch_size']} avg={metrics['avg_ms']:.3f} ms "
        f"throughput={metrics['throughput_sps']:.2f} samples/s"
    )


if __name__ == "__main__":
    main()
