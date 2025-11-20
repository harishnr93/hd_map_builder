"""ONNX Runtime helper utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ort = None  # type: ignore


def run_onnx_benchmark(
    model_path: str | Path,
    *,
    batch_size: int = 2048,
    coord_dim: int = 3,
    feature_dim: int = 0,
    steps: int = 100,
) -> dict:
    """Load an ONNX model and benchmark inference throughput."""
    if ort is None:
        raise RuntimeError("onnxruntime is required for this benchmark.")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = {}
    coords = np.random.randn(batch_size, coord_dim).astype(np.float32)
    inputs["coords"] = coords
    if feature_dim > 0 and any(inp.name == "features" for inp in session.get_inputs()):
        features = np.random.randn(batch_size, feature_dim).astype(np.float32)
        inputs["features"] = features

    # warmup
    for _ in range(5):
        session.run(None, inputs)

    start = time.perf_counter()
    for _ in range(steps):
        session.run(None, inputs)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / steps) * 1000.0
    throughput = batch_size / (avg_ms / 1000.0)
    return {
        "batch_size": batch_size,
        "steps": steps,
        "avg_ms": avg_ms,
        "throughput_sps": throughput,
        "provider": session.get_providers(),
    }
