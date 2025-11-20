"""Utilities for exporting and profiling implicit decoders."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import time

import torch
from torch.export import Dim

from .implicit_map import ImplicitMapConfig, ImplicitMapDecoder


def export_decoder_to_onnx(
    decoder: ImplicitMapDecoder,
    output_path: Path,
    *,
    sample_shape: Tuple[int, int] = (1, 3),
    opset: int = 17,
) -> Path:
    """Export decoder to ONNX with dummy inputs."""
    try:
        import onnxscript  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "onnxscript is required for ONNX export. Install with `pip install onnx onnxscript`."
        ) from exc

    decoder.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coords = torch.randn(sample_shape, dtype=torch.float32)
    features_dim = decoder.config.feature_dim
    features = torch.randn(sample_shape[0], features_dim, dtype=torch.float32) if features_dim else None

    inputs = (coords,) if features is None else (coords, features)
    batch_dim = Dim("batch")
    dynamic_shapes = [{0: batch_dim}]
    if features is not None:
        dynamic_shapes.append({0: batch_dim})

    torch.onnx.export(
        decoder,
        inputs,
        str(output_path),
        export_params=True,
        opset_version=opset,
        input_names=["coords", "features"] if features is not None else ["coords"],
        output_names=["sdf", "semantics"],
        dynamic_axes=None,
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
    )
    return output_path


def profile_decoder(
    decoder: ImplicitMapDecoder,
    *,
    batch_size: int = 1024,
    steps: int = 50,
    warmup: int = 5,
    device: str | None = None,
) -> dict:
    """Run synthetic inference loop and return throughput metrics."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    decoder = decoder.to(device).eval()
    coords = torch.randn(batch_size, decoder.config.coord_dim, device=device)
    features = (
        torch.randn(batch_size, decoder.config.feature_dim, device=device)
        if decoder.config.feature_dim
        else None
    )
    torch.cuda.synchronize() if device == "cuda" else None

    def _run_once():
        with torch.no_grad():
            decoder(coords, features)

    for _ in range(warmup):
        _run_once()

    torch.cuda.synchronize() if device == "cuda" else None
    start = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

    if device == "cuda":
        start.record()
    else:
        import time

        start_time = time.perf_counter()

    for _ in range(steps):
        _run_once()

    if device == "cuda":
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    avg_ms = elapsed_ms / steps
    throughput = batch_size / (avg_ms / 1000.0)
    return {"device": device, "batch_size": batch_size, "avg_ms": avg_ms, "throughput_sps": throughput}
