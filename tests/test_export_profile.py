from pathlib import Path

import importlib

import pytest

torch = pytest.importorskip("torch")

from hd_map_builder.neural_repr import ImplicitMapConfig, ImplicitMapDecoder
from hd_map_builder.neural_repr.export import export_decoder_to_onnx, profile_decoder


def _onnx_available() -> bool:
    return importlib.util.find_spec("onnxscript") is not None


@pytest.mark.skipif(not _onnx_available(), reason="onnxscript not installed")
def test_export_decoder_to_onnx(tmp_path: Path):
    config = ImplicitMapConfig(semantic_classes=4)
    decoder = ImplicitMapDecoder(config)
    out_path = tmp_path / "decoder.onnx"
    export_decoder_to_onnx(decoder, out_path, sample_shape=(2, config.coord_dim))
    assert out_path.exists()


def test_profile_decoder_returns_metrics():
    config = ImplicitMapConfig(semantic_classes=4)
    decoder = ImplicitMapDecoder(config)
    metrics = profile_decoder(decoder, batch_size=64, steps=5, warmup=1, device="cpu")
    assert metrics["avg_ms"] > 0.0
    assert metrics["throughput_sps"] > 0.0
