import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from hd_map_builder.neural_repr.onnx_runtime import run_onnx_benchmark


def _create_identity_model(path, input_dim=3):
    from onnx import helper, TensorProto

    input_tensor = helper.make_tensor_value_info("coords", TensorProto.FLOAT, [None, input_dim])
    output_tensor = helper.make_tensor_value_info("sdf", TensorProto.FLOAT, [None, input_dim])
    node = helper.make_node("Identity", inputs=["coords"], outputs=["sdf"])
    graph = helper.make_graph([node], "IdentityGraph", [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    onnx.save(model, path)


def test_run_onnx_benchmark(tmp_path):
    model_path = tmp_path / "identity.onnx"
    _create_identity_model(model_path)
    metrics = run_onnx_benchmark(model_path, batch_size=16, coord_dim=3, steps=3)
    assert metrics["avg_ms"] >= 0.0
    assert "CPUExecutionProvider" in "".join(metrics["provider"])
