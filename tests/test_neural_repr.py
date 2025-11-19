import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hd_map_builder.mapping import OccupancyGrid, OccupancyGridConfig
from hd_map_builder.neural_repr import (
    ImplicitMapConfig,
    ImplicitMapDecoder,
    OccupancySampleConfig,
    OccupancySampleDataset,
)


def _grid():
    cfg = OccupancyGridConfig(resolution=1.0, size=(2, 2, 1))
    grid = OccupancyGrid(cfg)
    pts = np.array([[0.1, 0.2, 0.1], [1.1, 0.2, 0.1]])
    semantics = np.array([1, 2])
    grid.integrate_hits(pts, semantics=semantics)
    return grid


def test_occupancy_sample_dataset_outputs_expected_keys():
    grid = _grid()
    dataset = OccupancySampleDataset(
        grid,
        config=OccupancySampleConfig(samples=8, semantics=True, seed=0),
    )
    assert len(dataset) == 8
    sample = dataset[0]
    assert set(sample.keys()) == {"coords", "occupancy", "semantics"}
    assert sample["coords"].shape == (3,)
    assert sample["occupancy"].shape == (1,)


def test_implicit_map_decoder_forward_and_backward():
    cfg = ImplicitMapConfig(feature_dim=1, hidden_dim=32, num_layers=2, semantic_classes=3)
    net = ImplicitMapDecoder(cfg)
    coords = torch.randn(16, 3)
    features = torch.randn(16, 1)

    outputs = net(coords, features)
    assert outputs["sdf"].shape == (16, 1)
    assert outputs["semantics"].shape == (16, 3)

    loss = outputs["sdf"].pow(2).mean() + outputs["semantics"].pow(2).mean()
    loss.backward()
    for param in net.parameters():
        assert param.grad is not None
