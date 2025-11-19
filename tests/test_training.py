import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hd_map_builder.mapping import OccupancyGrid, OccupancyGridConfig
from hd_map_builder.neural_repr import OccupancySampleConfig, OccupancySampleDataset
from hd_map_builder.neural_repr.training import TrainingConfig, train_decoder


def _dataset():
    cfg = OccupancyGridConfig(resolution=1.0, size=(2, 2, 1))
    grid = OccupancyGrid(cfg)
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    semantics = np.array([1, 2])
    grid.integrate_hits(pts, semantics=semantics)
    return OccupancySampleDataset(grid, config=OccupancySampleConfig(samples=16, seed=0))


def test_train_decoder_runs_epochs():
    dataset = _dataset()
    config = TrainingConfig(epochs=2, batch_size=4, lr=1e-2, semantic_classes=3)
    result = train_decoder(dataset, config=config)
    assert len(result["history"]) == 2
    assert result["history"][-1]["loss"] >= 0
