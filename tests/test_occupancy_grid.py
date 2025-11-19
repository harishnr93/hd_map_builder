import numpy as np
import pytest

from hd_map_builder.mapping import OccupancyGrid, OccupancyGridConfig


def test_integrate_hits_updates_log_odds_and_semantics():
    config = OccupancyGridConfig(resolution=1.0, size=(4, 4, 2), origin=(0.0, 0.0, 0.0))
    grid = OccupancyGrid(config)
    pts = np.array([[0.2, 0.2, 0.1], [1.2, 0.2, 0.1]])
    semantics = np.array([5, 6])
    count = grid.integrate_hits(pts, semantics=semantics)
    assert count == 2
    prob = grid.occupancy_probabilities()
    assert prob[0, 0, 0] > 0.5
    semantic_map = grid.semantic_mode()
    assert semantic_map[0, 0, 0] == 5
    assert semantic_map[1, 0, 0] == 6


def test_integrate_hits_validates_semantic_length():
    config = OccupancyGridConfig(resolution=1.0, size=(2, 2, 1))
    grid = OccupancyGrid(config)
    pts = np.array([[0, 0, 0]])
    with pytest.raises(ValueError):
        grid.integrate_hits(pts, semantics=np.array([1, 2]))


def test_integrate_free_decreases_log_odds():
    config = OccupancyGridConfig(resolution=1.0, size=(2, 2, 1))
    grid = OccupancyGrid(config)
    pts = np.array([[0.2, 0.2, 0.2]])
    grid.integrate_hits(pts)
    before = grid.occupancy_probabilities()[0, 0, 0]
    grid.integrate_free(pts)
    after = grid.occupancy_probabilities()[0, 0, 0]
    assert after < before
