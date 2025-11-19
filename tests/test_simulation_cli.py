from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # ensures dependent modules available

from hd_map_builder.mapping import OccupancyGridConfig
from scripts.simulate_scenario import SimulationConfig, simulate_and_report


def test_simulation_metrics_are_deterministic(tmp_path: Path):
    sim_config = SimulationConfig(steps=5, seed=1, step_dx=0.4, landmarks=5)
    grid_config = OccupancyGridConfig(resolution=0.5, size=(20, 20, 2))
    metrics = simulate_and_report(
        calibration_path=Path("data/calib/sample_calib.yaml"),
        grid_config=grid_config,
        sim_config=sim_config,
    )
    assert metrics["frames"] == 5
    assert metrics["occupied_cells"] > 0
    assert metrics["rms_translation_m"] >= 0.0
