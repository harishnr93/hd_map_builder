import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: F841

from pipeline import replay_cli


def test_run_replay_exports_dataset(tmp_path: Path):
    dataset_path = tmp_path / "samples.npz"
    summary = replay_cli.run_replay(
        calibration_path=Path("data/calib/sample_calib.yaml"),
        frames_path=Path("data/sample_frames.json"),
        grid_resolution=0.5,
        grid_size=(10, 10, 2),
        dataset_path=dataset_path,
        samples=8,
    )

    assert summary["frames"] == 2
    assert dataset_path.exists()
    payload = np.load(dataset_path)
    assert payload["coords"].shape[0] == 8
