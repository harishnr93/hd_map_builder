from pathlib import Path

import numpy as np

from hd_map_builder.mapping import OccupancyGridConfig
from pipeline.offline_builder import OfflineMapBuilder
from pipeline.replay_cli import _frame_to_inputs, _load_frames
from scripts import export_map


def _builder():
    builder = OfflineMapBuilder.from_calibration_file(
        "data/calib/sample_calib.yaml",
        grid_config=OccupancyGridConfig(resolution=0.5, size=(10, 10, 2)),
    )
    for frame in _load_frames(Path("data/sample_frames.json")):
        pose_delta, sensors = _frame_to_inputs(frame)
        builder.process_frame(pose_delta, sensors)
    builder.optimize()
    return builder


def test_write_ply_outputs_expected_lines(tmp_path: Path):
    builder = _builder()
    points, semantics = builder.occupancy_points_with_semantics(threshold=0.4)
    out_file = tmp_path / "map.ply"
    export_map._write_ply(out_file, points, semantics)
    content = out_file.read_text().splitlines()
    assert content[0] == "ply"
    assert f"element vertex {points.shape[0]}" in content[2]
    header_len = content.index("end_header") + 1
    assert len(content) == header_len + points.shape[0]


def test_write_ply_raises_on_empty(tmp_path: Path):
    out_file = tmp_path / "map.ply"
    points = np.empty((0, 3))
    try:
        export_map._write_ply(out_file, points, None)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for empty export.")
