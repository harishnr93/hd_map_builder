#!/usr/bin/env python
"""Export fused occupancy map to PLY for visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hd_map_builder.localization import Pose2
from hd_map_builder.mapping import OccupancyGridConfig
from pipeline.offline_builder import FrameSensors, OfflineMapBuilder
from pipeline.replay_cli import _frame_to_inputs, _load_frames


def _write_ply(path: Path, points: np.ndarray, semantics: np.ndarray | None = None) -> None:
    if points.size == 0:
        raise ValueError("No occupied points to export.")
    with open(path, "w", encoding="utf-8") as fp:
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {points.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if semantics is not None:
            header.append("property uchar semantic")
        header.append("end_header")
        fp.write("\n".join(header) + "\n")
        for idx, point in enumerate(points):
            row = [f"{coord:.5f}" for coord in point]
            if semantics is not None:
                row.append(str(int(semantics[idx])))
            fp.write(" ".join(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export occupancy map to PLY.")
    parser.add_argument("--calibration", type=Path, default=Path("data/calib/sample_calib.yaml"))
    parser.add_argument("--frames", type=Path, default=Path("data/sample_frames.json"))
    parser.add_argument("--grid-resolution", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, nargs=3, default=(20, 20, 4))
    parser.add_argument("--output", type=Path, default=Path("logs/fused_map.ply"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    builder = OfflineMapBuilder.from_calibration_file(
        args.calibration,
        grid_config=OccupancyGridConfig(resolution=args.grid_resolution, size=tuple(args.grid_size)),
    )
    for frame in _load_frames(args.frames):
        pose_delta, sensor_inputs = _frame_to_inputs(frame)
        builder.process_frame(pose_delta, sensor_inputs)
    builder.optimize()
    points, semantics = builder.occupancy_points_with_semantics(threshold=args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_ply(args.output, points, semantics)
    print(f"Exported {points.shape[0]} voxels to {args.output}")


if __name__ == "__main__":
    main()
