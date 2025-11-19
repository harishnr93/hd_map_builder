"""Command-line helper to replay frames using OfflineMapBuilder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

from hd_map_builder.localization import Pose2
from hd_map_builder.mapping import OccupancyGridConfig
from pipeline.offline_builder import FrameSensors, OfflineMapBuilder


def _load_frames(frames_path: Path) -> Iterable[dict]:
    with open(frames_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload.get("frames", [])


def _frame_to_inputs(frame: dict) -> Tuple[Pose2, FrameSensors]:
    odo = frame.get("odometry", {})
    pose = Pose2(float(odo.get("dx", 0.0)), float(odo.get("dy", 0.0)), float(odo.get("dtheta", 0.0)))
    sensors_cfg: Mapping[str, dict] = frame.get("sensors", {})

    pointclouds: Dict[str, np.ndarray] = {}
    semantics: Dict[str, np.ndarray] = {}
    for name, payload in sensors_cfg.items():
        pts = np.asarray(payload.get("points", []), dtype=float)
        pointclouds[name] = pts
        if "semantics" in payload:
            semantics[name] = np.asarray(payload["semantics"], dtype=int)
    frame_sensors = FrameSensors(pointclouds=pointclouds, semantics=semantics or None)
    return pose, frame_sensors


def run_replay(
    *,
    calibration_path: Path,
    frames_path: Path,
    grid_resolution: float,
    grid_size: Tuple[int, int, int],
    optimize: bool = True,
    dataset_path: Path | None = None,
    samples: int = 2048,
) -> dict:
    """Replay synthetic/recorded frames and optionally export dataset."""
    builder = OfflineMapBuilder.from_calibration_file(
        calibration_path,
        grid_config=OccupancyGridConfig(resolution=grid_resolution, size=grid_size),
    )

    for frame in _load_frames(frames_path):
        pose_delta, sensor_inputs = _frame_to_inputs(frame)
        builder.process_frame(pose_delta, sensor_inputs)

    if optimize:
        builder.optimize()

    dataset_file = None
    if dataset_path is not None:
        dataset = builder.build_training_dataset(samples=samples, semantics=True)
        coords = np.stack([dataset[i]["coords"].numpy() for i in range(len(dataset))])
        occupancy = np.stack([dataset[i]["occupancy"].numpy() for i in range(len(dataset))])
        semantics = (
            np.stack([dataset[i]["semantics"].numpy() for i in range(len(dataset))])
            if "semantics" in dataset[0]
            else None
        )
        np.savez(dataset_path, coords=coords, occupancy=occupancy, semantics=semantics)
        dataset_file = str(dataset_path)

    occupied = builder.occupancy_points(threshold=0.5)
    summary = {
        "frames": builder.frames_processed(),
        "occupied_cells": int(occupied.shape[0]),
        "dataset": dataset_file,
    }
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay frames through OfflineMapBuilder.")
    parser.add_argument("--calibration", type=Path, default=Path("data/calib/sample_calib.yaml"))
    parser.add_argument("--frames", type=Path, default=Path("data/sample_frames.json"))
    parser.add_argument("--grid-resolution", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, nargs=3, default=(20, 20, 4))
    parser.add_argument("--no-optimize", action="store_true", help="Skip pose graph optimization.")
    parser.add_argument("--dataset-out", type=Path, help="Optional path to save sampled dataset (npz).")
    parser.add_argument("--samples", type=int, default=2048, help="Number of samples when exporting dataset.")
    args = parser.parse_args(argv)

    summary = run_replay(
        calibration_path=args.calibration,
        frames_path=args.frames,
        grid_resolution=args.grid_resolution,
        grid_size=tuple(args.grid_size),
        optimize=not args.no_optimize,
        dataset_path=args.dataset_out,
        samples=args.samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
