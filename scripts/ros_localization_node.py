#!/usr/bin/env python
"""Launch ROS2 localization node that publishes Odometry messages."""

from __future__ import annotations

import argparse
from pathlib import Path

from hd_map_builder.mapping import OccupancyGridConfig
from pipeline.ros_localization_node import run_ros_localization_node


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 Localization Node Launcher")
    parser.add_argument("--calibration", type=Path, default=Path("data/calib/sample_calib.yaml"))
    parser.add_argument("--frames", type=Path, default=Path("data/sample_frames.json"))
    parser.add_argument("--grid-resolution", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, nargs=3, default=(20, 20, 4))
    parser.add_argument("--topic", type=str, default="/localization/odometry")
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    grid_cfg = OccupancyGridConfig(resolution=args.grid_resolution, size=tuple(args.grid_size))
    run_ros_localization_node(
        calibration_path=args.calibration,
        frames_path=args.frames,
        grid_config=grid_cfg,
        topic=args.topic,
        publish_rate_hz=args.rate,
        realtime=args.realtime,
    )


if __name__ == "__main__":
    main()
