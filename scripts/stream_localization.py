#!/usr/bin/env python
"""Stream localization poses in near real-time from frame JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from hd_map_builder.mapping import OccupancyGridConfig
from pipeline.localization_publisher import StdoutPublisher
from pipeline.localization_stream import LocalizationStreamer, _load_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Localization streaming demo.")
    parser.add_argument("--calibration", type=Path, default=Path("data/calib/sample_calib.yaml"))
    parser.add_argument("--frames", type=Path, default=Path("data/sample_frames.json"))
    parser.add_argument("--grid-resolution", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, nargs=3, default=(20, 20, 4))
    parser.add_argument("--rate", type=float, default=10.0, help="Publish rate in Hz.")
    parser.add_argument("--realtime", action="store_true", help="Sleep between publishes to mimic wall time.")
    args = parser.parse_args()

    streamer = LocalizationStreamer.from_paths(
        args.calibration,
        grid_config=OccupancyGridConfig(resolution=args.grid_resolution, size=tuple(args.grid_size)),
        publish_rate_hz=args.rate,
        publisher=StdoutPublisher(),
        realtime=args.realtime,
    )
    frames = _load_frames(args.frames)
    count = streamer.stream_frames(frames)
    print(f"Streamed {count} localization updates.")


if __name__ == "__main__":
    main()
