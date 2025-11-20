"""Streaming localization helper using OfflineMapBuilder."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from hd_map_builder.localization import Pose2
from hd_map_builder.mapping import OccupancyGridConfig
from .offline_builder import FrameSensors, OfflineMapBuilder
from .localization_publisher import PosePublisher, StdoutPublisher


def _load_frames(frames_path: Path) -> Iterable[dict]:
    with open(frames_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload.get("frames", [])


def _frame_to_inputs(frame: dict) -> Tuple[Pose2, FrameSensors]:
    odo = frame.get("odometry", {})
    pose = Pose2(float(odo.get("dx", 0.0)), float(odo.get("dy", 0.0)), float(odo.get("dtheta", 0.0)))
    sensors_cfg = frame.get("sensors", {})

    pointclouds = {}
    semantics = {}
    for name, payload in sensors_cfg.items():
        points = payload.get("points", [])
        sem = payload.get("semantics")
        if points:
            pointclouds[name] = np.asarray(points, dtype=float)
        if sem is not None:
            semantics[name] = np.asarray(sem, dtype=int)
    return pose, FrameSensors(
        pointclouds=pointclouds,
        semantics=semantics or None,
    )


class LocalizationStreamer:
    """Processes frames sequentially and publishes pose updates at a target rate."""

    def __init__(
        self,
        builder: OfflineMapBuilder,
        *,
        publish_rate_hz: float = 10.0,
        publisher: Optional[PosePublisher] = None,
        realtime: bool = False,
    ):
        self.builder = builder
        self.publish_rate_hz = publish_rate_hz
        self.publisher = publisher or StdoutPublisher()
        self.realtime = realtime

    def stream_frames(self, frames: Iterable[dict]) -> int:
        dt = 1.0 / self.publish_rate_hz
        timestamp_ns = 0
        frame_count = 0
        for frame in frames:
            pose_delta, sensors = _frame_to_inputs(frame)
            self.builder.process_frame(pose_delta, sensors)
            node_id = self.builder.node_ids[-1]
            pose = self.builder.pose_graph.nodes[node_id].pose
            self.publisher.publish(timestamp_ns, pose)
            frame_count += 1
            timestamp_ns += int(dt * 1e9)
            if self.realtime:
                time.sleep(dt)
        return frame_count

    @classmethod
    def from_paths(
        cls,
        calibration_path: Path,
        grid_config: OccupancyGridConfig,
        *,
        publish_rate_hz: float = 10.0,
        publisher: Optional[PosePublisher] = None,
        realtime: bool = False,
    ) -> "LocalizationStreamer":
        builder = OfflineMapBuilder.from_calibration_file(calibration_path, grid_config=grid_config)
        return cls(
            builder=builder,
            publish_rate_hz=publish_rate_hz,
            publisher=publisher,
            realtime=realtime,
        )
