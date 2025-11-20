#!/usr/bin/env python
"""Replay ROS bag sensor topics into OfflineMapBuilder pipeline."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from rosbags.highlevel import AnyReader

from hd_map_builder.mapping import OccupancyGridConfig
from hd_map_builder.sensors import load_calibration
from hd_map_builder.localization import Pose2
from pipeline.offline_builder import FrameSensors, OfflineMapBuilder
from pipeline.localization_publisher import StdoutPublisher
from pipeline.localization_stream import LocalizationStreamer


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def pointcloud2_to_xyz(msg) -> np.ndarray:
    if not hasattr(msg, "data"):
        return np.empty((0, 3))
    floats = np.frombuffer(msg.data, dtype=np.float32)
    step = max(1, msg.point_step // 4)
    reshaped = floats.reshape(-1, step)
    return reshaped[:, :3]


def odom_to_pose(msg) -> Pose2:
    pos = msg.pose.pose.position
    ori = msg.pose.pose.orientation
    yaw = quaternion_to_yaw(ori.x, ori.y, ori.z, ori.w)
    return Pose2(pos.x, pos.y, yaw)


def replay_rosbag(
    bag_path: Path,
    calibration_path: Path,
    *,
    lidar_topics: Sequence[str],
    radar_topics: Sequence[str],
    odom_topic: str,
    grid_config: OccupancyGridConfig,
    publish_rate_hz: float = 10.0,
) -> None:
    calibration = load_calibration(calibration_path)
    builder = OfflineMapBuilder(calibration=calibration, grid_config=grid_config)
    streamer = LocalizationStreamer(
        builder=builder,
        publish_rate_hz=publish_rate_hz,
        publisher=StdoutPublisher(),
    )

    frames = _generate_frames(bag_path, lidar_topics, radar_topics, odom_topic)
    streamer.stream_frames(frames)


def _generate_frames(
    bag_path: Path,
    lidar_topics: Sequence[str],
    radar_topics: Sequence[str],
    odom_topic: str,
) -> Iterable[dict]:
    with AnyReader([bag_path]) as reader:
        topic_set = set(lidar_topics) | set(radar_topics) | {odom_topic}
        connections = [c for c in reader.connections if c.topic in topic_set]
        sensor_buffer: Dict[str, list[np.ndarray]] = defaultdict(list)
        prev_pose: Optional[Pose2] = None
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic in lidar_topics or connection.topic in radar_topics:
                xyz = pointcloud2_to_xyz(msg)
                if xyz.size:
                    sensor_buffer[connection.topic].append(xyz)
            elif connection.topic == odom_topic:
                pose = odom_to_pose(msg)
                if prev_pose is None:
                    prev_pose = pose
                    continue
                delta = prev_pose.between(pose)
                sensors_dict = {}
                for topic, arrays in sensor_buffer.items():
                    if arrays:
                        sensors_dict[topic] = {"points": np.vstack(arrays)}
                frame = {
                    "odometry": {"dx": delta.x, "dy": delta.y, "dtheta": delta.theta},
                    "sensors": sensors_dict,
                }
                yield frame
                sensor_buffer.clear()
                prev_pose = pose


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay ROS bag into localization pipeline.")
    parser.add_argument("--bag", type=Path, required=True, help="Path to ROS bag directory")
    parser.add_argument("--calibration", type=Path, default=Path("data/calib/sample_calib.yaml"))
    parser.add_argument("--grid-resolution", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, nargs=3, default=(200, 200, 8))
    parser.add_argument("--lidar-topic", action="append", default=[], help="LiDAR topics to ingest")
    parser.add_argument("--radar-topic", action="append", default=[], help="Radar topics to ingest")
    parser.add_argument("--odom-topic", type=str, default="/odometry")
    parser.add_argument("--rate", type=float, default=10.0)
    args = parser.parse_args()

    grid_cfg = OccupancyGridConfig(resolution=args.grid_resolution, size=tuple(args.grid_size))
    replay_rosbag(
        bag_path=args.bag,
        calibration_path=args.calibration,
        lidar_topics=args.lidar_topic,
        radar_topics=args.radar_topic,
        odom_topic=args.odom_topic,
        grid_config=grid_cfg,
        publish_rate_hz=args.rate,
    )


if __name__ == "__main__":
    main()
