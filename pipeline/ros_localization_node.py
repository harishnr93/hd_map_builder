"""ROS2 localization node integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from hd_map_builder.localization import Pose2
from hd_map_builder.mapping import OccupancyGridConfig
from .localization_publisher import PosePublisher
from .localization_stream import LocalizationStreamer, _load_frames

try:  # pragma: no cover - ROS2 optional dependency
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
except ImportError:  # pragma: no cover
    rclpy = None
    Node = object  # type: ignore
    Odometry = None  # type: ignore


def yaw_to_quaternion(theta: float) -> tuple[float, float, float, float]:
    """Convert planar yaw angle to XYZW quaternion."""
    half = theta * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


@dataclass
class RosPosePublisher(PosePublisher):
    node: Node
    topic: str
    frame_id: str = "map"

    def __post_init__(self):
        if Odometry is None:  # pragma: no cover
            raise RuntimeError("ROS2 libraries not available.")
        self.publisher = self.node.create_publisher(Odometry, self.topic, 10)

    def publish(self, timestamp_ns: int, pose: Pose2) -> None:
        if Odometry is None:  # pragma: no cover
            raise RuntimeError("ROS2 libraries not available.")
        msg = Odometry()
        msg.header.stamp = rclpy.time.Time(nanoseconds=int(timestamp_ns)).to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.pose.position.x = pose.x
        msg.pose.pose.position.y = pose.y
        msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(pose.theta)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        self.publisher.publish(msg)


def run_ros_localization_node(
    *,
    calibration_path: Path,
    frames_path: Path,
    grid_config: OccupancyGridConfig,
    topic: str = "/localization/odometry",
    publish_rate_hz: float = 10.0,
    realtime: bool = False,
) -> None:
    """Spin a minimal ROS2 node that publishes localization updates."""
    if rclpy is None:  # pragma: no cover
        raise RuntimeError("ROS2 (rclpy) is required to run this node.")

    rclpy.init()
    node = rclpy.create_node("hd_map_localization")
    publisher = RosPosePublisher(node=node, topic=topic)
    streamer = LocalizationStreamer.from_paths(
        calibration_path,
        grid_config=grid_config,
        publish_rate_hz=publish_rate_hz,
        publisher=publisher,
        realtime=realtime,
    )

    try:
        frames = _load_frames(frames_path)
        streamer.stream_frames(frames)
    finally:
        node.destroy_node()
        rclpy.shutdown()
