"""Simple publisher interfaces for localization streaming."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from hd_map_builder.localization import Pose2


class PosePublisher(Protocol):
    """Interface for publishing vehicle poses."""

    def publish(self, timestamp_ns: int, pose: Pose2) -> None:
        ...


class StdoutPublisher:
    """Human-readable publisher that prints JSON per pose update."""

    def publish(self, timestamp_ns: int, pose: Pose2) -> None:
        payload = {"timestamp_ns": int(timestamp_ns), "x": pose.x, "y": pose.y, "theta": pose.theta}
        print(json.dumps(payload))


@dataclass
class BufferingPublisher:
    """Test helper storing published poses."""

    records: list[tuple[int, Pose2]] = field(default_factory=list)

    def publish(self, timestamp_ns: int, pose: Pose2) -> None:
        self.records.append((timestamp_ns, pose))
