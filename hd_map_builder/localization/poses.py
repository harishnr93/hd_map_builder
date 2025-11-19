"""SE(2) pose helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
    # Handle -pi edge case
    if wrapped == -math.pi:
        return math.pi
    return wrapped


@dataclass
class Pose2:
    x: float
    y: float
    theta: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=float)

    @classmethod
    def from_vector(cls, vec: Iterable[float]) -> "Pose2":
        x, y, theta = vec
        return cls(float(x), float(y), wrap_angle(float(theta)))

    def inverse(self) -> "Pose2":
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        x = -(c * self.x + s * self.y)
        y = -(-s * self.x + c * self.y)
        return Pose2(x, y, wrap_angle(-self.theta))

    def compose(self, other: "Pose2") -> "Pose2":
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        x = self.x + c * other.x - s * other.y
        y = self.y + s * other.x + c * other.y
        theta = wrap_angle(self.theta + other.theta)
        return Pose2(x, y, theta)

    def between(self, other: "Pose2") -> "Pose2":
        return self.inverse().compose(other)

    def minus(self, other: "Pose2") -> np.ndarray:
        diff = self.as_vector() - other.as_vector()
        diff[2] = wrap_angle(diff[2])
        return diff

    def perturb(self, delta: Iterable[float]) -> "Pose2":
        dx, dy, dtheta = delta
        return Pose2(self.x + dx, self.y + dy, wrap_angle(self.theta + dtheta))
