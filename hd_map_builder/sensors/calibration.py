"""Sensor calibration loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np
import yaml


def _rpy_deg_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll/pitch/yaw in degrees to rotation matrix."""
    r, p, y = np.deg2rad([roll, pitch, yaw])
    sr, cr = np.sin(r), np.cos(r)
    sp, cp = np.sin(p), np.cos(p)
    sy, cy = np.sin(y), np.cos(y)

    rot_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    rot_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rot_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to XYZW quaternion."""
    m = matrix
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=float)


@dataclass
class Extrinsics:
    translation: np.ndarray  # xyz
    quaternion: np.ndarray  # xyzw

    def as_matrix(self) -> np.ndarray:
        """Return 4x4 homogeneous transform."""
        mat = np.eye(4)
        mat[:3, 3] = self.translation
        x, y, z, w = self.quaternion
        mat[:3, :3] = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ],
            dtype=float,
        )
        return mat


@dataclass
class SensorCalibration:
    name: str
    frame_id: str
    extrinsics: Extrinsics
    intrinsics: Mapping[str, Any]

    def intrinsics_for(self, key: str, default: Optional[Any] = None) -> Any:
        return self.intrinsics.get(key, default)


class CalibrationSet(Mapping[str, SensorCalibration]):
    """Dictionary-like accessor for sensor calibrations."""

    def __init__(self, sensors: Iterable[SensorCalibration]):
        self._sensors: Dict[str, SensorCalibration] = {s.name: s for s in sensors}

    def __getitem__(self, key: str) -> SensorCalibration:
        return self._sensors[key]

    def __iter__(self):
        return iter(self._sensors)

    def __len__(self) -> int:
        return len(self._sensors)

    def as_dict(self) -> Mapping[str, SensorCalibration]:
        return dict(self._sensors)


def _load_sensor(name: str, cfg: MutableMapping[str, Any]) -> SensorCalibration:
    frame_id = cfg.get("frame_id", name)
    extr = cfg.get("extrinsics", {})
    translation = np.array(extr.get("translation", [0.0, 0.0, 0.0]), dtype=float)
    if "quaternion_xyzw" in extr:
        quat = np.array(extr["quaternion_xyzw"], dtype=float)
    else:
        rpy = extr.get("rotation_rpy_deg", [0.0, 0.0, 0.0])
        rot = _rpy_deg_to_matrix(*rpy)
        quat = _matrix_to_quaternion(rot)
    intrinsics = dict(cfg.get("intrinsics", {}))
    extrinsics = Extrinsics(translation=translation, quaternion=quat)
    return SensorCalibration(
        name=name,
        frame_id=frame_id,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )


def load_calibration(path: str | Path) -> CalibrationSet:
    """Load calibration YAML into structured dataclasses."""
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    sensors_cfg = data.get("sensors", {})
    sensors = [_load_sensor(name, cfg) for name, cfg in sensors_cfg.items()]
    return CalibrationSet(sensors)
