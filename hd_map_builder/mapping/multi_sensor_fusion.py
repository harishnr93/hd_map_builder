"""Multi-sensor occupancy fusion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from hd_map_builder.sensors import CalibrationSet

from .occupancy_grid import OccupancyGrid, OccupancyGridConfig


@dataclass
class SensorFusionConfig:
    grid_config: OccupancyGridConfig
    sensor_weights: Optional[Mapping[str, float]] = None


class MultiSensorOccupancyFusion:
    """Project LiDAR/radar points into a unified occupancy grid."""

    def __init__(self, calibration: CalibrationSet, config: SensorFusionConfig):
        self.calibration = calibration
        self.config = config
        self.grid = OccupancyGrid(config.grid_config)

    def integrate_point_cloud(
        self,
        sensor_name: str,
        points_sensor: np.ndarray,
        *,
        semantics: Optional[np.ndarray] = None,
    ) -> int:
        if sensor_name not in self.calibration:
            raise KeyError(f"Unknown sensor '{sensor_name}'")
        extr = self.calibration[sensor_name].extrinsics
        vehicle_points = extr.transform_points(points_sensor)
        weight = self._sensor_weight(sensor_name)
        return self.grid.integrate_hits(vehicle_points, semantics=semantics, weight=weight)

    def decay_free_space(
        self,
        sensor_name: str,
        free_points_sensor: np.ndarray,
    ) -> int:
        if sensor_name not in self.calibration:
            raise KeyError(f"Unknown sensor '{sensor_name}'")
        extr = self.calibration[sensor_name].extrinsics
        vehicle_points = extr.transform_points(free_points_sensor)
        weight = self._sensor_weight(sensor_name)
        return self.grid.integrate_free(vehicle_points, weight=weight)

    def _sensor_weight(self, sensor_name: str) -> float:
        if not self.config.sensor_weights:
            return 1.0
        return self.config.sensor_weights.get(sensor_name, 1.0)
