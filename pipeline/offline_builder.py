"""Offline map builder pipeline tying fusion, localization, and neural sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from hd_map_builder.localization import Pose2, PoseGraph
from hd_map_builder.mapping import (
    MultiSensorOccupancyFusion,
    OccupancyGridConfig,
)
from hd_map_builder.mapping.multi_sensor_fusion import SensorFusionConfig
from hd_map_builder.neural_repr import (
    OccupancySampleConfig,
    OccupancySampleDataset,
    TORCH_AVAILABLE as NEURAL_TORCH_AVAILABLE,
)
from hd_map_builder.sensors import CalibrationSet, load_calibration


@dataclass
class FrameSensors:
    """Container for per-frame sensor inputs."""

    pointclouds: Mapping[str, np.ndarray]
    semantics: Optional[Mapping[str, np.ndarray]] = None


class OfflineMapBuilder:
    """Incrementally fuse sensor frames, maintain pose graph, and export training samples."""

    def __init__(
        self,
        calibration: CalibrationSet,
        grid_config: OccupancyGridConfig,
        *,
        sensor_weights: Optional[Mapping[str, float]] = None,
        origin_pose: Pose2 = Pose2(0.0, 0.0, 0.0),
    ):
        self.calibration = calibration
        fusion_cfg = SensorFusionConfig(grid_config=grid_config, sensor_weights=sensor_weights)
        self.fusion = MultiSensorOccupancyFusion(calibration, fusion_cfg)
        self.pose_graph = PoseGraph()
        self.origin_pose = origin_pose
        self.node_ids: list[int] = []
        self._ensure_origin()

    @classmethod
    def from_calibration_file(
        cls,
        calibration_path: str | Path,
        grid_config: OccupancyGridConfig,
        *,
        sensor_weights: Optional[Mapping[str, float]] = None,
        origin_pose: Pose2 = Pose2(0.0, 0.0, 0.0),
    ) -> "OfflineMapBuilder":
        calibration = load_calibration(calibration_path)
        return cls(
            calibration=calibration,
            grid_config=grid_config,
            sensor_weights=sensor_weights,
            origin_pose=origin_pose,
        )

    def _ensure_origin(self) -> None:
        if not self.node_ids:
            node_id = self.pose_graph.add_node(self.origin_pose, fixed=True)
            self.node_ids.append(node_id)

    def process_frame(self, odometry: Pose2, sensors: FrameSensors) -> None:
        """Integrate one frame worth of odometry and sensor point clouds."""
        prev_node_id = self.node_ids[-1]
        prev_pose = self.pose_graph.nodes[prev_node_id].pose
        estimated_pose = prev_pose.compose(odometry)
        node_id = self.pose_graph.add_node(estimated_pose)
        self.pose_graph.add_odometry_edge(prev_node_id, node_id, odometry)
        self.node_ids.append(node_id)

        semantics = sensors.semantics or {}
        for sensor_name, points in sensors.pointclouds.items():
            sem = semantics.get(sensor_name)
            self.fusion.integrate_point_cloud(sensor_name, points, semantics=sem)

    def optimize(self, max_iterations: int = 10, tol: float = 1e-5) -> None:
        """Optimize pose graph to refine trajectory."""
        self.pose_graph.optimize(max_iterations=max_iterations, tol=tol)

    def occupancy_points(self, threshold: float = 0.5) -> np.ndarray:
        """Return occupied cell centers above a probability threshold."""
        return self.fusion.grid.occupied_points(threshold=threshold)

    def build_training_dataset(
        self,
        *,
        samples: int = 2048,
        semantics: bool = True,
        seed: int = 13,
    ) -> OccupancySampleDataset:
        """Convert fused grid to a dataset suitable for neural refinement."""
        if not NEURAL_TORCH_AVAILABLE or OccupancySampleDataset is None:
            raise RuntimeError("PyTorch is required to build training datasets.")
        dataset = OccupancySampleDataset(
            self.fusion.grid,
            config=OccupancySampleConfig(samples=samples, semantics=semantics, seed=seed),
        )
        return dataset

    def frames_processed(self) -> int:
        """Return number of odometry frames ingested (excluding origin)."""
        return max(0, len(self.node_ids) - 1)
