#!/usr/bin/env python
"""Synthetic simulation validation for OfflineMapBuilder."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from hd_map_builder.localization import Pose2
from hd_map_builder.mapping import OccupancyGridConfig
from hd_map_builder.sensors import CalibrationSet
from pipeline.offline_builder import FrameSensors, OfflineMapBuilder


@dataclass
class SimulationConfig:
    steps: int = 20
    step_dx: float = 0.5
    step_dtheta: float = 0.0
    odom_noise: float = 0.02
    yaw_noise: float = 0.01
    sensor_noise: float = 0.05
    seed: int = 7
    landmarks: int = 8
    trajectory: str = "straight"  # options: straight, circle, figure8
    laps: int = 1


def _generate_landmarks(rng: np.random.Generator, count: int) -> List[Tuple[np.ndarray, int]]:
    positions = rng.uniform(low=[2.0, -4.0, -0.2], high=[12.0, 4.0, 0.2], size=(count, 3))
    semantics = rng.integers(1, 10, size=count)
    return [(positions[i], int(semantics[i])) for i in range(count)]


def _world_to_vehicle(point: np.ndarray, pose: Pose2) -> np.ndarray:
    dx = point[0] - pose.x
    dy = point[1] - pose.y
    c = np.cos(pose.theta)
    s = np.sin(pose.theta)
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy
    return np.array([x_local, y_local, point[2]], dtype=float)


def _vehicle_to_sensor(point_vehicle: np.ndarray, calibration: CalibrationSet, sensor: str) -> np.ndarray:
    extr = calibration[sensor].extrinsics
    rot = extr.rotation_matrix()
    translated = point_vehicle - extr.translation
    return translated @ rot


def _build_frame(
    calibration: CalibrationSet,
    pose: Pose2,
    landmarks: List[Tuple[np.ndarray, int]],
    rng: np.random.Generator,
    config: SimulationConfig,
) -> FrameSensors:
    pointclouds: Dict[str, np.ndarray] = {}
    semantics: Dict[str, np.ndarray] = {}
    sensors = ["lidar_top", "radar_front"]
    for sensor in sensors:
        points = []
        sem = []
        for position, cls in landmarks:
            local = _world_to_vehicle(position, pose)
            if np.linalg.norm(local[:2]) > 20.0:
                continue
            sensor_point = _vehicle_to_sensor(local, calibration, sensor)
            noise = rng.normal(scale=config.sensor_noise, size=3)
            sensor_point = sensor_point + noise
            points.append(sensor_point)
            sem.append(cls)
        if points:
            pointclouds[sensor] = np.asarray(points, dtype=float)
            semantics[sensor] = np.asarray(sem, dtype=int)
    return FrameSensors(pointclouds=pointclouds, semantics=semantics or None)


def simulate_and_report(
    *,
    calibration_path: Path,
    grid_config: OccupancyGridConfig,
    sim_config: SimulationConfig,
) -> dict:
    builder = OfflineMapBuilder.from_calibration_file(
        calibration_path,
        grid_config=grid_config,
    )
    rng = np.random.default_rng(sim_config.seed)
    landmarks = _generate_landmarks(rng, sim_config.landmarks)
    gt_poses = [Pose2(0.0, 0.0, 0.0)]

    step_count = sim_config.steps * max(sim_config.laps, 1)
    trajectory = sim_config.trajectory.lower()
    for step_idx in range(step_count):
        if trajectory == "circle":
            angle = (2 * np.pi * step_idx) / step_count
            true_delta = Pose2(
                sim_config.step_dx,
                0.0,
                (2 * np.pi / step_count) * sim_config.laps,
            )
        elif trajectory == "figure8":
            angle = (2 * np.pi * step_idx) / step_count
            true_delta = Pose2(
                sim_config.step_dx * np.cos(angle),
                sim_config.step_dx * np.sin(angle),
                sim_config.step_dtheta + 0.05 * np.sin(angle),
            )
        else:
            true_delta = Pose2(sim_config.step_dx, 0.0, sim_config.step_dtheta)
        noisy_delta = Pose2(
            true_delta.x + rng.normal(scale=sim_config.odom_noise),
            true_delta.y + rng.normal(scale=sim_config.odom_noise),
            true_delta.theta + rng.normal(scale=sim_config.yaw_noise),
        )
        sensors = _build_frame(builder.calibration, gt_poses[-1], landmarks, rng, sim_config)
        builder.process_frame(noisy_delta, sensors)
        gt_poses.append(gt_poses[-1].compose(true_delta))

    builder.optimize()
    metrics = _compute_metrics(builder, gt_poses)
    metrics["occupied_cells"] = int(builder.occupancy_points().shape[0])
    metrics["frames"] = builder.frames_processed()
    return metrics


def _compute_metrics(builder: OfflineMapBuilder, gt_poses: List[Pose2]) -> dict:
    errors = []
    yaw_errors = []

    for idx in range(1, len(builder.pose_graph.nodes)):
        est_pose = builder.pose_graph.nodes[idx].pose
        gt_pose = gt_poses[idx]
        dx = est_pose.x - gt_pose.x
        dy = est_pose.y - gt_pose.y
        errors.append(np.hypot(dx, dy))
        yaw_errors.append(abs(est_pose.theta - gt_pose.theta))

    errors = np.asarray(errors)
    yaw_errors = np.asarray(yaw_errors)
    rms = float(np.sqrt(np.mean(errors**2)))
    max_err = float(np.max(errors)) if errors.size else 0.0
    yaw_deg = float(np.mean(yaw_errors) * 180.0 / np.pi)
    return {
        "rms_translation_m": rms,
        "max_translation_m": max_err,
        "mean_yaw_deg": yaw_deg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic simulation validation runner.")
    parser.add_argument("--calibration", type=Path, default=Path("data/calib/sample_calib.yaml"))
    parser.add_argument("--grid-resolution", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, nargs=3, default=(40, 40, 4))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--step-dx", type=float, default=0.5)
    parser.add_argument("--step-dtheta", type=float, default=0.0)
    parser.add_argument("--odom-noise", type=float, default=0.02)
    parser.add_argument("--yaw-noise", type=float, default=0.01)
    parser.add_argument("--sensor-noise", type=float, default=0.05)
    parser.add_argument("--landmarks", type=int, default=10)
    parser.add_argument("--trajectory", type=str, default="straight", choices=["straight", "circle", "figure8"])
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, help="Optional path to save metrics JSON.")
    args = parser.parse_args()

    sim_config = SimulationConfig(
        steps=args.steps,
        step_dx=args.step_dx,
        step_dtheta=args.step_dtheta,
        odom_noise=args.odom_noise,
        yaw_noise=args.yaw_noise,
        sensor_noise=args.sensor_noise,
        landmarks=args.landmarks,
        seed=args.seed,
        trajectory=args.trajectory,
        laps=args.laps,
    )
    grid_config = OccupancyGridConfig(resolution=args.grid_resolution, size=tuple(args.grid_size))
    metrics = simulate_and_report(
        calibration_path=args.calibration,
        grid_config=grid_config,
        sim_config=sim_config,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
