import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: F841

from hd_map_builder.localization import Pose2
from hd_map_builder.mapping import OccupancyGridConfig
from hd_map_builder.sensors import CalibrationSet, SensorCalibration
from hd_map_builder.sensors.calibration import Extrinsics
from pipeline.offline_builder import FrameSensors, OfflineMapBuilder


def _calibration():
    lidar = SensorCalibration(
        name="lidar",
        frame_id="lidar",
        extrinsics=Extrinsics(
            translation=np.array([0.0, 0.0, 0.0]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        ),
        intrinsics={},
    )
    radar = SensorCalibration(
        name="radar",
        frame_id="radar",
        extrinsics=Extrinsics(
            translation=np.array([0.5, 0.0, 0.0]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        ),
        intrinsics={},
    )
    return CalibrationSet([lidar, radar])


def test_offline_map_builder_processes_frames_and_exports_dataset():
    calib = _calibration()
    grid_cfg = OccupancyGridConfig(resolution=0.5, size=(6, 6, 2))
    builder = OfflineMapBuilder(calibration=calib, grid_config=grid_cfg)

    frame1 = FrameSensors(
        pointclouds={
            "lidar": np.array([[0.1, 0.0, 0.0]]),
            "radar": np.array([[0.0, 0.0, 0.0]]),
        },
        semantics={"lidar": np.array([5]), "radar": np.array([7])},
    )
    builder.process_frame(Pose2(0.5, 0.0, 0.0), frame1)

    frame2 = FrameSensors(
        pointclouds={
            "lidar": np.array([[0.6, 0.0, 0.0]]),
        },
    )
    builder.process_frame(Pose2(0.5, 0.0, 0.0), frame2)

    builder.optimize()

    assert builder.frames_processed() == 2
    occupied = builder.occupancy_points(threshold=0.4)
    assert occupied.shape[0] >= 1

    dataset = builder.build_training_dataset(samples=16, semantics=True, seed=2)
    assert len(dataset) == 16
    sample = dataset[0]
    assert "coords" in sample and "occupancy" in sample
