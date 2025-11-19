import numpy as np

from hd_map_builder.mapping import MultiSensorOccupancyFusion, OccupancyGridConfig
from hd_map_builder.mapping.multi_sensor_fusion import SensorFusionConfig
from hd_map_builder.sensors import CalibrationSet, SensorCalibration
from hd_map_builder.sensors.calibration import Extrinsics


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
            translation=np.array([1.0, 0.0, 0.0]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        ),
        intrinsics={},
    )
    return CalibrationSet([lidar, radar])


def test_multi_sensor_fusion_transforms_points():
    calib = _calibration()
    grid_cfg = OccupancyGridConfig(resolution=1.0, size=(4, 4, 1), origin=(0.0, 0.0, 0.0))
    fusion = MultiSensorOccupancyFusion(
        calibration=calib,
        config=SensorFusionConfig(grid_config=grid_cfg, sensor_weights={"radar": 0.5}),
    )

    lidar_pts = np.array([[0.2, 0.2, 0.0]])
    radar_pts = np.array([[0.0, 0.0, 0.0]])
    fusion.integrate_point_cloud("lidar", lidar_pts)
    fusion.integrate_point_cloud("radar", radar_pts)

    probs = fusion.grid.occupancy_probabilities()
    # Lidar point remains at x=0, radar shifted by +1m translation
    assert probs[0, 0, 0] > probs[1, 0, 0]
    assert probs[1, 0, 0] > 0
