from pathlib import Path

import numpy as np

from hd_map_builder.sensors import load_calibration


def test_load_calibration_returns_expected_sensors(tmp_path: Path):
    calib_path = Path("data/calib/sample_calib.yaml")
    calib = load_calibration(calib_path)

    assert "lidar_top" in calib
    assert "camera_front" in calib

    lidar = calib["lidar_top"]
    np.testing.assert_allclose(lidar.extrinsics.translation, [0.0, 0.0, 1.8])

    imu = calib["imu"]
    mat = imu.extrinsics.as_matrix()
    np.testing.assert_allclose(mat[:3, 3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(mat[:3, :3], np.eye(3))
    assert imu.intrinsics_for("gyro_noise") == 0.001
