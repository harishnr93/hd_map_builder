import numpy as np

from scripts.replay_rosbag import pointcloud2_to_xyz, quaternion_to_yaw, odom_to_pose
from hd_map_builder.localization import Pose2


class DummyPointCloud:
    def __init__(self, points):
        self.point_step = 16
        self.data = np.array(points, dtype=np.float32).tobytes()


class DummyPose:
    def __init__(self, x, y, theta):
        self.pose = type(
            "PoseStruct",
            (),
            {
                "pose": type(
                    "Pose",
                    (),
                    {
                        "position": type("Pos", (), {"x": x, "y": y, "z": 0.0})(),
                        "orientation": type(
                            "Ori",
                            (),
                            {
                                "x": 0.0,
                                "y": 0.0,
                                "z": np.sin(theta / 2),
                                "w": np.cos(theta / 2),
                            },
                        )(),
                    },
                )()
            },
        )()


def test_pointcloud2_to_xyz_returns_array():
    msg = DummyPointCloud([[1.0, 2.0, 3.0, 0.0]])
    pts = pointcloud2_to_xyz(msg)
    assert pts.shape == (1, 3)
    np.testing.assert_allclose(pts[0], [1.0, 2.0, 3.0])


def test_quaternion_to_yaw():
    yaw = quaternion_to_yaw(0.0, 0.0, 0.0, 1.0)
    assert yaw == 0.0


def test_odom_to_pose_extracts_pose():
    msg = DummyPose(2.0, -1.0, np.pi / 4)
    pose = odom_to_pose(msg)
    assert isinstance(pose, Pose2)
    assert pose.x == 2.0
    assert pose.y == -1.0
