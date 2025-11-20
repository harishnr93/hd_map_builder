from math import pi

from pipeline.ros_localization_node import yaw_to_quaternion


def test_yaw_to_quaternion_identity():
    q = yaw_to_quaternion(0.0)
    assert q == (0.0, 0.0, 0.0, 1.0)


def test_yaw_to_quaternion_half_turn():
    q = yaw_to_quaternion(pi)
    assert q[2] == 1.0 or q[2] == -1.0
    assert abs(q[3]) < 1e-6
