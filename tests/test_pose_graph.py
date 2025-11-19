import numpy as np
import pytest

from hd_map_builder.localization import Pose2, PoseGraph


def test_pose_graph_converges_simple_chain():
    graph = PoseGraph()
    n0 = graph.add_node(Pose2(0.0, 0.0, 0.0), fixed=True)
    n1 = graph.add_node(Pose2(0.5, 0.2, 0.1))
    n2 = graph.add_node(Pose2(2.4, 0.5, -0.2))

    graph.add_odometry_edge(n0, n1, Pose2(1.0, 0.0, 0.0))
    graph.add_odometry_edge(n1, n2, Pose2(1.0, 0.0, 0.0))
    graph.add_odometry_edge(n0, n2, Pose2(2.0, 0.0, 0.0))

    graph.optimize(max_iterations=20, tol=1e-6)

    pose1 = graph.nodes[n1].pose
    pose2 = graph.nodes[n2].pose

    np.testing.assert_allclose([pose1.x, pose1.y], [1.0, 0.0], atol=1e-2)
    assert abs(pose1.theta) < 1e-2
    np.testing.assert_allclose([pose2.x, pose2.y], [2.0, 0.0], atol=1e-2)
    assert abs(pose2.theta) < 1e-2


def test_pose_graph_requires_anchor():
    graph = PoseGraph()
    graph.add_node(Pose2(0.0, 0.0, 0.0))
    graph.add_node(Pose2(1.0, 0.0, 0.0))
    graph.add_odometry_edge(0, 1, Pose2(1.0, 0.0, 0.0))

    with pytest.raises(ValueError):
        graph.optimize()
