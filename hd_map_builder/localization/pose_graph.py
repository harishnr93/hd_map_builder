"""Lightweight SE(2) pose graph optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .poses import Pose2


@dataclass
class PoseNode:
    pose: Pose2
    fixed: bool = False


@dataclass
class PoseEdge:
    i: int
    j: int
    measurement: Pose2
    information: np.ndarray

    def __post_init__(self):
        if self.information.shape != (3, 3):
            raise ValueError("Information matrix must be 3x3.")

    def residual(self, pose_i: Pose2, pose_j: Pose2) -> np.ndarray:
        predicted = pose_i.between(pose_j)
        return predicted.minus(self.measurement)

    def jacobian(
        self, pose_i: Pose2, pose_j: Pose2, *, wrt: str, epsilon: float = 1e-6
    ) -> np.ndarray:
        """Numerical Jacobian of residual wrt pose i or j."""
        base_res = self.residual(pose_i, pose_j)
        J = np.zeros((3, 3))
        for k in range(3):
            delta = np.zeros(3)
            delta[k] = epsilon
            if wrt == "i":
                perturbed = pose_i.perturb(delta)
                res = self.residual(perturbed, pose_j)
            elif wrt == "j":
                perturbed = pose_j.perturb(delta)
                res = self.residual(pose_i, perturbed)
            else:
                raise ValueError("wrt must be 'i' or 'j'")
            J[:, k] = (res - base_res) / epsilon
        return J


class PoseGraph:
    """Container for SE(2) pose graph with Gauss-Newton solver."""

    def __init__(self):
        self.nodes: List[PoseNode] = []
        self.edges: List[PoseEdge] = []

    def add_node(self, pose: Pose2, fixed: bool = False) -> int:
        node_id = len(self.nodes)
        self.nodes.append(PoseNode(pose=pose, fixed=fixed))
        return node_id

    def add_odometry_edge(
        self,
        i: int,
        j: int,
        measurement: Pose2,
        information: Optional[np.ndarray] = None,
    ) -> None:
        if information is None:
            information = np.eye(3)
        edge = PoseEdge(i=i, j=j, measurement=measurement, information=information)
        self.edges.append(edge)

    def optimize(self, max_iterations: int = 10, tol: float = 1e-5) -> None:
        optimizer = PoseGraphOptimizer(self)
        optimizer.optimize(max_iterations=max_iterations, tol=tol)


class PoseGraphOptimizer:
    """Numeric optimizer for PoseGraph."""

    def __init__(self, graph: PoseGraph):
        self.graph = graph

    def optimize(self, max_iterations: int, tol: float) -> None:
        if not any(node.fixed for node in self.graph.nodes):
            raise ValueError("Pose graph requires at least one fixed node.")

        variable_index: Dict[int, int] = {}
        for node_id, node in enumerate(self.graph.nodes):
            if not node.fixed:
                variable_index[node_id] = len(variable_index)
        if not variable_index:
            return

        dim = 3 * len(variable_index)

        for _ in range(max_iterations):
            H = np.zeros((dim, dim))
            b = np.zeros(dim)

            for edge in self.graph.edges:
                pose_i = self.graph.nodes[edge.i].pose
                pose_j = self.graph.nodes[edge.j].pose
                res = edge.residual(pose_i, pose_j)
                info = edge.information

                idx_i = variable_index.get(edge.i)
                idx_j = variable_index.get(edge.j)
                J_i = edge.jacobian(pose_i, pose_j, wrt="i") if idx_i is not None else None
                J_j = edge.jacobian(pose_i, pose_j, wrt="j") if idx_j is not None else None

                if idx_i is not None:
                    slice_i = slice(3 * idx_i, 3 * idx_i + 3)
                    H[slice_i, slice_i] += J_i.T @ info @ J_i
                    b[slice_i] += J_i.T @ info @ res
                if idx_j is not None:
                    slice_j = slice(3 * idx_j, 3 * idx_j + 3)
                    H[slice_j, slice_j] += J_j.T @ info @ J_j
                    b[slice_j] += J_j.T @ info @ res

                if idx_i is not None and idx_j is not None:
                    slice_i = slice(3 * idx_i, 3 * idx_i + 3)
                    slice_j = slice(3 * idx_j, 3 * idx_j + 3)
                    H[slice_i, slice_j] += J_i.T @ info @ J_j
                    H[slice_j, slice_i] += J_j.T @ info @ J_i

            # Regularize to avoid singularity
            H += np.eye(dim) * 1e-9
            try:
                delta = -np.linalg.solve(H, b)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError("Pose graph linear system is singular.") from exc

            max_update = 0.0
            for node_id, var_idx in variable_index.items():
                slice_idx = slice(3 * var_idx, 3 * var_idx + 3)
                update = delta[slice_idx]
                max_update = max(max_update, np.linalg.norm(update))
                node = self.graph.nodes[node_id]
                node.pose = node.pose.perturb(update)

            if max_update < tol:
                break
