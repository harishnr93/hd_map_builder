# Simulation Validation Plan

We will implement a lightweight scenario generator that:

1. Synthesizes a ground-truth vehicle trajectory (SE(2)) and multi-sensor point clouds (LiDAR + radar) for each timestep.
2. Adds configurable noise to odometry and sensor measurements.
3. Feeds the frames through `OfflineMapBuilder`, optimizes the pose graph, and measures drift between optimized poses and the ground truth.
4. Reports metrics (RMS translation error, max drift, occupied voxel count) as JSON, serving as a quick validation pass in lieu of a full CARLA/LGSVL run.

Deliverables:

- `scripts/simulate_scenario.py` CLI to run the above flow and save metrics.
- Unit tests covering metric calculations and deterministic output with a fixed random seed.
- README section documenting how to run the simulation validation stage.
