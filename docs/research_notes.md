# Research Notes & Future Work

## Current Capabilities

- Occupancy fusion handles LiDAR + radar inputs with semantic counts and free-space decay.
- Pose graph optimizer supports SE(2) odometry chains and loop edges; IMU factors can be added via additional edges.
- Neural implicit decoder learns SDF + semantics from occupancy samples and can be exported to ONNX for deployment, with profiling data available via CLI.
- Simulation runner produces deterministic drift metrics to validate algorithmic changes.

## Open Research Directions

1. **IMU Preintegration & Loop Closures** – integrate IMU noise models and scan-context-based closures for tighter online localization.
2. **Neural Map Compression** – experiment with neural SDF distillation techniques (e.g., Instant-NGP) to reduce memory footprint while matching occupancy fidelity.
3. **Uncertainty-Aware Fusion** – incorporate per-sensor uncertainty (radar elevation ambiguity, LiDAR rain attenuation) into log-odds updates.
4. **Dynamic Object Filtering** – add semantics or motion detection to remove moving objects prior to neural training, improving map stability.
5. **Real-World Benchmarks** – replay public datasets (nuScenes, Waymo Open) to benchmark drift vs. baseline SLAM pipelines and compare neural map accuracy.

## Publication & Community Contributions

- Potential whitepaper: *“Streaming Neural Occupancy Maps from Heterogeneous Sensors”* – outline the pipeline, simulation validation, and runtime characteristics.
- Open-source contributions: upstream occupancy fusion semantics and PyTorch export utilities once tested on larger datasets.

## Next Engineering Steps

1. Wrap the localization streamer into a ROS 2 node, publishing `nav_msgs/Odometry`.
2. Build CARLA/LGSVL scenarios that exercise localization under dynamic objects; log metrics into the benchmarking table.
3. Automate ONNX Runtime / TensorRT benchmarking scripts to compare CPU vs. GPU throughput on the trained implicit decoder.
4. Add integration tests that replay short ROS bags (or synthetic rosbag2 archives) end-to-end, verifying localization output matches expected trajectories. Consider using rosbags' writer API to build deterministic bags for CI.
