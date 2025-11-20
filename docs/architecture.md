# Architecture Overview

The system is organized around five layers:

1. **Sensor IO** – ROS bag ingestion and calibration loading (`hd_map_builder/sensors`). Provides calibrated extrinsics/intrinsics and timestamped raw messages.
2. **Mapping Core** – Occupancy grid fusion and semantic accumulation (`hd_map_builder/mapping`). Multi-sensor point clouds are projected into the vehicle frame and update log-odds voxels.
3. **Localization** – Pose graph manager (`hd_map_builder/localization`) and streaming helpers (`pipeline/localization_stream.py`) keep track of vehicle pose estimates and publish them in real time.
4. **Neural Refinement** – PyTorch decoder, datasets, training loop, and export/profiling utilities (`hd_map_builder/neural_repr`). Converts occupancy data into neural implicit maps.
5. **Pipelines & Tooling** – Offline builder, replay CLI, simulation runner, PLY/ONNX exporters, and localization streamer (`pipeline/*`, `scripts/*`).

```mermaid
flowchart LR
    subgraph Sensors
        S1[ROS Bag] -->|BagSensorStream| IO1
        S2[Calibration YAML] -->|load_calibration| IO1[Sensor IO]
    end
    IO1 -->|calibrated clouds| FUSION[Occupancy Fusion]
    IO1 -->|odometry| SLAM[Pose Graph]
    FUSION -->|grid| NEURAL[Neural Refinement]
    SLAM -->|poses| STREAM[Localization Streamer]
    FUSION -->|voxels| STREAM
    STREAM -->|published poses| CLIENTS[Planning / Logging]
    NEURAL -->|training + export| DEPLOY[Deployment]
    FUSION -->|PLY/metrics| DOCS[Simulation & Docs]
```

All components are designed to work offline (unit tests) and via CLI demos to avoid heavy runtime dependencies while keeping the structure close to an eventual ROS/embedded deployment.
