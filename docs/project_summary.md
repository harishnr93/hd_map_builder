# HD Map Builder – Project Summary

## Implemented Capabilities

| Stage | Highlights | Entry Points |
| --- | --- | --- |
| Sensor Ingestion & Calibration | YAML extrinsics/intrinsics loader; ROS bag reader utilities (`BagSensorStream`). | `hd_map_builder/sensors/`, `tests/test_calibration.py`, `tests/test_bag_reader.py`. |
| LiDAR/Radar Occupancy Fusion | 3D log-odds grid with semantic voting; multi-sensor fusion respecting extrinsics. | `hd_map_builder/mapping/`, `tests/test_occupancy_grid.py`, `tests/test_multi_sensor_fusion.py`. |
| Localization (Pose Graph + Streaming) | SE(2) pose graph optimizer, ROS2 publisher, CLI streamer. | `hd_map_builder/localization/`, `pipeline/localization_stream.py`, `scripts/stream_localization.py`, `scripts/ros_localization_node.py`. |
| Neural Implicit Refinement | PyTorch decoder, dataset samplers, training script, ONNX export, PyTorch/ORT profiling. | `hd_map_builder/neural_repr/`, `scripts/train_implicit.py`, `scripts/export_onnx.py`, `scripts/profile_decoder.py`, `scripts/ort_profile.py`. |
| Simulation & Validation | Synthetic straight/circle/figure-8 scenarios; drift metrics saved to JSON, benchmark table. | `scripts/simulate_scenario.py`, `docs/benchmarks.md`. |
| ROS Bag Replay | `rosbags`-based reader converting bag topics into `LocalizationStreamer` frames. | `scripts/replay_rosbag.py`, `tests/test_replay_rosbag.py`. |
| Packaging | Dockerfile (Python 3.10 slim + deps) for reproducible CLI runs. | `Dockerfile`, README’s Docker section. |

## Pending Improvements

1. **Benchmark Completion**
   - Run `scripts/profile_decoder.py` and `scripts/ort_profile.py` on target hardware (with torch/onnxruntime installed) and update `docs/benchmarks.md` with real latency/throughput numbers. Current entries show “TBD”.

2. **Deterministic ROS Bag Integration Test**
   - Use `rosbags` writer API to generate a tiny ROS2 bag containing PointCloud2 + Odometry messages. Add a regression test that runs `scripts/replay_rosbag.py` against this synthetic bag to ensure localization outputs match expected poses.

3. **Visualization/Portfolio Assets**
   - Export `docs/architecture.md` Mermaid diagram to PNG/SVG, capture PLY screenshots (`logs/fused_map.ply` in CloudCompare), and include terminal/GIF captures of `stream_localization.py`. Store assets under `docs/media/` and reference them in README.

4. **Carla/LGSVL or Real Bag Validation (Optional)**
   - Integrate actual simulator or recorded datasets (nuScenes/Waymo). Feed data through the ROS node and document drift metrics + screenshots to demonstrate robustness beyond synthetic scenes.

5. **TensorRT / Embedded Profiling (Optional)**
   - Convert the ONNX decoder to TensorRT, run on target GPUs (Orin/AGX), and log throughput to showcase automotive deployment readiness.

6. **ROS2 Launch Integration**
   - Provide launch files or `ros2 run` entrypoints that start `ros_localization_node.py` and bag replay simultaneously, easing real-world testing.

## Usage Snapshot

```bash
# Run synthetic replay
python scripts/stream_localization.py --frames data/sample_frames.json

# Train implicit decoder and profile
python scripts/train_implicit.py --dataset logs/demo_samples.npz --epochs 3
python scripts/export_onnx.py --output logs/implicit_decoder.onnx
python scripts/profile_decoder.py --batch-size 4096 --steps 100
python scripts/ort_profile.py --model logs/implicit_decoder.onnx --batch-size 2048

# Replay ROS bag (requires rosbags + PointCloud2/Odometry topics)
python scripts/replay_rosbag.py --bag /path/to/bag --lidar-topic /lidar --odom-topic /odom

# Docker
docker build -t hd-map-builder .
docker run --rm -v "$(pwd)":/app hd-map-builder python scripts/stream_localization.py --frames data/sample_frames.json
```
