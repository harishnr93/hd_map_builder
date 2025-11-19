# Neural Multi-Sensor HD Map Builder

Project showcasing multi-sensor mapping for autonomous vehicles. I implement a staged pipeline that ingests ROS bag sensor streams, builds a fused occupancy map, refines localization via pose graph SLAM, and sets up neural implicit map refinement hooks.

## Planned Milestones

1. **Sensor ingestion & calibration** – ROS bag reader utilities, calibration loader, synthetic tests.
2. **LiDAR/radar fusion** – GPU-friendly occupancy grid builder with semantic channels.
3. **Localization pose graph** – Factor graph scaffolding ready to consume odometry + loop closures.
4. **Neural implicit refinement** – PyTorch module placeholder with dataset adapters.
5. **Simulation hooks & docs** – Scripts to replay sample data and README updates.

## Repository Layout (target)

```
hd_map_builder/
├── data/                 # sample sensor logs + calibration yaml
├── mapping/              # core occupancy grid fusion
├── localization/         # pose graph + optimization utilities
├── neural_repr/          # implicit representation modules
├── pipeline/             # orchestration scripts (ROS, CLI)
├── scripts/              # tooling for evaluation and viz
└── tests/                # pytest-based regression suite
```

## Environment

- Python 3.10+
- PyTorch 2.x (for neural implicit stage), NumPy, SciPy
- ROS bag parsing via `rosbags` (no ROS runtime required)

## Neural Implicit Module

`hd_map_builder/neural_repr` contains:

- `ImplicitMapDecoder`: PyTorch MLP predicting signed distance + semantic logits from fused voxel samples.
- `OccupancySampleDataset`: utility to turn `OccupancyGrid` volumes into random training pairs.

Install PyTorch per your platform instructions before running the neural repr tests; the suite skips these tests if PyTorch is absent.

## Integration Pipeline

`pipeline/offline_builder.py` exposes an `OfflineMapBuilder` helper that:

1. Loads calibration (or accepts a `CalibrationSet`) and instantiates `MultiSensorOccupancyFusion`.
2. Maintains a pose graph (`PoseGraph`) while sequential odometry edges and sensor point clouds are ingested via `FrameSensors`.
3. Returns fused occupied points or builds neural training datasets (requires PyTorch).

Example skeleton:

```python
from pipeline.offline_builder import FrameSensors, OfflineMapBuilder
from hd_map_builder.localization import Pose2
from hd_map_builder.mapping import OccupancyGridConfig

builder = OfflineMapBuilder.from_calibration_file(
    "data/calib/sample_calib.yaml",
    grid_config=OccupancyGridConfig(resolution=0.5, size=(200, 200, 10)),
)

builder.process_frame(
    Pose2(0.5, 0.0, 0.0),
    FrameSensors(pointclouds={"lidar_top": lidar_points}),
)
builder.optimize()
dataset = builder.build_training_dataset(samples=4096)
```

Tie this helper into ROS bag replay or simulation feeds to demonstrate end-to-end HD map generation.

### Replay CLI

Run the synthetic demo end-to-end:

```bash
python -m pipeline.replay_cli \
  --calibration data/calib/sample_calib.yaml \
  --frames data/sample_frames.json \
  --grid-resolution 0.5 \
  --grid-size 20 20 4 \
  --dataset-out logs/demo_samples.npz \
  --samples 1024
```

This ingests `data/sample_frames.json`, optimizes the pose graph, prints a summary, and exports a neural training dataset to an `.npz`.

### Export Occupancy Map

Produce a quick visualization-ready point cloud (PLY) from fused occupancy cells:

```bash
python scripts/export_map.py \
  --calibration data/calib/sample_calib.yaml \
  --frames data/sample_frames.json \
  --output logs/fused_map.ply
```

The resulting PLY stores XYZ (and semantic IDs when available) for inspection in CloudCompare or MeshLab.

### Neural Training Demo

After exporting samples, train the implicit decoder:

```bash
python scripts/train_implicit.py \
  --dataset logs/demo_samples.npz \
  --epochs 3 \
  --batch-size 512 \
  --checkpoint logs/implicit_decoder.pt
```

The script reports the final loss and saves a checkpoint (optional). Use the trained decoder for downstream map refinement or visualization experiments.

### Simulation Validation

Run a synthetic scenario that perturbs odometry/sensor noise and reports drift metrics:

```bash
python scripts/simulate_scenario.py \
  --calibration data/calib/sample_calib.yaml \
  --steps 30 \
  --output logs/sim_metrics.json
```

The CLI prints RMS/max translation error, mean yaw drift, frame count, and occupied voxel stats, serving as a lightweight validation loop.

## Testing & Logs

Run the full suite and persist logs (timestamped) via:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
scripts/run_tests.sh
```

Pytest stdout is captured into `logs/pytest_<timestamp>.log` for later review, keeping historical runtime artifacts under version control.

## Next Steps

- Wire pose-graph optimized poses + fused occupancy grids into a ROS/CARLA replay script.
- Extend neural implicit decoder with training loop + loss notebook.
- Add simulation scenarios and benchmarks documenting accuracy + runtime.
