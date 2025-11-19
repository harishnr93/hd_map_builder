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
- PyTorch 2.x, NumPy, SciPy
- ROS bag parsing via `rosbags` (no ROS runtime required)

## Testing & Logs

Run the full suite and persist logs (timestamped) via:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
scripts/run_tests.sh
```

Pytest stdout is captured into `logs/pytest_<timestamp>.log` for later review, keeping historical runtime artifacts under version control.

## Next Steps

Initialize Python package scaffolding, add ingestion utilities, and back them with unit tests before moving on to the mapping fusion module.
