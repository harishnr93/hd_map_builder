# Benchmark Snapshots

| Scenario | Metric | Value | Notes |
| --- | --- | --- | --- |
| Synthetic circle sim (steps=60, laps=2, seed=11) | RMS translation drift | 0.19 m | `logs/sim_circle_metrics.json`. |
| Synthetic circle sim | Max translation drift | 0.32 m | yaw drift mean 1.16°. |
| Synthetic straight sim (steps=30, seed=11) | RMS translation drift | 0.17 m | `logs/sim_metrics_latest.json`. |
| Localization streamer | Latency | ~0.0 s (offline) | Publishes immediately per frame; add `--realtime` to enforce wall timing. |
| Neural decoder profiling (CUDA, batch=4096) | Avg latency | 0.773 ms | `python scripts/profile_decoder.py --batch-size 4096 --steps 100`. |
| Neural decoder profiling (CUDA, batch=4096) | Throughput | 5.30 M samples/s | Same run as above. |
| ONNX Runtime benchmark (CPU, batch=2048) | Throughput | TBD (install onnxruntime) | `python scripts/ort_profile.py --model logs/implicit_decoder.onnx` (requires onnxruntime). |

> These numbers are for the synthetic demo hardware (developer workstation with CUDA-enabled GPU). When running on embedded hardware, rerun the CLI commands and append results to this table.
