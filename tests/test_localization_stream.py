from pathlib import Path

import numpy as np

from hd_map_builder.mapping import OccupancyGridConfig
from pipeline.localization_publisher import BufferingPublisher
from pipeline.localization_stream import LocalizationStreamer, _load_frames


def test_localization_streamer_publishes_for_each_frame():
    publisher = BufferingPublisher()
    streamer = LocalizationStreamer.from_paths(
        Path("data/calib/sample_calib.yaml"),
        grid_config=OccupancyGridConfig(resolution=0.5, size=(10, 10, 2)),
        publish_rate_hz=5.0,
        publisher=publisher,
    )
    frames = list(_load_frames(Path("data/sample_frames.json")))
    count = streamer.stream_frames(frames)

    assert count == len(frames)
    assert len(publisher.records) == len(frames)
    timestamps = [ts for ts, _ in publisher.records]
    assert timestamps == sorted(timestamps)
