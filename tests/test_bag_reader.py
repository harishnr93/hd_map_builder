from dataclasses import dataclass

from hd_map_builder.sensors import BagSensorStream


@dataclass
class DummyConnection:
    topic: str


class DummyReader:
    def __init__(self, records):
        self._records = records
        self.closed = False

    def messages(self):
        for record in self._records:
            yield record

    def close(self):
        self.closed = True


def _factory(*_args, **_kwargs):
    records = [
        (DummyConnection("/lidar"), 1, b"a"),
        (DummyConnection("/camera"), 2, b"b"),
        (DummyConnection("/lidar"), 4, b"c"),
    ]
    return DummyReader(records)


def test_bag_sensor_stream_filters_topic_and_time_bounds():
    stream = BagSensorStream("dummy", reader_factory=_factory)
    with stream:
        msgs = list(stream.iter_messages(topics=["/lidar"], start_ns=2, end_ns=3))

    assert len(msgs) == 0

    with BagSensorStream("dummy", reader_factory=_factory) as stream2:
        msgs = list(stream2.iter_messages(topics=["/lidar"], end_ns=3))

    assert [m.rawdata for m in msgs] == [b"a"]
