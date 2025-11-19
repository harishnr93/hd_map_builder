"""ROS bag sensor ingestion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence


@dataclass
class SensorMessage:
    topic: str
    timestamp_ns: int
    rawdata: bytes


class BagSensorStream:
    """Lightweight wrapper over rosbags Reader with topic/time filtering."""

    def __init__(
        self,
        bag_path: str | Path,
        *,
        storage_id: str = "sqlite3",
        reader_factory: Optional[Callable[[str | Path, str], any]] = None,
    ):
        self.bag_path = Path(bag_path)
        self.storage_id = storage_id
        self._reader_factory = reader_factory or _default_reader_factory
        self._reader = None
        self._opened = False

    def open(self) -> None:
        if self._opened:
            return
        self._reader = self._reader_factory(self.bag_path, self.storage_id)
        self._opened = True

    def close(self) -> None:
        if self._reader and hasattr(self._reader, "close"):
            self._reader.close()
        self._reader = None
        self._opened = False

    def __enter__(self) -> "BagSensorStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def iter_messages(
        self,
        topics: Optional[Sequence[str]] = None,
        *,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
    ) -> Iterator[SensorMessage]:
        if not self._opened or self._reader is None:
            raise RuntimeError("Stream must be opened before iterating messages.")

        allowed_topics = set(topics) if topics else None
        for connection, timestamp, rawdata in self._reader.messages():
            topic = getattr(connection, "topic", None)
            if allowed_topics and topic not in allowed_topics:
                continue
            if start_ns is not None and timestamp < start_ns:
                continue
            if end_ns is not None and timestamp > end_ns:
                continue
            yield SensorMessage(topic=topic, timestamp_ns=timestamp, rawdata=rawdata)


def _default_reader_factory(bag_path: str | Path, storage_id: str):
    from rosbags.rosbag2 import Reader

    reader = Reader(bag_path, storage_id=storage_id)
    reader.open()
    return reader
