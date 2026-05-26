"""RealSense metadata subscriptions for timing and drop monitoring."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RealSenseMetadataEvent:
    """One parsed RealSense metadata message."""

    topic: str
    frame_number: int | None
    header_stamp_ns: int
    frame_timestamp_ns: int | None
    hw_timestamp_ns: int | None
    clock_domain: str | None
    recv_time_ns: int
    recv_monotonic_ns: int


def stamp_to_ns(stamp) -> int:
    """Convert a ROS builtin_interfaces/Time to nanoseconds."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def metadata_int(data: dict, key: str) -> int | None:
    """Extract an integer metadata field."""
    value = data.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(round(float(value)))
        except Exception:
            return None


def metadata_ms_to_ns(data: dict, key: str) -> int | None:
    """Extract a RealSense millisecond timestamp field as nanoseconds."""
    value = data.get(key)
    if value is None:
        return None
    try:
        return int(round(float(value) * 1_000_000.0))
    except Exception:
        return None


def metadata_str(data: dict, key: str) -> str | None:
    """Extract a non-empty string metadata field."""
    value = data.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class RealSenseMetadataMonitor:
    """Small rclpy node runner for RealSense metadata topics."""

    def __init__(self, topics: tuple[str, ...], on_event: Callable[[RealSenseMetadataEvent], None]):
        self.topics = topics
        self.on_event = on_event
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._node = None
        self._rclpy = None

    def start(self) -> None:
        """Start the metadata subscriber thread."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='RealSenseMetadataMonitor', daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the metadata subscriber."""
        self._stop.set()
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def _run(self) -> None:
        try:
            import rclpy
            from realsense2_camera_msgs.msg import Metadata
        except Exception as exc:  # pragma: no cover - depends on ROS environment.
            raise RuntimeError(f'RealSense metadata dependencies are unavailable: {exc}') from exc

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        node = rclpy.create_node('main_controller_realsense_metadata')
        self._node = node
        for topic in self.topics:
            node.create_subscription(Metadata, topic, lambda msg, topic=topic: self._on_metadata(topic, msg), 10)
        while not self._stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)

    def _on_metadata(self, topic: str, msg) -> None:
        try:
            data = json.loads(msg.json_data)
        except Exception:
            data = {}
        event = RealSenseMetadataEvent(
            topic=topic,
            frame_number=metadata_int(data, 'frame_number'),
            header_stamp_ns=stamp_to_ns(msg.header.stamp),
            frame_timestamp_ns=metadata_ms_to_ns(data, 'frame_timestamp'),
            hw_timestamp_ns=metadata_ms_to_ns(data, 'hw_timestamp'),
            clock_domain=metadata_str(data, 'clock_domain'),
            recv_time_ns=time.time_ns(),
            recv_monotonic_ns=time.monotonic_ns(),
        )
        self.on_event(event)
