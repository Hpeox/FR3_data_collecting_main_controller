from __future__ import annotations

import struct

from MainController.drop_monitor import DropMonitor
from MainController.realsense_metadata import metadata_int, metadata_ms_to_ns
from MainController.zmq_telemetry import FRAME_SIZE, MAGIC, VERSION, unpack_frame


def test_zmq_unpack_frame_minimal():
    frame_struct = struct.Struct("<4sBBHQdQ58dBB6x")
    floats = [float(i) for i in range(58)]
    payload = frame_struct.pack(MAGIC, VERSION, 2, 0, 42, 1.25, 2, *floats, 12, 3)
    assert len(payload) == FRAME_SIZE

    frame = unpack_frame(payload)

    assert frame.source == 2
    assert frame.seq == 42
    assert frame.stamp == 1.25
    assert frame.valid_mask == 2
    assert frame.floats_58[:3] == (0.0, 1.0, 2.0)
    assert frame.gripper_gPO == 12
    assert frame.gripper_gCU == 3


def test_drop_monitor_warns_on_gap_and_large_interval():
    monitor = DropMonitor("stream", expected_interval_ns=10, warning_interval_ns=20)
    assert monitor.observe(0, 0) == []

    warnings = monitor.observe(2, 30)

    assert [warning.reason for warning in warnings] == ["non_contiguous_key", "large_interval"]
    assert monitor.summary()["missing_frame_count"] == 1
    assert monitor.summary()["warning_count"] == 2


def test_drop_monitor_reset_baseline():
    monitor = DropMonitor("stream", expected_interval_ns=10, warning_interval_ns=20)
    monitor.observe(10, 100)
    monitor.reset_baseline()

    assert monitor.observe(30, 1_000_000) == []


def test_realsense_metadata_helpers():
    data = {"frame_number": "7", "frame_timestamp": "1.5", "hw_timestamp": 2.25}

    assert metadata_int(data, "frame_number") == 7
    assert metadata_ms_to_ns(data, "frame_timestamp") == 1_500_000
    assert metadata_ms_to_ns(data, "hw_timestamp") == 2_250_000
