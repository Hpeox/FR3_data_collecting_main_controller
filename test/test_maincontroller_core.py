from __future__ import annotations

import struct
import sys
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MainController.drop_monitor import DropMonitor
from MainController.config import RuntimeConfig
from MainController.realsense_image_guard import validate_rosbag_image_metadata
from MainController.realsense_metadata import metadata_int, metadata_ms_to_ns
from MainController.zmq_telemetry import FRAME_SIZE, FRAME_STRUCT, MAGIC, VERSION, ZmqTelemetryReceiver, unpack_frame


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


def test_zmq_receiver_invalid_frame_is_nonfatal_and_receiver_continues():
    zmq = pytest.importorskip("zmq")
    endpoint = f"inproc://receiver-invalid-continues-{uuid.uuid4().hex}"
    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    publisher.bind(endpoint)
    errors: list[str] = []
    fatals: list[str] = []
    frames = []

    def on_frame(frame, _recv_time_ns, _recv_monotonic_ns):
        frames.append(frame)

    receiver = ZmqTelemetryReceiver(endpoint, on_frame, errors.append, fatals.append)
    receiver.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not frames:
            publisher.send(b"bad")
            publisher.send(
                FRAME_STRUCT.pack(
                    MAGIC,
                    VERSION,
                    1,
                    0,
                    7,
                    1.5,
                    1,
                    *(float(index) for index in range(58)),
                    0,
                    0,
                )
            )
            time.sleep(0.05)
        assert frames
        assert frames[0].seq == 7
        assert any("invalid ZMQ frame" in error for error in errors)
        assert fatals == []
    finally:
        receiver.stop()
        publisher.close(linger=0)


def test_zmq_receiver_on_frame_exception_reports_fatal():
    zmq = pytest.importorskip("zmq")
    endpoint = f"inproc://receiver-fatal-{uuid.uuid4().hex}"
    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    publisher.bind(endpoint)
    fatals: list[str] = []

    def on_frame(_frame, _recv_time_ns, _recv_monotonic_ns):
        raise RuntimeError("callback exploded")

    receiver = ZmqTelemetryReceiver(endpoint, on_frame, lambda _message: None, fatals.append)
    receiver.start()
    try:
        payload = FRAME_STRUCT.pack(
            MAGIC,
            VERSION,
            1,
            0,
            8,
            1.5,
            1,
            *(float(index) for index in range(58)),
            0,
            0,
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not fatals:
            publisher.send(payload)
            time.sleep(0.05)
        assert any("ZMQ receiver failed" in fatal for fatal in fatals)
    finally:
        receiver.stop()
        publisher.close(linger=0)


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


def test_realsense_formal_image_requirements_are_four_cameras_eight_topics():
    config = RuntimeConfig()
    requirements = config.realsense_image_requirements

    assert config.realsense_capture_mode == "formal"
    assert len(requirements) == 8
    assert [requirement.topic for requirement in requirements] == [
        "/cam1/camera/color/image_raw",
        "/cam1/camera/aligned_depth_to_color/image_raw",
        "/cam2/camera/color/image_raw",
        "/cam2/camera/aligned_depth_to_color/image_raw",
        "/cam3/camera/color/image_raw",
        "/cam3/camera/aligned_depth_to_color/image_raw",
        "/cam4/camera/color/image_raw",
        "/cam4/camera/aligned_depth_to_color/image_raw",
    ]
    assert {requirement.message_type for requirement in requirements} == {"sensor_msgs/msg/Image"}


def test_realsense_debug_degraded_image_requirements_use_configured_subset():
    config = RuntimeConfig(
        realsense_capture_mode="debug_degraded",
        realsense_debug_image_topics=(
            "/cam3/camera/color/image_raw",
            "/cam3/camera/aligned_depth_to_color/image_raw",
        ),
    )

    assert [requirement.topic for requirement in config.realsense_image_requirements] == [
        "/cam3/camera/color/image_raw",
        "/cam3/camera/aligned_depth_to_color/image_raw",
    ]


def test_realsense_rosbag_postcheck_detects_missing_and_skew(tmp_path):
    config = RuntimeConfig()
    requirements = config.realsense_image_requirements
    metadata = {
        requirement.topic: {"message_type": requirement.message_type, "count": 10}
        for requirement in requirements[:-1]
    }
    metadata[requirements[0].topic]["count"] = 100

    result = validate_rosbag_image_metadata(
        mode=config.realsense_capture_mode,
        rosbag_uri=tmp_path / "demo" / "rosbag",
        requirements=requirements,
        topic_metadata=metadata,
        count_skew_limit=config.realsense_rosbag_count_skew_limit,
    )

    assert not result.ok
    assert result.missing_topics == (requirements[-1].topic,)
    assert result.count_skew == 90
