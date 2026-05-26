from __future__ import annotations

import struct
import sys
import time
import uuid
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main_controller.drop_monitor import DropMonitor
from main_controller.config import RuntimeConfig
from main_controller.realsense_image_guard import validate_rosbag_image_metadata
from main_controller.realsense_metadata import metadata_int, metadata_ms_to_ns, metadata_str
from main_controller.timestamp_alignment import AlignmentOptions, align_demo_timestamps
from main_controller.zmq_telemetry import FRAME_SIZE, FRAME_STRUCT, MAGIC, VERSION, ZmqTelemetryReceiver, unpack_frame


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
    data = {"frame_number": "7", "frame_timestamp": "1.5", "hw_timestamp": 2.25, "clock_domain": "SYSTEM_TIME"}

    assert metadata_int(data, "frame_number") == 7
    assert metadata_ms_to_ns(data, "frame_timestamp") == 1_500_000
    assert metadata_ms_to_ns(data, "hw_timestamp") == 2_250_000
    assert metadata_str(data, "clock_domain") == "SYSTEM_TIME"
    assert metadata_str({}, "clock_domain") is None


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


def test_timestamp_alignment_xense_base_uses_timestamp_ns_0(tmp_path):
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    t = np.asarray([1_000_000_000, 1_033_000_000, 1_066_000_000], dtype=np.int64)
    np.savez(demo_dir / "ft300_timestamps.npz", frame_id=np.arange(3), timestamp_ns=t - 1_000_000, recv_time_ns=t, recv_monotonic_ns=t)
    np.savez(demo_dir / "xense_timestamps.npz", frame_id=np.arange(3), timestamp_ns_0=t, timestamp_ns_1=t + 6_000_000, recv_time_ns=t, recv_monotonic_ns=t)
    np.savez(
        demo_dir / "realsense_metadata.npz",
        topic=np.asarray(["/cam1/camera/color/metadata"] * 3),
        frame_number=np.arange(3),
        header_stamp_ns=t,
        frame_timestamp_ns=t,
        hw_timestamp_ns=t,
        clock_domain=np.asarray(["SYSTEM_TIME"] * 3),
        recv_time_ns=t,
        recv_monotonic_ns=t,
    )
    np.savez(
        demo_dir / "zmq_telemetry.npz",
        source=np.asarray([2, 2, 2]),
        seq=np.arange(3),
        stamp_s=t.astype(float) / 1_000_000_000.0,
        valid_mask=np.ones(3, dtype=np.uint64),
        floats_58=np.zeros((3, 58)),
        gripper_gPO=np.zeros(3, dtype=np.uint8),
        gripper_gCU=np.zeros(3, dtype=np.uint8),
        recv_time_ns=t,
        recv_monotonic_ns=t,
    )
    manifest = {
        "status": "done",
        "rosbag_uri": str(demo_dir / "rosbag"),
        "sensor_saved_files": {"ft300": None, "xense": None},
        "npz": {
            "ft300": str(demo_dir / "ft300_timestamps.npz"),
            "xense": str(demo_dir / "xense_timestamps.npz"),
            "realsense": str(demo_dir / "realsense_metadata.npz"),
            "zmq": str(demo_dir / "zmq_telemetry.npz"),
        },
        "realsense_image_readiness": {"required_topics": ["/cam1/camera/color/image_raw"]},
        "realsense_rosbag_postcheck": {"required_topics": ["/cam1/camera/color/image_raw"]},
    }
    (demo_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = align_demo_timestamps(
        demo_dir,
        AlignmentOptions(alignment_base_source="xense", start_trim_s=0.0),
    )

    index = np.load(result.index_path, allow_pickle=True)
    np.testing.assert_array_equal(index["t_ns"], np.asarray([t[1]], dtype=np.int64))
    assert result.base == "xense:0"
    assert np.all(index["ft300s_delta_ns"] <= 0)
