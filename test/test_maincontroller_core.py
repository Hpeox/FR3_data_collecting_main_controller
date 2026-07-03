from __future__ import annotations

import struct
import sys
import time
import uuid
import json
import argparse
import io
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main_controller.drop_monitor import DropMonitor
import main_controller.config as config_module
from main_controller.config import (
    RuntimeConfig as RuntimeConfigClass,
    load_task_instruction_config,
    parse_task_instruction_payload,
    validate_repo_root,
    validate_task_name,
)
import main_controller.main as main_module
from main_controller.main import MainController, build_config, parse_args
from main_controller.gripper_plot import load_gripper_series, load_tactile_preview, render_gripper_plot
from main_controller.processes import ManagedProcess
from main_controller.realsense_image_guard import (
    check_ros_image_topic_readiness,
    read_rosbag_topic_metadata,
    validate_rosbag_image_metadata,
)
from main_controller.realsense_metadata import RealSenseMetadataMonitor, metadata_int, metadata_ms_to_ns, metadata_str
import main_controller.timestamp_alignment as timestamp_alignment_module
from main_controller.timestamp_alignment import AlignmentOptions, align_demo_timestamps
from main_controller.uds_client import UdsClient
from main_controller.zmq_telemetry import (
    FRAME_SIZE,
    FRAME_STRUCT,
    MAGIC,
    VERSION,
    ZmqTelemetryReceiver,
    unpack_frame,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
TASK_NAME = "16mm-peg-in-hole"
LANGUAGE_INSTRUCTION = "Pick up the test object"
TASK_INSTRUCTIONS = (LANGUAGE_INSTRUCTION,)
TASK_WEIGHTS = (1.0,)


def RuntimeConfig(**kwargs):
    kwargs.setdefault("task_name", TASK_NAME)
    kwargs.setdefault("task_instructions", TASK_INSTRUCTIONS)
    kwargs.setdefault("task_instruction_weights", TASK_WEIGHTS)
    return RuntimeConfigClass(**kwargs)


def test_managed_process_reports_one_fatal_per_matching_line(tmp_path):
    calls: list[tuple[str, str]] = []

    class FakeStdout:
        def __iter__(self):
            return iter(['Depth stream start failure, Hardware Error\n'])

    class FakeProcess:
        stdout = FakeStdout()

        def wait(self) -> int:
            return 0

    process = ManagedProcess(
        'realsense_camera',
        ['true'],
        tmp_path,
        tmp_path / 'process.log',
        fatal_patterns=('Hardware Error', 'Depth stream start failure'),
        on_fatal=lambda name, line: calls.append((name, line)),
    )
    process.process = FakeProcess()

    process._read_output(io.StringIO())

    assert calls == [('realsense_camera', 'Depth stream start failure, Hardware Error')]


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


def test_gripper_plot_uses_source_local_indices_and_command_mapping(tmp_path):
    npz_path = tmp_path / 'zmq_telemetry.npz'
    floats = np.full((6, 58), np.nan, dtype=np.float64)
    floats[0, 7] = 0.0
    floats[2, 7] = 0.25
    floats[5, 7] = 1.0
    np.savez(
        npz_path,
        source=np.array([1, 3, 1, 2, 3, 1], dtype=np.uint8),
        floats_58=floats,
        gripper_gPO=np.array([0, 10, 0, 0, 240, 0], dtype=np.uint8),
        gripper_gCU=np.array([0, 1, 0, 0, 3, 0], dtype=np.uint8),
    )

    series = load_gripper_series(npz_path)

    np.testing.assert_array_equal(series.command_index, [0, 1, 2])
    np.testing.assert_allclose(series.mapped_command, [255.0, 191.25, 0.0])
    np.testing.assert_array_equal(series.feedback_index, [0, 1])
    np.testing.assert_array_equal(series.gpo, [10, 240])
    np.testing.assert_array_equal(series.gcu, [1, 3])


def test_gripper_plot_renders_png_and_atomically_overwrites(tmp_path):
    npz_path = tmp_path / 'zmq_telemetry.npz'
    output_path = tmp_path / 'gripper.png'
    floats = np.full((4, 58), np.nan, dtype=np.float64)
    floats[0, 7] = 0.2
    floats[2, 7] = 0.8
    np.savez(
        npz_path,
        source=np.array([1, 3, 1, 3], dtype=np.uint8),
        floats_58=floats,
        gripper_gPO=np.array([0, 20, 0, 220], dtype=np.uint8),
        gripper_gCU=np.array([0, 1, 0, 3], dtype=np.uint8),
    )
    output_path.write_bytes(b'old-preview')

    result = render_gripper_plot(npz_path, output_path)

    assert output_path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert result['output_path'] == str(output_path)
    assert result['command_samples'] == 2
    assert result['feedback_samples'] == 2
    assert result['tactile_preview_ok'] is False
    assert not list(tmp_path.glob('.gripper.*.tmp.png'))


def test_gripper_plot_renders_tactile_preview_and_deletes_preview(tmp_path):
    npz_path = tmp_path / 'zmq_telemetry.npz'
    output_path = tmp_path / 'gripper.png'
    preview_path = tmp_path / 'tactile_preview.npz'
    floats = np.full((4, 58), np.nan, dtype=np.float64)
    floats[0, 7] = 0.2
    floats[2, 7] = 0.8
    np.savez(
        npz_path,
        source=np.array([1, 3, 1, 3], dtype=np.uint8),
        floats_58=floats,
        gripper_gPO=np.array([0, 20, 0, 220], dtype=np.uint8),
        gripper_gCU=np.array([0, 1, 0, 3], dtype=np.uint8),
    )
    force_resultant = np.zeros((2, 3, 6), dtype=np.float64)
    force_resultant[0, :, 0] = [0.1, 0.2, 0.3]
    force_resultant[1, :, 5] = [1.0, 1.1, 1.2]
    np.savez(
        preview_path,
        sensor_ids=np.array(['OG000544', 'OG001009']),
        frame_index=np.arange(3, dtype=np.int64),
        force_resultant=force_resultant,
        edge_warning=np.array([False, True]),
        edge_max=np.array([0.3, 1.2]),
    )

    loaded = load_tactile_preview(preview_path)
    assert loaded.sensor_ids == ('OG000544', 'OG001009')

    result = render_gripper_plot(npz_path, output_path, preview_path)

    assert output_path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert result['tactile_preview_ok'] is True
    assert result['tactile_samples'] == 3
    assert result['tactile_sensor_ids'] == ['OG000544', 'OG001009']
    assert not preview_path.exists()


def test_gripper_plot_bad_tactile_preview_falls_back_and_deletes_preview(tmp_path):
    npz_path = tmp_path / 'zmq_telemetry.npz'
    output_path = tmp_path / 'gripper.png'
    preview_path = tmp_path / 'bad_preview.npz'
    floats = np.full((4, 58), np.nan, dtype=np.float64)
    floats[0, 7] = 0.2
    floats[2, 7] = 0.8
    np.savez(
        npz_path,
        source=np.array([1, 3, 1, 3], dtype=np.uint8),
        floats_58=floats,
        gripper_gPO=np.array([0, 20, 0, 220], dtype=np.uint8),
        gripper_gCU=np.array([0, 1, 0, 3], dtype=np.uint8),
    )
    preview_path.write_bytes(b'not an npz')

    result = render_gripper_plot(npz_path, output_path, preview_path)

    assert output_path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert result['tactile_preview_ok'] is False
    assert 'tactile_preview_error' in result
    assert not preview_path.exists()


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

    receiver = ZmqTelemetryReceiver(endpoint, on_frame, errors.append, fatals.append, context=context)
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

    receiver = ZmqTelemetryReceiver(endpoint, on_frame, lambda _message: None, fatals.append, context=context)
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


def test_uds_client_disconnect_callback_wakes_pending_ack():
    disconnects: list[tuple[str, list[str]]] = []
    client = UdsClient(
        "ft300",
        "/tmp/nonexistent-ft300.sock",
        lambda _event: None,
        on_disconnect=lambda name, pending_cmds: disconnects.append((name, pending_cmds)),
    )
    ack_event = client._ack_event("STOP_REQ")
    client._connected.set()
    with client._ack_lock:
        client._pending_ack_cmds.add("STOP_REQ")

    client._mark_disconnected()

    assert disconnects == [("ft300", ["STOP_REQ"])]
    assert ack_event.is_set()
    assert client.last_error_for("STOP_REQ") == {"error": "uds_disconnected", "cmd": "STOP_REQ"}


def test_sensor_path_from_payload_allows_missing_saved_file(tmp_path):
    controller = MainController(
        RuntimeConfig(
            repo_root=REPO_ROOT,
            runtime_root=tmp_path,
        )
    )

    assert controller.runtime_sessions_dir == tmp_path / "runtime_sessions"
    assert controller.runtime_sessions_dir.is_dir()
    assert controller.config.runtime_frames_dir == tmp_path / "runtime_frames"
    assert controller.config.runtime_frames_dir.is_dir()
    assert controller._sensor_path_from_payload({}) is None
    assert controller._sensor_path_from_payload({"saved_file": None}) is None
    assert controller._sensor_path_from_payload({"saved_file": "data_FT_demo.npy"}) == (
        "runtime_frames/data_FT_demo.npy"
    )
    assert controller._sensor_path_from_payload({"saved_file": ""}) is None
    assert controller._sensor_path_from_payload({"saved_file": "/tmp/data_FT_demo.npy"}) is None
    assert controller._sensor_path_from_payload({"saved_file": "nested/data_FT_demo.npy"}) is None
    assert controller._sensor_path_from_payload({"saved_file": r"nested\data_FT_demo.npy"}) is None


def test_realsense_metadata_helpers():
    data = {
        "frame_number": "7",
        "frame_timestamp": "1.5",
        "hw_timestamp": 2.25,
        "clock_domain": "SYSTEM_TIME",
    }

    assert metadata_int(data, "frame_number") == 7
    assert metadata_ms_to_ns(data, "frame_timestamp") == 1_500_000
    assert metadata_ms_to_ns(data, "hw_timestamp") == 2_250_000
    assert metadata_str(data, "clock_domain") == "SYSTEM_TIME"
    assert metadata_str({}, "clock_domain") is None


def test_realsense_metadata_monitor_reports_dependency_failure(monkeypatch):
    failures: list[str] = []
    monkeypatch.setitem(sys.modules, "rclpy", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "realsense2_camera_msgs", types.ModuleType("realsense2_camera_msgs"))
    monkeypatch.setitem(sys.modules, "realsense2_camera_msgs.msg", None)
    monitor = RealSenseMetadataMonitor(("/cam1/camera/color/metadata",), lambda _event: None, failures.append)

    monitor.start()

    assert monitor.wait_ready(1.0) is False
    assert monitor.fatal_error() is not None
    assert "RealSense metadata monitor failed" in monitor.fatal_error()
    assert failures == [monitor.fatal_error()]
    monitor.stop()


def test_realsense_metadata_monitor_ready_after_subscriptions(monkeypatch):
    subscriptions: list[str] = []

    class FakeNode:
        def create_subscription(self, _message_type, topic, _callback, _qos):
            subscriptions.append(topic)

        def destroy_node(self):
            pass

    fake_rclpy = types.SimpleNamespace(
        ok=lambda: True,
        init=lambda args=None: None,
        create_node=lambda _name: FakeNode(),
        spin_once=lambda _node, timeout_sec=0.1: time.sleep(0.01),
    )
    message_module = types.ModuleType("realsense2_camera_msgs.msg")
    message_module.Metadata = object
    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(sys.modules, "realsense2_camera_msgs", types.ModuleType("realsense2_camera_msgs"))
    monkeypatch.setitem(sys.modules, "realsense2_camera_msgs.msg", message_module)
    monitor = RealSenseMetadataMonitor(("/cam1/camera/color/metadata",), lambda _event: None)

    monitor.start()

    assert monitor.wait_ready(1.0) is True
    assert subscriptions == ["/cam1/camera/color/metadata"]
    assert monitor.fatal_error() is None
    monitor.stop()


def test_realsense_metadata_monitor_reports_spin_failure(monkeypatch):
    failures: list[str] = []

    class FakeNode:
        def create_subscription(self, _message_type, _topic, _callback, _qos):
            pass

        def destroy_node(self):
            pass

    def spin_once(_node, timeout_sec=0.1):
        raise RuntimeError("injected spin failure")

    fake_rclpy = types.SimpleNamespace(
        ok=lambda: True,
        init=lambda args=None: None,
        create_node=lambda _name: FakeNode(),
        spin_once=spin_once,
    )
    message_module = types.ModuleType("realsense2_camera_msgs.msg")
    message_module.Metadata = object
    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(sys.modules, "realsense2_camera_msgs", types.ModuleType("realsense2_camera_msgs"))
    monkeypatch.setitem(sys.modules, "realsense2_camera_msgs.msg", message_module)
    monitor = RealSenseMetadataMonitor(("/cam1/camera/color/metadata",), lambda _event: None, failures.append)

    monitor.start()

    assert monitor.wait_ready(1.0) is True
    deadline = time.monotonic() + 1.0
    while not monitor.fatal_event.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert monitor.fatal_error() is not None
    assert "injected spin failure" in monitor.fatal_error()
    assert failures == [monitor.fatal_error()]
    monitor.stop()


def test_repo_root_validation_finds_integrated_modules():
    root = validate_repo_root(REPO_ROOT)

    assert (root / "FT300S").is_dir()
    assert (root / "XenseTacSensor").is_dir()
    assert (root / "RealSense" / "launch").is_dir()


def test_repo_root_validation_rejects_missing_modules(tmp_path):
    with pytest.raises(RuntimeError, match="FT300S"):
        validate_repo_root(tmp_path)


def _config_args(**overrides):
    values = {
        "repo_root": None,
        "runtime_root": None,
        "task_name": TASK_NAME,
        "zmq_connect": "tcp://127.0.0.1:6000",
        "xense_sdk_version": "2.0",
        "startup_timeout_s": 60.0,
        "ack_timeout_s": 2.0,
        "sensor_flush_timeout_s": 300.0,
        "progress_log_period_s": 5.0,
        "alignment_base": "realsense:bundle",
        "alignment_mode": "causal",
        "alignment_hz": 30.0,
        "alignment_start_trim_s": 2.0,
        "alignment_end_trim_s": 0.0,
        "gripper_plot_timeout_s": 30.0,
        "realsense_image_ready_timeout_s": 30.0,
        "realsense_rosbag_count_skew_limit_percent": 0.5,
        "realsense_capture_mode": "formal",
        "realsense_debug_image_topic": [],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_config_uses_explicit_repo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        main_module,
        "load_task_instruction_config",
        lambda repo_root, task_name: (TASK_INSTRUCTIONS, TASK_WEIGHTS),
    )
    config = build_config(
        _config_args(
            repo_root=str(REPO_ROOT),
            runtime_root=str(tmp_path / "runtime"),
            xense_sdk_version="1.x",
            alignment_base="xense:pair",
            alignment_end_trim_s=0.25,
            gripper_plot_timeout_s=12.5,
            realsense_image_ready_timeout_s=12.0,
            realsense_rosbag_count_skew_limit_percent=0.75,
            realsense_capture_mode="debug_degraded",
            realsense_debug_image_topic=("/cam1/camera/color/image_raw",),
        )
    )

    assert config.repo_root == REPO_ROOT
    assert config.runtime_root == (tmp_path / "runtime").resolve()
    assert config.runtime_sessions_dir == (tmp_path / "runtime" / "runtime_sessions").resolve()
    assert config.runtime_frames_dir == (tmp_path / "runtime" / "runtime_frames").resolve()
    assert config.task_name == TASK_NAME
    assert config.task_instructions == TASK_INSTRUCTIONS
    assert config.task_instruction_weights == TASK_WEIGHTS
    assert config.xense_sdk_version == "1.x"
    assert config.alignment_base == "xense:pair"
    assert config.alignment_end_trim_s == 0.25
    assert config.gripper_plot_timeout_s == 12.5
    assert config.realsense_image_ready_timeout_s == 12.0
    assert config.realsense_rosbag_count_skew_limit_percent == 0.75
    assert config.realsense_capture_mode == "debug_degraded"
    assert config.realsense_debug_image_topics == ("/cam1/camera/color/image_raw",)


def test_build_config_uses_build_time_hint(monkeypatch):
    monkeypatch.setattr(config_module, "build_time_repo_root_hint", lambda: REPO_ROOT)
    monkeypatch.setattr(
        main_module,
        "load_task_instruction_config",
        lambda repo_root, task_name: (TASK_INSTRUCTIONS, TASK_WEIGHTS),
    )

    config = build_config(_config_args())

    assert config.repo_root == REPO_ROOT
    assert config.runtime_root == REPO_ROOT
    assert config.runtime_sessions_dir == REPO_ROOT / "runtime_sessions"
    assert config.runtime_frames_dir == REPO_ROOT / "runtime_frames"
    assert config.xense_sdk_version == "2.0"


def test_sensor_apps_accept_save_dir(monkeypatch, tmp_path):
    from FT300S.app import parse_args as parse_ft300_args
    from XenseTacSensor.app import parse_args as parse_xense_args

    save_dir = tmp_path / "runtime_frames"
    monkeypatch.setattr(sys, "argv", ["ft300", "--save-dir", str(save_dir)])
    assert parse_ft300_args().save_dir == save_dir

    monkeypatch.setattr(sys, "argv", ["xense", "--save-dir", str(save_dir)])
    assert parse_xense_args().save_dir == save_dir


def test_parse_args_rejects_removed_output_dir(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main_controller",
            "--task-name",
            TASK_NAME,
            "--output-dir",
            "/tmp/sessions",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_requires_task_name(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main_controller"])

    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_accepts_task_name(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main_controller", "--task-name", TASK_NAME],
    )

    assert parse_args().task_name == TASK_NAME


@pytest.mark.parametrize(
    "task_name",
    ["", ".hidden", "task/name", "task..name", "任务", "task name"],
)
def test_parse_args_rejects_invalid_task_name(monkeypatch, task_name):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main_controller", "--task-name", task_name],
    )

    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {
                "instructions": [
                    {"text": "first", "weight": 0.6},
                    {"text": "second", "weight": -1},
                ]
            },
            (0.6, 0.4),
        ),
        (
            {
                "instructions": [
                    {"text": "first", "weight": 0.2},
                    {"text": "second", "weight": -1},
                    {"text": "third", "weight": -1},
                ]
            },
            (0.2, 0.4, 0.4),
        ),
        (
            {
                "instructions": [
                    {"text": "first", "weight": -1},
                    {"text": "second", "weight": -1},
                ]
            },
            (0.5, 0.5),
        ),
        (
            {"instructions": [{"text": "only", "weight": -1}]},
            (1.0,),
        ),
    ],
)
def test_parse_task_instruction_payload_resolves_weights(payload, expected):
    texts, weights = parse_task_instruction_payload(payload)

    assert texts == tuple(item["text"] for item in payload["instructions"])
    assert weights == pytest.approx(expected)


def test_parse_task_instruction_payload_accepts_explicit_sum():
    texts, weights = parse_task_instruction_payload(
        {
            "instructions": [
                {"text": "first", "weight": 0.25},
                {"text": "second", "weight": 0.75},
            ]
        }
    )

    assert texts == ("first", "second")
    assert weights == pytest.approx((0.25, 0.75))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"instructions": []},
        {"instructions": [{"text": "first", "weight": 1.0}]},
        {"instructions": [{"text": "", "weight": -1}]},
        {"instructions": [{"text": "first", "weight": True}]},
        {"instructions": [{"text": "first", "weight": float("nan")}]},
        {"instructions": [{"text": "first", "weight": 0.6, "extra": 1}]},
        {
            "instructions": [
                {"text": "first", "weight": 0.7},
                {"text": "second", "weight": 0.4},
                {"text": "third", "weight": -1},
            ]
        },
    ],
)
def test_parse_task_instruction_payload_rejects_invalid_payload(payload):
    with pytest.raises(RuntimeError):
        parse_task_instruction_payload(payload)


def test_load_task_instruction_config_uses_repo_root(tmp_path):
    repo_root = tmp_path / "repo"
    path = repo_root / "TaskInstruction" / f"{TASK_NAME}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "instructions": [
                    {"text": LANGUAGE_INSTRUCTION, "weight": -1}
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_task_instruction_config(repo_root, TASK_NAME) == (
        TASK_INSTRUCTIONS,
        TASK_WEIGHTS,
    )


def test_load_task_instruction_config_rejects_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        load_task_instruction_config(tmp_path, TASK_NAME)


@pytest.mark.parametrize(
    ("content", "error"),
    [
        (b"\xff", "valid UTF-8"),
        (b"{", "valid JSON"),
        (json.dumps({"instructions": []}).encode("utf-8"), "non-empty array"),
    ],
)
def test_load_task_instruction_config_rejects_invalid_file(
    tmp_path,
    content,
    error,
):
    path = tmp_path / "TaskInstruction" / f"{TASK_NAME}.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)

    with pytest.raises(RuntimeError, match=error):
        load_task_instruction_config(tmp_path, TASK_NAME)


def test_runtime_config_rejects_unknown_xense_sdk_version():
    with pytest.raises(ValueError, match="unsupported xense_sdk_version"):
        RuntimeConfig(
            repo_root=REPO_ROOT,
            xense_sdk_version="3.0",
        )


@pytest.mark.parametrize("task_name", [".hidden", "task..name", "task/name"])
def test_runtime_config_rejects_invalid_task_name(task_name):
    with pytest.raises((TypeError, ValueError), match="task_name"):
        RuntimeConfig(repo_root=REPO_ROOT, task_name=task_name)


def test_realsense_formal_image_requirements_are_four_cameras_eight_topics():
    config = RuntimeConfig(
        repo_root=REPO_ROOT,
    )
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
        repo_root=REPO_ROOT,
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


def test_realsense_image_readiness_uses_shallow_qos_and_destroys_ready_subscriptions(monkeypatch):
    class FakeImage:
        width = 640
        height = 480
        encoding = "rgb8"
        step = 1920

    message_module = types.ModuleType("sensor_msgs.msg")
    message_module.Image = FakeImage
    monkeypatch.setitem(sys.modules, "sensor_msgs", types.ModuleType("sensor_msgs"))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", message_module)

    class FakeHistoryPolicy:
        KEEP_LAST = "keep_last"

    class FakeReliabilityPolicy:
        BEST_EFFORT = "best_effort"

    class FakeQoSProfile:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    qos_module = types.ModuleType("rclpy.qos")
    qos_module.HistoryPolicy = FakeHistoryPolicy
    qos_module.ReliabilityPolicy = FakeReliabilityPolicy
    qos_module.QoSProfile = FakeQoSProfile
    monkeypatch.setitem(sys.modules, "rclpy", types.ModuleType("rclpy"))
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos_module)

    callbacks = {}
    qos_values = []
    destroyed = []

    class FakeNode:
        def create_subscription(self, _message_type, topic, callback, qos):
            callbacks[topic] = callback
            qos_values.append(qos)
            return topic

        def destroy_subscription(self, subscription):
            destroyed.append(subscription)

    class FakeExecutor:
        def __init__(self):
            self.calls = 0

        def spin_once(self, timeout_sec=0.05):
            del timeout_sec
            self.calls += 1
            if self.calls == 1:
                callbacks["/cam1/camera/color/image_raw"](FakeImage())
            elif self.calls == 2:
                callbacks["/cam2/camera/color/image_raw"](FakeImage())

    requirements = RuntimeConfig(
        repo_root=REPO_ROOT,
        cameras=("cam1", "cam2"),
    ).realsense_image_requirements[::2]
    result = check_ros_image_topic_readiness(FakeNode(), FakeExecutor(), requirements, timeout_s=1.0, mode="formal")

    assert result.ok
    assert all(qos.kwargs["depth"] == 1 for qos in qos_values)
    assert all(qos.kwargs["reliability"] == FakeReliabilityPolicy.BEST_EFFORT for qos in qos_values)
    assert destroyed == ["/cam1/camera/color/image_raw", "/cam2/camera/color/image_raw"]


def test_realsense_rosbag_postcheck_detects_missing_and_skew(tmp_path):
    config = RuntimeConfig(
        repo_root=REPO_ROOT,
    )
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
        count_skew_limit_percent=config.realsense_rosbag_count_skew_limit_percent,
    )

    assert not result.ok
    assert result.missing_topics == (requirements[-1].topic,)
    assert result.count_skew == 90
    assert result.count_skew_reference_count == 10
    assert result.count_skew_limit == 0.05


def test_read_rosbag_topic_metadata_uses_detected_storage_id(tmp_path, monkeypatch):
    bag_dir = tmp_path / "rosbag"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n  storage_identifier: mcap\n",
        encoding="utf-8",
    )
    calls: list[tuple[str, str]] = []

    class FakeTopicMetadata:
        name = "/cam1/camera/color/image_raw"
        type = "sensor_msgs/msg/Image"

    class FakeTopicWithCount:
        topic_metadata = FakeTopicMetadata()
        message_count = 42

    class FakeMetadata:
        topics_with_message_count = [FakeTopicWithCount()]

    class FakeInfo:
        def read_metadata(self, uri: str, storage_id: str):
            calls.append((uri, storage_id))
            return FakeMetadata()

    fake_rosbag2_py = types.SimpleNamespace(Info=lambda: FakeInfo())
    monkeypatch.setitem(sys.modules, "rosbag2_py", fake_rosbag2_py)

    result = read_rosbag_topic_metadata(bag_dir)

    assert calls == [(str(bag_dir), "mcap")]
    assert result == {
        "/cam1/camera/color/image_raw": {
            "message_type": "sensor_msgs/msg/Image",
            "count": 42,
        }
    }


def _write_timestamp_alignment_demo(
    demo_dir: Path,
    zmq_offset_ns: int = 0,
    required_realsense_topics: list[str] | None = None,
) -> np.ndarray:
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
        stamp_s=(t - zmq_offset_ns).astype(float) / 1_000_000_000.0,
        valid_mask=np.ones(3, dtype=np.uint64),
        floats_58=np.zeros((3, 58)),
        gripper_gPO=np.zeros(3, dtype=np.uint8),
        gripper_gCU=np.zeros(3, dtype=np.uint8),
        recv_time_ns=t,
        recv_monotonic_ns=t,
    )
    required_realsense_topics = required_realsense_topics or []
    manifest = {
        "status": "done",
        "rosbag_uri": "rosbag",
        "sensor_paths": {"ft300": None, "xense": None},
        "npz": {
            "ft300": "ft300_timestamps.npz",
            "xense": "xense_timestamps.npz",
            "realsense": "realsense_metadata.npz",
            "zmq": "zmq_telemetry.npz",
        },
        "realsense_image_readiness": {"required_topics": required_realsense_topics},
        "realsense_rosbag_postcheck": {"required_topics": []},
    }
    (demo_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return t


_BUNDLE_COLOR_TOPIC = "/cam1/camera/color/image_raw"
_BUNDLE_DEPTH_TOPIC = "/cam1/camera/aligned_depth_to_color/image_raw"


def _patch_realsense_bundle_streams(monkeypatch, streams: dict[str, np.ndarray]) -> None:
    def fake_read_rosbag_image_streams(rosbag_uri: Path | None, topics: list[str], _warnings: list[str]):
        if rosbag_uri is None or not topics:
            return {}
        return {
            topic: timestamp_alignment_module.RosbagImageStream(
                np.asarray(streams[topic], dtype=np.int64),
                (np.asarray(streams[topic], dtype=np.int64) + 1_000).astype(np.int64, copy=False),
            )
            for topic in topics
        }

    monkeypatch.setattr(timestamp_alignment_module, "read_rosbag_image_streams", fake_read_rosbag_image_streams)


def test_realsense_bundle_exact_required_topic_stamps_remain_valid(tmp_path, monkeypatch):
    demo_dir = tmp_path / "demo"
    t = _write_timestamp_alignment_demo(
        demo_dir,
        required_realsense_topics=[_BUNDLE_COLOR_TOPIC, _BUNDLE_DEPTH_TOPIC],
    )
    _patch_realsense_bundle_streams(
        monkeypatch,
        {
            _BUNDLE_COLOR_TOPIC: t,
            _BUNDLE_DEPTH_TOPIC: t,
        },
    )

    result = align_demo_timestamps(
        demo_dir,
        AlignmentOptions(repo_root=REPO_ROOT, base="realsense:bundle", start_trim_s=0.0),
    )

    index = np.load(result.index_path, allow_pickle=True)
    depth_name = timestamp_alignment_module.realsense_stream_name(_BUNDLE_DEPTH_TOPIC)
    assert np.all(index["realsense_bundle_valid"])
    assert np.all(index[f"{depth_name}_valid"])
    assert np.all(index["sample_valid"])
    aligned_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert aligned_manifest["realsense_details"]["quality_counts"] == {"ok": len(index["t_ns"])}


def test_realsense_bundle_uses_causal_child_projection_and_invalidates_mismatch(tmp_path, monkeypatch):
    demo_dir = tmp_path / "demo"
    t = _write_timestamp_alignment_demo(
        demo_dir,
        required_realsense_topics=[_BUNDLE_COLOR_TOPIC, _BUNDLE_DEPTH_TOPIC],
    )
    depth_times = t + 5_000_000
    _patch_realsense_bundle_streams(
        monkeypatch,
        {
            _BUNDLE_COLOR_TOPIC: t,
            _BUNDLE_DEPTH_TOPIC: depth_times,
        },
    )

    result = align_demo_timestamps(
        demo_dir,
        AlignmentOptions(repo_root=REPO_ROOT, base="realsense:bundle", start_trim_s=0.0),
    )

    index = np.load(result.index_path, allow_pickle=True)
    depth_name = timestamp_alignment_module.realsense_stream_name(_BUNDLE_DEPTH_TOPIC)
    assert not np.any(index["realsense_bundle_valid"])
    assert not np.any(index[f"{depth_name}_valid"])
    assert not np.any(index["sample_valid"])
    assert index[f"{depth_name}_time_ns"][0] == depth_times[0]
    assert index[f"{depth_name}_time_ns"][0] <= index["realsense_bundle_time_ns"][0]
    aligned_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert aligned_manifest["realsense_details"]["quality_counts"] == {
        "invalid_timestamp_mismatch": len(index["t_ns"])
    }
    assert aligned_manifest["realsense_details"]["invalid_timestamp_mismatch_count"] == len(index["t_ns"])
    assert any("header stamp mismatch" in warning for warning in aligned_manifest["warnings"])


def test_realsense_bundle_reuse_count_requires_same_topic_source_index_reuse(tmp_path, monkeypatch):
    demo_dir = tmp_path / "demo"
    t = _write_timestamp_alignment_demo(
        demo_dir,
        required_realsense_topics=[_BUNDLE_COLOR_TOPIC, _BUNDLE_DEPTH_TOPIC],
    )
    color_times = t - 10_000_000
    _patch_realsense_bundle_streams(
        monkeypatch,
        {
            _BUNDLE_COLOR_TOPIC: color_times,
            _BUNDLE_DEPTH_TOPIC: np.asarray([color_times[0], color_times[-1] + 33_000_000], dtype=np.int64),
        },
    )

    result = align_demo_timestamps(
        demo_dir,
        AlignmentOptions(repo_root=REPO_ROOT, base="realsense:bundle", start_trim_s=0.0),
    )

    index = np.load(result.index_path, allow_pickle=True)
    depth_name = timestamp_alignment_module.realsense_stream_name(_BUNDLE_DEPTH_TOPIC)
    np.testing.assert_array_equal(index[f"{depth_name}_index"], np.zeros(len(index["t_ns"]), dtype=np.int64))
    aligned_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert aligned_manifest["realsense_details"]["reused_count"] == max(len(index["t_ns"]) - 1, 0)
    assert aligned_manifest["realsense_details"]["invalid_timestamp_mismatch_count"] == len(index["t_ns"])


@pytest.mark.parametrize("base", ["xense:pair", "grid"])
def test_external_base_realsense_bundle_projection_inherits_invalid_rows(tmp_path, monkeypatch, base):
    demo_dir = tmp_path / "demo"
    t = _write_timestamp_alignment_demo(
        demo_dir,
        required_realsense_topics=[_BUNDLE_COLOR_TOPIC, _BUNDLE_DEPTH_TOPIC],
    )
    _patch_realsense_bundle_streams(
        monkeypatch,
        {
            _BUNDLE_COLOR_TOPIC: t,
            _BUNDLE_DEPTH_TOPIC: t + 5_000_000,
        },
    )

    result = align_demo_timestamps(
        demo_dir,
        AlignmentOptions(repo_root=REPO_ROOT, base=base, start_trim_s=0.0),
    )

    index = np.load(result.index_path, allow_pickle=True)
    assert "realsense_bundle_valid" in index.files
    assert not np.any(index["realsense_bundle_valid"])
    assert not np.any(index["sample_valid"])
    aligned_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert aligned_manifest["realsense_alignment_kind"] == "bundle"
    assert aligned_manifest["realsense_details"]["quality_counts"] == {
        "invalid_timestamp_mismatch": aligned_manifest["realsense_details"]["bundle_count"]
    }


def test_timestamp_alignment_xense_pair_base_uses_same_row_max(tmp_path):
    demo_dir = tmp_path / "demo"
    t = _write_timestamp_alignment_demo(demo_dir)
    result = align_demo_timestamps(
        demo_dir,
        AlignmentOptions(repo_root=REPO_ROOT, base="xense:pair", start_trim_s=0.0),
    )

    index = np.load(result.index_path, allow_pickle=True)
    expected = np.asarray([t[0] + 6_000_000, t[1] + 6_000_000], dtype=np.int64)
    np.testing.assert_array_equal(index["t_ns"], expected)
    np.testing.assert_array_equal(index["t_ns"], index["xense_pair_time_ns"])
    np.testing.assert_array_equal(index["xense_0_index"], index["xense_pair_source_index"])
    np.testing.assert_array_equal(index["xense_1_index"], index["xense_pair_source_index"])
    assert result.base == "xense:pair"
    assert np.all(index["ft300s_delta_ns"] <= 0)


def test_timestamp_alignment_records_zmq_clock_offset_without_warning(tmp_path):
    demo_dir = tmp_path / "demo"
    offset_ns = 50_000_000
    _write_timestamp_alignment_demo(demo_dir, zmq_offset_ns=offset_ns)

    result = align_demo_timestamps(
        demo_dir,
        AlignmentOptions(repo_root=REPO_ROOT, base="xense:pair", start_trim_s=0.0),
    )

    aligned_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    offset = aligned_manifest["zmq_clock_offsets"]["zmq_source_2"]
    assert offset == {
        "source": 2,
        "offset_ms": 50.0,
        "offset_ns": offset_ns,
        "frame_count": 3,
    }
    assert result.to_manifest_entry(started_ns=1, finished_ns=2)["zmq_clock_offsets"] == aligned_manifest["zmq_clock_offsets"]
    assert not any("clock offset" in warning for warning in aligned_manifest["warnings"])
    assert "## ZMQ Clock Offsets" in result.report_path.read_text(encoding="utf-8")


def test_timestamp_alignment_warns_on_large_zmq_clock_offset(tmp_path):
    demo_dir = tmp_path / "demo"
    offset_ns = 150_000_000
    _write_timestamp_alignment_demo(demo_dir, zmq_offset_ns=offset_ns)

    result = align_demo_timestamps(
        demo_dir,
        AlignmentOptions(repo_root=REPO_ROOT, base="xense:pair", start_trim_s=0.0),
    )

    aligned_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert aligned_manifest["zmq_clock_offsets"]["zmq_source_2"]["offset_ms"] == 150.0
    assert aligned_manifest["warnings"] == [
        "ZMQ source 2 clock offset 150.000 ms exceeds 100.000 ms; "
        "check chrony/NTP sync: run `chronyc sources -v`; expected first line: ^*192.168.10.1"
    ]
