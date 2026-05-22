from __future__ import annotations

import json
import socket
import struct
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import FT300S.protocol.messages as ft300_protocol
from MainController.buffers import DemoStore
from MainController.config import RuntimeConfig
from MainController.main import Command, ControllerState, MainController
from MainController.realsense_image_guard import ImageReadinessResult, ImageTopicBaseline, validate_rosbag_image_metadata
from MainController.realsense_metadata import RealSenseMetadataEvent
from MainController.uds_client import MsgType
from MainController.zmq_telemetry import FRAME_STRUCT, MAGIC, VERSION
import XenseTacSensor.protocol.messages as xense_protocol


def wait_for(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError('condition did not become true before timeout')


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError('socket closed')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


class MockUdsSensor:
    def __init__(self, name: str, socket_path: Path, hz: float, protocol: Any):
        self.name = name
        self.socket_path = socket_path
        self.hz = hz
        self.protocol = protocol
        self.saved_file = str(socket_path.with_suffix('.npy'))
        self.commands: list[str] = []
        self.error_commands: set[str] = set()
        self.received_magics: list[bytes] = []
        self.frames_sent = 0
        self._stop = threading.Event()
        self._collecting = threading.Event()
        self._ready = threading.Event()
        self._send_lock = threading.Lock()
        self._server: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._server_thread = threading.Thread(target=self._run_server, name=f'MockUdsSensor:{name}', daemon=True)
        self._frame_thread = threading.Thread(target=self._run_frames, name=f'MockUdsSensorFrames:{name}', daemon=True)

    def start(self) -> None:
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._server_thread.start()
        self._frame_thread.start()
        wait_for(self._ready.is_set)

    def stop(self) -> None:
        self._stop.set()
        self._collecting.clear()
        for sock in (self._conn, self._server):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._server_thread.join(timeout=1.0)
        self._frame_thread.join(timeout=1.0)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def _run_server(self) -> None:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server = server
        server.bind(str(self.socket_path))
        server.listen(1)
        server.settimeout(0.1)
        self._ready.set()
        while not self._stop.is_set():
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self._conn = conn
            conn.settimeout(0.1)
            self._serve_connection(conn)

    def _serve_connection(self, conn: socket.socket) -> None:
        while not self._stop.is_set():
            try:
                header = recv_exact(conn, self.protocol.HEADER_SIZE)
                self.received_magics.append(header[:2])
                _version, msg_type, _flags, payload_len, _frame_id = self.protocol.unpack_header(header)
                if payload_len:
                    self.protocol.decode_payload(recv_exact(conn, payload_len))
                self._handle_command(msg_type)
            except socket.timeout:
                continue
            except Exception:
                self._collecting.clear()
                return

    def _handle_command(self, msg_type: Any) -> None:
        protocol_msg = self.protocol.MsgType
        if msg_type == protocol_msg.INIT_REQ:
            self._send(protocol_msg.INIT_READY, payload={'sensor': self.name})
            return
        if msg_type == protocol_msg.START_REQ:
            self.commands.append('START_REQ')
            if self._send_error_for('START_REQ'):
                return
            self._send_ack('START_REQ')
            self._collecting.set()
            return
        if msg_type == protocol_msg.PAUSE_REQ:
            self.commands.append('PAUSE_REQ')
            if self._send_error_for('PAUSE_REQ'):
                return
            self._collecting.clear()
            self._send_ack('PAUSE_REQ')
            return
        if msg_type == protocol_msg.DEMO_DONE_REQ:
            self.commands.append('DEMO_DONE_REQ')
            if self._send_error_for('DEMO_DONE_REQ'):
                return
            self._collecting.clear()
            self._send_ack('DEMO_DONE_REQ', saved_file=self.saved_file)
            return
        if msg_type == protocol_msg.DEMO_DISCARD_REQ:
            self.commands.append('DEMO_DISCARD_REQ')
            if self._send_error_for('DEMO_DISCARD_REQ'):
                return
            self._collecting.clear()
            self._send_ack('DEMO_DISCARD_REQ')
            return
        if msg_type == protocol_msg.STOP_REQ:
            self.commands.append('STOP_REQ')
            if self._send_error_for('STOP_REQ'):
                return
            self._collecting.clear()
            self._send_ack('STOP_REQ')

    def _run_frames(self) -> None:
        period_s = 1.0 / self.hz
        next_time = time.monotonic()
        while not self._stop.is_set():
            if not self._collecting.is_set():
                time.sleep(0.002)
                next_time = time.monotonic()
                continue
            now = time.monotonic()
            if now < next_time:
                time.sleep(min(0.002, next_time - now))
                continue
            self.frames_sent += 1
            stamp_ns = time.time_ns()
            if self.name == 'xense':
                payload = {'timestamp_ns_0': stamp_ns, 'timestamp_ns_1': stamp_ns + 100}
            else:
                payload = {'timestamp_ns': stamp_ns}
            self._send(self.protocol.MsgType.FRAME_READY, frame_id=self.frames_sent, payload=payload)
            next_time += period_s

    def _send_ack(self, cmd: str, **extra: Any) -> None:
        payload = {'cmd': cmd, **extra}
        self._send(self.protocol.MsgType.ACK, payload=payload)

    def _send_error_for(self, cmd: str) -> bool:
        if cmd not in self.error_commands:
            return False
        self._send(self.protocol.MsgType.ERROR, payload={'cmd': cmd, 'reason': f'injected {cmd} error'})
        return True

    def _send(self, msg_type: Any, frame_id: int = -1, payload: dict[str, Any] | None = None) -> None:
        conn = self._conn
        if conn is None:
            return
        data = self.protocol.pack_message(msg_type, frame_id=frame_id, payload=payload)
        with self._send_lock:
            try:
                conn.sendall(data)
            except OSError:
                self._collecting.clear()


class LocalZmqTelemetryPublisher:
    def __init__(self, hz: float = 80.0, source: int = 1):
        pytest.importorskip('zmq')
        import zmq

        self._zmq = zmq
        self.hz = hz
        self.source = source
        self.seq = 0
        self._stop = threading.Event()
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self.endpoint = f'inproc://mock-telemetry-{uuid.uuid4().hex}'
        self._socket.bind(self.endpoint)
        self._thread = threading.Thread(target=self._run, name='LocalZmqTelemetryPublisher', daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._socket.close(linger=0)

    def _run(self) -> None:
        period_s = 1.0 / self.hz
        floats = tuple(float(index) for index in range(58))
        next_time = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_time:
                time.sleep(min(0.002, next_time - now))
                continue
            self.seq += 1
            payload = FRAME_STRUCT.pack(
                MAGIC,
                VERSION,
                self.source,
                0,
                self.seq,
                time.time_ns() / 1_000_000_000.0,
                1,
                *floats,
                0,
                0,
            )
            try:
                self._socket.send(payload)
            except Exception:
                return
            next_time += period_s


class FakeRealSenseMetadataMonitor:
    def __init__(self, topics, on_event):
        self.topics = topics
        self.on_event = on_event
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeRosbagControl:
    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []
        self.fail_methods: set[str] = set()
        self.readiness_missing_topics: tuple[str, ...] = ()
        self.postcheck_topic_metadata: dict[str, dict[str, Any]] | None = None
        self.readiness_requirements: tuple[Any, ...] = ()
        self.postcheck_requirements: tuple[Any, ...] = ()

    def wait_ready(self, _timeout_s: float) -> bool:
        return True

    def record(self, uri: Path, timeout_s: float) -> None:
        if 'record' in self.fail_methods:
            raise RuntimeError('injected rosbag record failure')
        self.calls.append(('record', str(uri)))

    def resume(self, timeout_s: float) -> None:
        if 'resume' in self.fail_methods:
            raise RuntimeError('injected rosbag resume failure')
        self.calls.append(('resume', None))

    def pause(self, timeout_s: float) -> None:
        if 'pause' in self.fail_methods:
            raise RuntimeError('injected rosbag pause failure')
        self.calls.append(('pause', None))

    def stop(self, timeout_s: float) -> None:
        if 'stop' in self.fail_methods:
            raise RuntimeError('injected rosbag stop failure')
        self.calls.append(('stop', None))

    def check_image_readiness(self, requirements, timeout_s: float, mode: str):
        self.readiness_requirements = requirements
        missing = set(self.readiness_missing_topics)
        baselines = [
            ImageTopicBaseline(
                topic=requirement.topic,
                message_type=requirement.message_type,
                width=requirement.width,
                height=requirement.height,
                encoding=requirement.encoding,
                step=requirement.step,
                stream_role=requirement.stream_role,
            )
            for requirement in requirements
            if requirement.topic not in missing
        ]
        return ImageReadinessResult(
            ok=not missing,
            mode=mode,
            required_topics=tuple(requirement.topic for requirement in requirements),
            baselines=tuple(baselines),
            missing_topics=tuple(self.readiness_missing_topics),
        )

    def validate_recorded_images(self, rosbag_uri: Path, requirements, count_skew_limit: int, mode: str):
        self.postcheck_requirements = requirements
        metadata = self.postcheck_topic_metadata
        if metadata is None:
            metadata = {
                requirement.topic: {'message_type': requirement.message_type, 'count': 10}
                for requirement in requirements
            }
        return validate_rosbag_image_metadata(
            mode=mode,
            rosbag_uri=rosbag_uri,
            requirements=requirements,
            topic_metadata=metadata,
            count_skew_limit=count_skew_limit,
        )

    def close(self) -> None:
        self.calls.append(('close', None))


class FakeSensorClient:
    def __init__(self, name: str):
        self.name = name
        self.commands: list[str] = []
        self.stop_count = 0

    def send_and_wait_ack(self, _msg_type, cmd_name, timeout_s, **_kwargs):
        self.commands.append(cmd_name)
        return {'cmd': cmd_name}

    def stop(self) -> None:
        self.stop_count += 1

    def last_error_for(self, _cmd_name: str):
        return None


class FakeProcess:
    def __init__(self):
        self.start_count = 0
        self.restart_count = 0
        self.stop_count = 0

    def start(self) -> None:
        self.start_count += 1

    def restart(self) -> None:
        self.restart_count += 1

    def stop(self) -> None:
        self.stop_count += 1


class FakeReceiver:
    def __init__(self):
        self.stop_count = 0

    def stop(self) -> None:
        self.stop_count += 1


class MockRuntime:
    def __init__(self, tmp_path: Path, monkeypatch, **config_overrides):
        from MainController import main as main_module

        self.tmp_path = tmp_path
        self.ft300 = MockUdsSensor('ft300', tmp_path / 'ft300.sock', hz=100.0, protocol=ft300_protocol)
        self.xense = MockUdsSensor('xense', tmp_path / 'xense.sock', hz=30.0, protocol=xense_protocol)
        self.zmq_pub = LocalZmqTelemetryPublisher(hz=80.0)
        self.rosbag = FakeRosbagControl()
        self.config_overrides = config_overrides

        monkeypatch.setattr(main_module, 'RealSenseMetadataMonitor', FakeRealSenseMetadataMonitor)
        monkeypatch.setattr(main_module, 'RosbagControl', lambda: self.rosbag)

        self.controller: MainController | None = None
        self._monkeypatch = monkeypatch

    def __enter__(self):
        self.ft300.start()
        self.xense.start()
        self.zmq_pub.start()

        config = RuntimeConfig(
            output_dir=self.tmp_path / 'sessions',
            zmq_connect=self.zmq_pub.endpoint,
            ft_uds_path=str(self.tmp_path / 'ft300.sock'),
            xense_uds_path=str(self.tmp_path / 'xense.sock'),
            startup_timeout_s=3.0,
            init_timeout_s=1.0,
            ack_timeout_s=1.0,
            zmq_first_frame_timeout_s=3.0,
            rosbag_timeout_s=1.0,
            **self.config_overrides,
        )
        self.controller = MainController(config)
        self._monkeypatch.setattr(self.controller, '_start_processes', lambda: None)
        self.controller.startup()
        assert self.controller.get_state() == ControllerState.WAIT_START
        assert self.ft300.received_magics
        assert self.xense.received_magics
        assert all(magic == ft300_protocol.MAGIC for magic in self.ft300.received_magics)
        assert all(magic == xense_protocol.MAGIC for magic in self.xense.received_magics)
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        if self.controller is not None:
            self.controller.stop_all()
        self.zmq_pub.stop()
        self.ft300.stop()
        self.xense.stop()

    def start_and_wait_for_frames(self) -> Path:
        assert self.controller is not None
        self.controller.start_or_resume_demo()
        assert self.controller.get_state() == ControllerState.COLLECTING
        wait_for(lambda: self.controller.demo_store is not None and len(self.controller.demo_store.ft300) >= 4)
        wait_for(lambda: self.controller.demo_store is not None and len(self.controller.demo_store.xense) >= 1)
        wait_for(lambda: self.controller.demo_store is not None and len(self.controller.demo_store.zmq) >= 4)
        assert self.controller.demo_store is not None
        return self.controller.demo_store.demo_dir

    def wait_for_zmq_drain_outside_demo(self, previous_key: int | None) -> None:
        assert self.controller is not None
        monitor = self.controller.drop_monitors['zmq_source_1']
        wait_for(lambda: monitor.previous_key is not None and monitor.previous_key != previous_key)


def emit_realsense_metadata(controller: MainController, frame_number: int) -> None:
    for index, topic in enumerate(controller.config.realsense_metadata_topics):
        stamp_ns = time.time_ns() + index
        controller._on_realsense_metadata(
            RealSenseMetadataEvent(
                topic=topic,
                frame_number=frame_number,
                header_stamp_ns=stamp_ns,
                frame_timestamp_ns=stamp_ns // 1_000_000 * 1_000_000,
                hw_timestamp_ns=stamp_ns // 1_000_000 * 1_000_000,
                recv_time_ns=time.time_ns(),
                recv_monotonic_ns=time.monotonic_ns(),
            )
        )


def run_with_timeout(fn, timeout_s: float = 1.0):
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result['value'] = fn()
        except BaseException as exc:
            result['exc'] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    if thread.is_alive():
        raise AssertionError('operation did not complete before timeout')
    if 'exc' in result:
        raise result['exc']
    return result.get('value')


def assert_npz_fields_same_length(npz) -> int:
    assert npz.files
    expected = len(npz[npz.files[0]])
    for field in npz.files:
        assert len(npz[field]) == expected, field
    return expected


def test_default_realsense_topics_are_four_cameras_eight_streams():
    topics = RuntimeConfig().realsense_metadata_topics

    assert len(topics) == 8
    assert topics == (
        '/cam1/camera/color/metadata',
        '/cam1/camera/depth/metadata',
        '/cam2/camera/color/metadata',
        '/cam2/camera/depth/metadata',
        '/cam3/camera/color/metadata',
        '/cam3/camera/depth/metadata',
        '/cam4/camera/color/metadata',
        '/cam4/camera/depth/metadata',
    )


def test_mock_runtime_start_pause_resume_done(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None
        demo_dir = runtime.start_and_wait_for_frames()

        emit_realsense_metadata(controller, frame_number=1)
        assert len(controller.demo_store.realsense) == 8

        rows_before_pause = len(controller.demo_store.zmq)
        wait_for(lambda: 'zmq_source_1' in controller.drop_monitors)
        monitor = controller.drop_monitors['zmq_source_1']
        key_before_pause = monitor.previous_key

        assert controller.pause_demo(reason='test')
        assert controller.get_state() == ControllerState.PAUSED
        wait_for(lambda: monitor.previous_key is not None and monitor.previous_key != key_before_pause)
        assert len(controller.demo_store.zmq) == rows_before_pause

        controller.start_or_resume_demo()
        assert controller.get_state() == ControllerState.COLLECTING
        wait_for(lambda: len(controller.demo_store.ft300) > 4)
        emit_realsense_metadata(controller, frame_number=2)

        controller.finish_demo()

        assert controller.get_state() == ControllerState.WAIT_START
        assert runtime.rosbag.calls[:5] == [
            ('record', str(demo_dir / 'rosbag')),
            ('resume', None),
            ('pause', None),
            ('resume', None),
            ('stop', None),
        ]
        assert runtime.ft300.commands[:3] == ['START_REQ', 'PAUSE_REQ', 'START_REQ']
        assert 'DEMO_DONE_REQ' in runtime.ft300.commands
        assert 'DEMO_DONE_REQ' in runtime.xense.commands
        assert all(magic == ft300_protocol.MAGIC for magic in runtime.ft300.received_magics)
        assert all(magic == xense_protocol.MAGIC for magic in runtime.xense.received_magics)

        manifest = json.loads((demo_dir / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'done'
        assert manifest['sensor_saved_files']['ft300'] == runtime.ft300.saved_file
        assert manifest['sensor_saved_files']['xense'] == runtime.xense.saved_file

        ft_npz = np.load(demo_dir / 'ft300_timestamps.npz', allow_pickle=True)
        xense_npz = np.load(demo_dir / 'xense_timestamps.npz', allow_pickle=True)
        realsense_npz = np.load(demo_dir / 'realsense_metadata.npz', allow_pickle=True)
        zmq_npz = np.load(demo_dir / 'zmq_telemetry.npz', allow_pickle=True)
        ft_rows = assert_npz_fields_same_length(ft_npz)
        xense_rows = assert_npz_fields_same_length(xense_npz)
        realsense_rows = assert_npz_fields_same_length(realsense_npz)
        zmq_rows = assert_npz_fields_same_length(zmq_npz)
        assert manifest['frame_counts'] == {
            'ft300': ft_rows,
            'xense': xense_rows,
            'realsense': realsense_rows,
            'zmq': zmq_rows,
        }
        assert ft_rows >= 4
        assert xense_rows >= 1
        assert realsense_rows == 16
        assert zmq_rows >= 4


def test_mock_runtime_paused_finish_returns_to_wait_start(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None
        runtime.start_and_wait_for_frames()

        assert controller.pause_demo(reason='test')
        assert controller.get_state() == ControllerState.PAUSED

        run_with_timeout(controller.finish_demo)

        assert controller.get_state() == ControllerState.WAIT_START
        assert runtime.ft300.commands == ['START_REQ', 'PAUSE_REQ', 'DEMO_DONE_REQ']
        assert runtime.xense.commands == ['START_REQ', 'PAUSE_REQ', 'DEMO_DONE_REQ']


def test_mock_runtime_paused_discard_returns_to_wait_start(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None
        runtime.start_and_wait_for_frames()

        assert controller.pause_demo(reason='test')
        assert controller.get_state() == ControllerState.PAUSED

        run_with_timeout(controller.discard_demo)

        assert controller.get_state() == ControllerState.WAIT_START
        assert controller.demo_store is None
        assert runtime.ft300.commands == ['START_REQ', 'PAUSE_REQ', 'DEMO_DISCARD_REQ']
        assert runtime.xense.commands == ['START_REQ', 'PAUSE_REQ', 'DEMO_DISCARD_REQ']


def test_uds_error_response_wakes_ack_waiter(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None
        runtime.xense.error_commands.add('DEMO_DONE_REQ')

        result = run_with_timeout(
            lambda: controller.xense_client.send_and_wait_ack(
                MsgType.DEMO_DONE_REQ,
                'DEMO_DONE_REQ',
                timeout_s=None,
                progress_period_s=100.0,
            )
        )

        assert result is None
        assert runtime.xense.commands == ['DEMO_DONE_REQ']


def test_startup_failure_cleans_started_resources_and_reraises(tmp_path, monkeypatch):
    controller = MainController(RuntimeConfig(output_dir=tmp_path / 'sessions', ack_timeout_s=0.01, rosbag_timeout_s=0.01))
    fake_process = FakeProcess()
    fake_receiver = FakeReceiver()
    fake_monitor = FakeRealSenseMetadataMonitor((), lambda _event: None)
    fake_rosbag = FakeRosbagControl()

    def start_processes() -> None:
        controller.processes['ft300'] = fake_process

    def start_receivers() -> None:
        controller.zmq_receiver = fake_receiver
        controller.realsense_monitor = fake_monitor
        controller.rosbag = fake_rosbag

    def wait_startup_ready() -> None:
        raise RuntimeError('injected startup failure')

    monkeypatch.setattr(controller, '_start_processes', start_processes)
    monkeypatch.setattr(controller, '_start_receivers', start_receivers)
    monkeypatch.setattr(controller, '_wait_startup_ready', wait_startup_ready)

    with pytest.raises(RuntimeError, match='injected startup failure'):
        controller.startup()

    assert controller.get_state() == ControllerState.STOPPED
    assert fake_process.stop_count == 1
    assert fake_receiver.stop_count == 1
    assert fake_monitor.stopped
    assert ('stop', None) in fake_rosbag.calls
    assert ('close', None) in fake_rosbag.calls

    events = [
        json.loads(line)
        for line in controller.logger.path.read_text(encoding='utf-8').splitlines()
    ]
    assert any(event['event'] == 'startup_failed' for event in events)
    transitions = [event for event in events if event['event'] == 'state_transition']
    assert any(event.get('current') == 'ERROR' for event in transitions)
    assert any(event.get('current') == 'STOPPED' for event in transitions)


def test_start_processes_stops_earlier_processes_on_later_failure(tmp_path, monkeypatch):
    from MainController import main as main_module

    instances: list[Any] = []

    class FakeManagedProcess:
        def __init__(
            self,
            name,
            cmd,
            cwd,
            log_path,
            fatal_patterns=(),
            on_fatal=None,
            on_exit=None,
        ):
            self.name = name
            self.cmd = cmd
            self.stop_count = 0
            self.started = False
            instances.append(self)

        def start(self) -> None:
            if self.name == 'xense':
                raise RuntimeError('xense start failed')
            self.started = True

        def stop(self) -> None:
            self.stop_count += 1

    monkeypatch.setattr(main_module, 'ManagedProcess', FakeManagedProcess)
    controller = MainController(RuntimeConfig(output_dir=tmp_path / 'sessions'))

    with pytest.raises(RuntimeError, match='xense start failed'):
        controller._start_processes()

    by_name = {process.name: process for process in instances}
    assert by_name['ft300'].started
    assert by_name['ft300'].stop_count == 1
    assert by_name['xense'].stop_count == 0
    assert by_name['realsense_camera'].stop_count == 0
    assert by_name['rosbag_recorder'].stop_count == 0


def test_start_transaction_rolls_back_acked_sensor_on_later_sensor_error(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None
        runtime.xense.error_commands.add('START_REQ')

        controller.start_or_resume_demo()

        assert controller.get_state() == ControllerState.WAIT_START
        assert controller.demo_store is None
        assert runtime.ft300.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
        assert runtime.xense.commands == ['START_REQ']
        demo_dirs = sorted((controller.session_dir / 'demos').iterdir())
        assert len(demo_dirs) == 1
        manifest = json.loads((demo_dirs[0] / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'failed'
        assert manifest['failure_stage'] == 'xense_start'
        assert manifest['acked_start_sensors'] == ['ft300']
        assert manifest['rollback_action'] == 'DEMO_DISCARD_REQ'
        assert manifest['rollback_results']['ft300']['ok'] is True
        assert manifest['npz'] == {}
        assert not (demo_dirs[0] / 'ft300_timestamps.npz').exists()


def test_resume_transaction_failure_invalidates_paused_demo(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None
        demo_dir = runtime.start_and_wait_for_frames()
        assert controller.pause_demo(reason='test')
        runtime.xense.error_commands.add('START_REQ')

        controller.start_or_resume_demo()

        assert controller.get_state() == ControllerState.WAIT_START
        assert controller.demo_store is None
        assert runtime.ft300.commands == ['START_REQ', 'PAUSE_REQ', 'START_REQ', 'DEMO_DISCARD_REQ']
        assert runtime.xense.commands == ['START_REQ', 'PAUSE_REQ', 'START_REQ']
        manifest = json.loads((demo_dir / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'failed'
        assert manifest['failure_stage'] == 'xense_start'
        assert manifest['new_demo'] is False
        assert manifest['acked_start_sensors'] == ['ft300']
        assert manifest['npz'] == {}


def test_rosbag_resume_failure_rolls_back_started_sensors_and_writes_failed_manifest(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None
        runtime.rosbag.fail_methods.add('resume')

        controller.start_or_resume_demo()

        assert controller.get_state() == ControllerState.WAIT_START
        assert controller.demo_store is None
        assert runtime.ft300.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
        assert runtime.xense.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
        demo_dirs = sorted((controller.session_dir / 'demos').iterdir())
        assert len(demo_dirs) == 1
        manifest = json.loads((demo_dirs[0] / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'failed'
        assert manifest['failure_stage'] == 'rosbag_resume'
        assert manifest['acked_start_sensors'] == ['ft300', 'xense']
        assert manifest['rosbag_record_resume']['record_started'] is True
        assert manifest['rosbag_record_resume']['failed_action'] == 'resume'
        assert manifest['rosbag_record_resume']['stop']['ok'] is True
        assert manifest['npz'] == {}
        assert runtime.rosbag.calls == [
            ('record', str(demo_dirs[0] / 'rosbag')),
            ('stop', None),
        ]


def test_realsense_readiness_failure_blocks_formal_recording(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None
        missing_topic = controller.config.realsense_image_requirements[-1].topic
        runtime.rosbag.readiness_missing_topics = (missing_topic,)

        controller.start_or_resume_demo()

        assert controller.get_state() == ControllerState.WAIT_START
        assert controller.demo_store is None
        assert runtime.ft300.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
        assert runtime.xense.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
        demo_dirs = sorted((controller.session_dir / 'demos').iterdir())
        manifest = json.loads((demo_dirs[0] / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'failed'
        assert manifest['failure_stage'] == 'realsense_image_readiness'
        readiness = manifest['rosbag_record_resume']['image_readiness']
        assert readiness['ok'] is False
        assert readiness['mode'] == 'formal'
        assert readiness['missing_topics'] == [missing_topic]
        assert runtime.rosbag.calls == []


def test_realsense_rosbag_postcheck_failure_marks_demo_failed(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None
        demo_dir = runtime.start_and_wait_for_frames()
        requirements = controller.config.realsense_image_requirements
        runtime.rosbag.postcheck_topic_metadata = {
            requirement.topic: {'message_type': requirement.message_type, 'count': 10}
            for requirement in requirements[:-1]
        }

        controller.finish_demo()

        manifest = json.loads((demo_dir / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'failed'
        assert manifest['realsense_image_readiness']['ok'] is True
        assert manifest['realsense_rosbag_postcheck']['ok'] is False
        assert manifest['realsense_rosbag_postcheck']['missing_topics'] == [requirements[-1].topic]
        assert manifest['npz']


def test_realsense_debug_degraded_mode_uses_configured_subset(tmp_path, monkeypatch):
    subset = (
        '/cam3/camera/color/image_raw',
        '/cam3/camera/aligned_depth_to_color/image_raw',
    )
    with MockRuntime(
        tmp_path,
        monkeypatch,
        realsense_capture_mode='debug_degraded',
        realsense_debug_image_topics=subset,
    ) as runtime:
        controller = runtime.controller
        assert controller is not None
        demo_dir = runtime.start_and_wait_for_frames()
        controller.finish_demo()

        assert [requirement.topic for requirement in runtime.rosbag.readiness_requirements] == list(subset)
        assert [requirement.topic for requirement in runtime.rosbag.postcheck_requirements] == list(subset)
        manifest = json.loads((demo_dir / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'done'
        assert manifest['realsense_image_readiness']['mode'] == 'debug_degraded'
        assert manifest['realsense_image_readiness']['required_topics'] == list(subset)
        assert manifest['realsense_rosbag_postcheck']['required_topics'] == list(subset)


def test_pause_partial_failure_writes_failed_manifest_and_stops(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None
        demo_dir = runtime.start_and_wait_for_frames()
        runtime.xense.error_commands.add('PAUSE_REQ')

        assert controller.pause_demo(reason='test') is False

        assert controller.get_state() == ControllerState.STOPPED
        manifest = json.loads((demo_dir / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'failed'
        assert manifest['failure_stage'] == 'pause_command'
        assert manifest['command_results']['ft300']['ok'] is True
        assert manifest['command_results']['xense']['ok'] is False
        assert manifest['npz'] == {}


def test_finish_partial_failure_writes_failed_manifest_and_stops(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None
        demo_dir = runtime.start_and_wait_for_frames()
        runtime.xense.error_commands.add('DEMO_DONE_REQ')

        controller.finish_demo()

        assert controller.get_state() == ControllerState.STOPPED
        manifest = json.loads((demo_dir / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'failed'
        assert manifest['failure_stage'] == 'finish_command'
        assert manifest['sensor_saved_files']['ft300'] == runtime.ft300.saved_file
        assert manifest['sensor_saved_files']['xense'] is None
        assert manifest['command_results']['ft300']['ok'] is True
        assert manifest['command_results']['xense']['ok'] is False
        assert manifest['npz']


def test_discard_partial_failure_writes_failed_manifest_and_stops(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None
        demo_dir = runtime.start_and_wait_for_frames()
        runtime.xense.error_commands.add('DEMO_DISCARD_REQ')

        controller.discard_demo()

        assert controller.get_state() == ControllerState.STOPPED
        manifest = json.loads((demo_dir / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'failed'
        assert manifest['failure_stage'] == 'discard_command'
        assert manifest['command_results']['ft300']['ok'] is True
        assert manifest['command_results']['xense']['ok'] is False
        assert manifest['npz'] == {}


def test_zmq_warning_does_not_stop_controller(tmp_path):
    controller = MainController(RuntimeConfig(output_dir=tmp_path / 'sessions'))
    controller.set_state(ControllerState.COLLECTING)

    controller._on_zmq_error('invalid ZMQ frame: injected')

    assert controller.get_state() == ControllerState.COLLECTING


def test_zmq_fatal_stops_controller_and_cleans_resources(tmp_path):
    controller = MainController(RuntimeConfig(output_dir=tmp_path / 'sessions', ack_timeout_s=0.01, rosbag_timeout_s=0.01))
    fake_rosbag = FakeRosbagControl()
    fake_receiver = FakeReceiver()
    fake_monitor = FakeRealSenseMetadataMonitor((), lambda _event: None)
    fake_process = FakeProcess()
    controller.ft_client = FakeSensorClient('ft300')
    controller.xense_client = FakeSensorClient('xense')
    controller.rosbag = fake_rosbag
    controller.zmq_receiver = fake_receiver
    controller.realsense_monitor = fake_monitor
    controller.processes['ft300'] = fake_process
    controller.set_state(ControllerState.COLLECTING)

    controller.handle_command(Command('zmq_fatal', {'message': 'injected receiver failure'}))

    assert controller.get_state() == ControllerState.STOPPED
    assert controller.ft_client.commands == ['STOP_REQ']
    assert controller.xense_client.commands == ['STOP_REQ']
    assert ('stop', None) in fake_rosbag.calls
    assert ('close', None) in fake_rosbag.calls
    assert fake_receiver.stop_count == 1
    assert fake_monitor.stopped
    assert fake_process.stop_count == 1


def test_mock_runtime_start_done_start_done_keeps_zmq_drain_between_demos(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None

        first_demo_dir = runtime.start_and_wait_for_frames()
        first_monitor_key = controller.drop_monitors['zmq_source_1'].previous_key
        controller.finish_demo()

        assert controller.get_state() == ControllerState.WAIT_START
        assert controller.demo_store is None
        runtime.wait_for_zmq_drain_outside_demo(first_monitor_key)

        second_demo_dir = runtime.start_and_wait_for_frames()
        assert second_demo_dir != first_demo_dir
        assert len(controller.demo_store.zmq) >= 4
        controller.finish_demo()

        assert controller.get_state() == ControllerState.WAIT_START
        first_manifest = json.loads((first_demo_dir / 'manifest.json').read_text(encoding='utf-8'))
        second_manifest = json.loads((second_demo_dir / 'manifest.json').read_text(encoding='utf-8'))
        assert first_manifest['status'] == 'done'
        assert second_manifest['status'] == 'done'
        first_zmq = np.load(first_demo_dir / 'zmq_telemetry.npz', allow_pickle=True)
        second_zmq = np.load(second_demo_dir / 'zmq_telemetry.npz', allow_pickle=True)
        assert len(first_zmq['seq']) > 0
        assert len(second_zmq['seq']) > 0
        assert int(second_zmq['seq'][0]) > int(first_zmq['seq'][-1])
        assert runtime.rosbag.calls == [
            ('record', str(first_demo_dir / 'rosbag')),
            ('resume', None),
            ('stop', None),
            ('record', str(second_demo_dir / 'rosbag')),
            ('resume', None),
            ('stop', None),
        ]


def test_mock_runtime_start_discard_start_done_keeps_zmq_drain_after_discard(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None

        discarded_demo_dir = runtime.start_and_wait_for_frames()
        monitor_key_before_discard = controller.drop_monitors['zmq_source_1'].previous_key
        controller.discard_demo()

        assert controller.get_state() == ControllerState.WAIT_START
        assert controller.demo_store is None
        discard_manifest = json.loads((discarded_demo_dir / 'manifest.json').read_text(encoding='utf-8'))
        assert discard_manifest['status'] == 'discarded'
        assert discard_manifest['npz'] == {}
        assert discard_manifest['frame_counts']['ft300'] >= 4
        assert discard_manifest['frame_counts']['xense'] >= 1
        assert discard_manifest['frame_counts']['zmq'] >= 4
        assert not (discarded_demo_dir / 'ft300_timestamps.npz').exists()
        assert not (discarded_demo_dir / 'xense_timestamps.npz').exists()
        assert not (discarded_demo_dir / 'realsense_metadata.npz').exists()
        assert not (discarded_demo_dir / 'zmq_telemetry.npz').exists()
        runtime.wait_for_zmq_drain_outside_demo(monitor_key_before_discard)

        saved_demo_dir = runtime.start_and_wait_for_frames()
        assert saved_demo_dir != discarded_demo_dir
        controller.finish_demo()

        manifest = json.loads((saved_demo_dir / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'done'
        assert runtime.ft300.commands == ['START_REQ', 'DEMO_DISCARD_REQ', 'START_REQ', 'DEMO_DONE_REQ']
        assert runtime.xense.commands == ['START_REQ', 'DEMO_DISCARD_REQ', 'START_REQ', 'DEMO_DONE_REQ']
        assert runtime.rosbag.calls == [
            ('record', str(discarded_demo_dir / 'rosbag')),
            ('resume', None),
            ('stop', None),
            ('record', str(saved_demo_dir / 'rosbag')),
            ('resume', None),
            ('stop', None),
        ]


def test_realsense_fatal_pauses_collecting_and_restarts(tmp_path):
    controller = MainController(RuntimeConfig(output_dir=tmp_path / 'sessions'))
    fake_rosbag = FakeRosbagControl()
    fake_process = FakeProcess()
    controller.ft_client = FakeSensorClient('ft300')
    controller.xense_client = FakeSensorClient('xense')
    controller.rosbag = fake_rosbag
    controller.processes['realsense_camera'] = fake_process
    controller.demo_store = DemoStore(tmp_path / 'demo')
    controller.set_state(ControllerState.COLLECTING)

    controller.handle_realsense_fatal({'line': 'Hardware Error', 'process': 'realsense_camera'})

    assert controller.get_state() == ControllerState.PAUSED
    assert controller.ft_client.commands == ['PAUSE_REQ']
    assert controller.xense_client.commands == ['PAUSE_REQ']
    assert ('pause', None) in fake_rosbag.calls
    assert fake_process.restart_count == 1
    assert controller.realsense_restart_count == 1
