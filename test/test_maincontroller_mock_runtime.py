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

from MainController.buffers import DemoStore
from MainController.config import RuntimeConfig
from MainController.main import ControllerState, MainController
from MainController.realsense_metadata import RealSenseMetadataEvent
from MainController.uds_client import HEADER_SIZE, MsgType, decode_payload, pack_message, unpack_header
from MainController.zmq_telemetry import FRAME_STRUCT, MAGIC, VERSION


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
    def __init__(self, name: str, socket_path: Path, hz: float):
        self.name = name
        self.socket_path = socket_path
        self.hz = hz
        self.saved_file = str(socket_path.with_suffix('.npy'))
        self.commands: list[str] = []
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
                header = recv_exact(conn, HEADER_SIZE)
                _version, msg_type, _flags, payload_len, _frame_id = unpack_header(header)
                if payload_len:
                    decode_payload(recv_exact(conn, payload_len))
                self._handle_command(msg_type)
            except socket.timeout:
                continue
            except Exception:
                self._collecting.clear()
                return

    def _handle_command(self, msg_type: MsgType) -> None:
        if msg_type == MsgType.INIT_REQ:
            self._send(MsgType.INIT_READY, payload={'sensor': self.name})
            return
        if msg_type == MsgType.START_REQ:
            self.commands.append('START_REQ')
            self._send_ack('START_REQ')
            self._collecting.set()
            return
        if msg_type == MsgType.PAUSE_REQ:
            self.commands.append('PAUSE_REQ')
            self._collecting.clear()
            self._send_ack('PAUSE_REQ')
            return
        if msg_type == MsgType.DEMO_DONE_REQ:
            self.commands.append('DEMO_DONE_REQ')
            self._collecting.clear()
            self._send_ack('DEMO_DONE_REQ', saved_file=self.saved_file)
            return
        if msg_type == MsgType.DEMO_DISCARD_REQ:
            self.commands.append('DEMO_DISCARD_REQ')
            self._collecting.clear()
            self._send_ack('DEMO_DISCARD_REQ')
            return
        if msg_type == MsgType.STOP_REQ:
            self.commands.append('STOP_REQ')
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
            self._send(MsgType.FRAME_READY, frame_id=self.frames_sent, payload=payload)
            next_time += period_s

    def _send_ack(self, cmd: str, **extra: Any) -> None:
        payload = {'cmd': cmd, **extra}
        self._send(MsgType.ACK, payload=payload)

    def _send(self, msg_type: MsgType, frame_id: int = -1, payload: dict[str, Any] | None = None) -> None:
        conn = self._conn
        if conn is None:
            return
        data = pack_message(msg_type, frame_id=frame_id, payload=payload)
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

    def wait_ready(self, _timeout_s: float) -> bool:
        return True

    def record(self, uri: Path, timeout_s: float) -> None:
        self.calls.append(('record', str(uri)))

    def resume(self, timeout_s: float) -> None:
        self.calls.append(('resume', None))

    def pause(self, timeout_s: float) -> None:
        self.calls.append(('pause', None))

    def stop(self, timeout_s: float) -> None:
        self.calls.append(('stop', None))

    def close(self) -> None:
        self.calls.append(('close', None))


class FakeSensorClient:
    def __init__(self, name: str):
        self.name = name
        self.commands: list[str] = []

    def send_and_wait_ack(self, _msg_type, cmd_name, timeout_s, **_kwargs):
        self.commands.append(cmd_name)
        return {'cmd': cmd_name}


class FakeProcess:
    def __init__(self):
        self.restart_count = 0
        self.stop_count = 0

    def restart(self) -> None:
        self.restart_count += 1

    def stop(self) -> None:
        self.stop_count += 1


class MockRuntime:
    def __init__(self, tmp_path: Path, monkeypatch):
        from MainController import main as main_module

        self.tmp_path = tmp_path
        self.ft300 = MockUdsSensor('ft300', tmp_path / 'ft300.sock', hz=100.0)
        self.xense = MockUdsSensor('xense', tmp_path / 'xense.sock', hz=30.0)
        self.zmq_pub = LocalZmqTelemetryPublisher(hz=80.0)
        self.rosbag = FakeRosbagControl()

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
        )
        self.controller = MainController(config)
        self._monkeypatch.setattr(self.controller, '_start_processes', lambda: None)
        self.controller.startup()
        assert self.controller.get_state() == ControllerState.WAIT_START
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

        manifest = json.loads((demo_dir / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'done'
        assert manifest['sensor_saved_files']['ft300'] == runtime.ft300.saved_file
        assert manifest['sensor_saved_files']['xense'] == runtime.xense.saved_file

        ft_npz = np.load(demo_dir / 'ft300_timestamps.npz', allow_pickle=True)
        xense_npz = np.load(demo_dir / 'xense_timestamps.npz', allow_pickle=True)
        realsense_npz = np.load(demo_dir / 'realsense_metadata.npz', allow_pickle=True)
        zmq_npz = np.load(demo_dir / 'zmq_telemetry.npz', allow_pickle=True)
        assert len(ft_npz['frame_id']) >= 4
        assert len(xense_npz['frame_id']) >= 1
        assert len(realsense_npz['topic']) == 16
        assert len(zmq_npz['seq']) >= 4


def test_mock_runtime_start_done_start_done_keeps_zmq_drain_between_demos(tmp_path, monkeypatch):
    with MockRuntime(tmp_path, monkeypatch) as runtime:
        controller = runtime.controller
        assert controller is not None

        first_demo_dir = runtime.start_and_wait_for_frames()
        first_demo_rows = len(controller.demo_store.zmq)
        first_monitor_key = controller.drop_monitors['zmq_source_1'].previous_key
        controller.finish_demo()

        assert controller.get_state() == ControllerState.WAIT_START
        assert controller.demo_store is None
        runtime.wait_for_zmq_drain_outside_demo(first_monitor_key)

        second_demo_dir = runtime.start_and_wait_for_frames()
        assert second_demo_dir != first_demo_dir
        assert len(controller.demo_store.zmq) >= 4
        assert len(controller.demo_store.zmq) != first_demo_rows or second_demo_dir.exists()
        controller.finish_demo()

        assert controller.get_state() == ControllerState.WAIT_START
        first_manifest = json.loads((first_demo_dir / 'manifest.json').read_text(encoding='utf-8'))
        second_manifest = json.loads((second_demo_dir / 'manifest.json').read_text(encoding='utf-8'))
        assert first_manifest['status'] == 'done'
        assert second_manifest['status'] == 'done'
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
        assert not (discarded_demo_dir / 'manifest.json').exists()
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
