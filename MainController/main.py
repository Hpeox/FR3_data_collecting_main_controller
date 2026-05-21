"""MainController entrypoint and orchestration state machine."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .buffers import DemoStore, JsonlLogger
from .config import RuntimeConfig, ns_from_hz
from .drop_monitor import DropMonitor, DropWarning
from .processes import ManagedProcess, bash_cmd
from .realsense_metadata import RealSenseMetadataEvent, RealSenseMetadataMonitor
from .rosbag_control import RosbagControl
from .uds_client import MsgType, UdsClient, UdsEvent
from .zmq_telemetry import TelemetryFrame, ZmqTelemetryReceiver


class ControllerState(Enum):
    """Main controller state model."""

    BOOT = auto()
    STARTING_SERVICES = auto()
    INIT = auto()
    WAIT_START = auto()
    COLLECTING = auto()
    PAUSED = auto()
    FINALIZING = auto()
    DISCARDING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass(frozen=True)
class Command:
    """Command queued into the controller loop."""

    name: str
    payload: dict[str, Any] | None = None


class InputThread:
    """Blocking stdin reader that only enqueues commands."""

    def __init__(self, commands: queue.Queue[Command]):
        self.commands = commands
        self._thread = threading.Thread(target=self._run, name='InputThread', daemon=True)

    def start(self) -> None:
        """Start the input thread."""
        self._thread.start()

    def _run(self) -> None:
        print('Commands: s=start/resume, p=pause, d=done, x=discard, q=quit')
        while True:
            try:
                raw = input('cmd> ').strip().lower()
            except EOFError:
                self.commands.put(Command('q'))
                return
            except KeyboardInterrupt:
                self.commands.put(Command('q'))
                return
            if not raw:
                continue
            key = raw[:1]
            if key in {'s', 'p', 'd', 'x', 'q'}:
                self.commands.put(Command(key))
            else:
                print(f'unknown command: {raw}')


class MainController:
    """Orchestrate sensor services, ZMQ telemetry, rosbag, and demo state."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.session_dir = config.output_dir / time.strftime('session_%Y%m%d_%H%M%S')
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.logger = JsonlLogger(self.session_dir / 'controller_events.jsonl')
        self.commands: queue.Queue[Command] = queue.Queue()
        self.state = ControllerState.BOOT
        self.state_lock = threading.RLock()
        self.demo_store: DemoStore | None = None
        self.demo_started_ns: int | None = None
        self.queued_stop_after_finalizing = False
        self.rosbag_record_started = False
        self.rosbag_uri: Path | None = None
        self.sensor_saved_files: dict[str, str | None] = {}
        self.realsense_restart_count = 0
        self.realsense_restart_events: list[dict[str, Any]] = []

        self.drop_monitors: dict[str, DropMonitor] = {}
        self.processes: dict[str, ManagedProcess] = {}
        self.rosbag: RosbagControl | None = None
        self.realsense_monitor: RealSenseMetadataMonitor | None = None
        self.zmq_receiver: ZmqTelemetryReceiver | None = None
        self.ft_client = UdsClient('ft300', config.ft_uds_path, self._on_uds_event)
        self.xense_client = UdsClient('xense', config.xense_uds_path, self._on_uds_event)

    def run(self) -> None:
        """Start the controller and process commands until shutdown."""
        try:
            self.startup()
            InputThread(self.commands).start()
            while self.get_state() != ControllerState.STOPPED:
                try:
                    command = self.commands.get(timeout=0.2)
                except queue.Empty:
                    continue
                self.handle_command(command)
        except KeyboardInterrupt:
            self.log('keyboard_interrupt')
            self.stop_all()
        finally:
            self.logger.close()

    def startup(self) -> None:
        """Start services, connect streams, and initialize sensors."""
        self.set_state(ControllerState.STARTING_SERVICES)
        self._start_processes()
        self._start_receivers()
        self.set_state(ControllerState.INIT)
        self._wait_startup_ready()
        self.set_state(ControllerState.WAIT_START)
        self.log('ready')
        print('MainController ready.')

    def handle_command(self, command: Command) -> None:
        """Dispatch one command according to current state."""
        name = command.name
        if name == 's':
            self.start_or_resume_demo()
        elif name == 'p':
            self.pause_demo(reason='user')
        elif name == 'd':
            self.finish_demo()
        elif name == 'x':
            self.discard_demo()
        elif name == 'q':
            state = self.get_state()
            if state == ControllerState.FINALIZING:
                self.queued_stop_after_finalizing = True
                self.log('stop_queued_after_finalizing')
            else:
                self.stop_all()
        elif name == 'realsense_fatal':
            self.handle_realsense_fatal(command.payload or {})
        else:
            self.log('unknown_command', command=name)

    def start_or_resume_demo(self) -> None:
        """Start a new demo or resume a paused demo."""
        state = self.get_state()
        if state not in {ControllerState.WAIT_START, ControllerState.PAUSED}:
            self.reject_command('s', state)
            return

        new_demo = state == ControllerState.WAIT_START
        if new_demo:
            demo_tag = time.strftime('demo_%Y%m%d_%H%M%S')
            demo_dir = self.session_dir / 'demos' / demo_tag
            demo_dir.mkdir(parents=True, exist_ok=True)
            self.demo_store = DemoStore(demo_dir)
            self.demo_started_ns = time.time_ns()
            self.rosbag_record_started = False
            self.rosbag_uri = demo_dir / 'rosbag'
            self.sensor_saved_files = {}
            self.log('demo_created', demo_dir=str(demo_dir))

        if not self._sensor_command(self.ft_client, MsgType.START_REQ, 'START_REQ', self.config.ack_timeout_s):
            self.log('start_failed', sensor='ft300')
            return
        if not self._sensor_command(self.xense_client, MsgType.START_REQ, 'START_REQ', self.config.ack_timeout_s):
            self.log('start_failed', sensor='xense')
            return

        try:
            if self.rosbag is not None and self.rosbag_uri is not None:
                if not self.rosbag_record_started:
                    self.rosbag.record(self.rosbag_uri, timeout_s=self.config.rosbag_timeout_s)
                    self.rosbag_record_started = True
                self.rosbag.resume(timeout_s=self.config.rosbag_timeout_s)
        except Exception as exc:
            self.log('rosbag_start_failed', error=str(exc))
            print(f'[ERROR] rosbag start/resume failed: {exc}')
            return

        self.reset_drop_baselines()
        self.set_state(ControllerState.COLLECTING)
        self.log('demo_collecting', new_demo=new_demo)

    def pause_demo(self, reason: str) -> bool:
        """Pause current demo, including sensors and rosbag recording."""
        if self.get_state() != ControllerState.COLLECTING:
            self.reject_command('p', self.get_state())
            return False
        self.set_state(ControllerState.PAUSED)
        self.log('pause_started', reason=reason)
        ft_ok = self._sensor_command(self.ft_client, MsgType.PAUSE_REQ, 'PAUSE_REQ', self.config.ack_timeout_s)
        xense_ok = self._sensor_command(self.xense_client, MsgType.PAUSE_REQ, 'PAUSE_REQ', self.config.ack_timeout_s)
        try:
            if self.rosbag is not None:
                self.rosbag.pause(timeout_s=self.config.rosbag_timeout_s)
        except Exception as exc:
            self.log('rosbag_pause_failed', error=str(exc))
        self.reset_drop_baselines()
        self.log('pause_done', reason=reason, ft300=ft_ok, xense=xense_ok)
        return ft_ok and xense_ok

    def finish_demo(self) -> None:
        """Finish the current demo and save all controller buffers."""
        if self.get_state() not in {ControllerState.COLLECTING, ControllerState.PAUSED}:
            self.reject_command('d', self.get_state())
            return
        self.set_state(ControllerState.FINALIZING)
        self.log('finalizing_started')

        ft_payload = self._sensor_command_no_timeout(self.ft_client, MsgType.DEMO_DONE_REQ, 'DEMO_DONE_REQ')
        xense_payload = self._sensor_command_no_timeout(self.xense_client, MsgType.DEMO_DONE_REQ, 'DEMO_DONE_REQ')
        self.sensor_saved_files = {
            'ft300': None if ft_payload is None else ft_payload.get('saved_file'),
            'xense': None if xense_payload is None else xense_payload.get('saved_file'),
        }

        try:
            if self.rosbag is not None:
                self.rosbag.stop(timeout_s=self.config.rosbag_timeout_s)
        except Exception as exc:
            self.log('rosbag_stop_failed', error=str(exc))

        self._save_current_demo(status='done')
        self.demo_store = None
        self.rosbag_record_started = False
        self.rosbag_uri = None
        self.set_state(ControllerState.WAIT_START)
        self.log('finalizing_done')
        if self.queued_stop_after_finalizing:
            self.stop_all()

    def discard_demo(self) -> None:
        """Discard current demo buffers and stop current recording."""
        if self.get_state() not in {ControllerState.COLLECTING, ControllerState.PAUSED}:
            self.reject_command('x', self.get_state())
            return
        self.set_state(ControllerState.DISCARDING)
        self.log('discard_started')
        self._sensor_command(self.ft_client, MsgType.DEMO_DISCARD_REQ, 'DEMO_DISCARD_REQ', self.config.ack_timeout_s)
        self._sensor_command(self.xense_client, MsgType.DEMO_DISCARD_REQ, 'DEMO_DISCARD_REQ', self.config.ack_timeout_s)
        try:
            if self.rosbag is not None:
                self.rosbag.stop(timeout_s=self.config.rosbag_timeout_s)
        except Exception as exc:
            self.log('rosbag_stop_failed', error=str(exc))
        self.demo_store = None
        self.rosbag_record_started = False
        self.rosbag_uri = None
        self.set_state(ControllerState.WAIT_START)
        self.log('discard_done')

    def stop_all(self) -> None:
        """Stop sensors, receivers, ROS helpers, and subprocesses."""
        if self.get_state() == ControllerState.STOPPED:
            return
        self.set_state(ControllerState.STOPPING)
        self.log('stopping')
        try:
            if self.rosbag is not None:
                self.rosbag.stop(timeout_s=self.config.rosbag_timeout_s)
        except Exception as exc:
            self.log('rosbag_stop_failed', error=str(exc))
        self._sensor_command(self.ft_client, MsgType.STOP_REQ, 'STOP_REQ', self.config.ack_timeout_s)
        self._sensor_command(self.xense_client, MsgType.STOP_REQ, 'STOP_REQ', self.config.ack_timeout_s)
        self.ft_client.stop()
        self.xense_client.stop()
        if self.zmq_receiver is not None:
            self.zmq_receiver.stop()
        if self.realsense_monitor is not None:
            self.realsense_monitor.stop()
        if self.rosbag is not None:
            try:
                self.rosbag.close()
            except Exception:
                pass
        for process in reversed(list(self.processes.values())):
            process.stop()
        self.set_state(ControllerState.STOPPED)
        self.log('stopped')

    def handle_realsense_fatal(self, payload: dict[str, Any]) -> None:
        """Pause if needed and restart the RealSense camera launch."""
        self.log('realsense_fatal_detected', **payload)
        print(f"[WARN] RealSense fatal output: {payload.get('line')}")
        if self.get_state() == ControllerState.COLLECTING:
            self.pause_demo(reason='realsense_fatal')
        process = self.processes.get('realsense_camera')
        if process is None:
            self.log('realsense_restart_skipped', reason='process_not_found')
            return
        self.realsense_restart_count += 1
        event = {'time_ns': time.time_ns(), **payload}
        self.realsense_restart_events.append(event)
        self.log('realsense_restart_started', count=self.realsense_restart_count)
        process.restart()
        self.reset_realsense_drop_baselines()
        self.log('realsense_restart_done', count=self.realsense_restart_count)

    def _start_processes(self) -> None:
        logs = self.session_dir / 'process_logs'
        root = self.config.repo_root
        self.processes['ft300'] = ManagedProcess(
            'ft300',
            ['conda', 'run', '-n', 'Modbus314', 'python', '-m', 'FT300S.app', '--uds-path', self.config.ft_uds_path, '--shm-name', self.config.ft_shm_name, '--fps', str(self.config.ft_fps)],
            root,
            logs / 'ft300.log',
            on_exit=self._on_process_exit,
        )
        self.processes['xense'] = ManagedProcess(
            'xense',
            ['conda', 'run', '-n', 'Xense310', 'python', '-m', 'XenseTacSensor.app', '--uds-path', self.config.xense_uds_path, '--shm-name', self.config.xense_shm_name, '--fps', str(self.config.xense_fps)],
            root,
            logs / 'xense.log',
            on_exit=self._on_process_exit,
        )
        self.processes['realsense_camera'] = ManagedProcess(
            'realsense_camera',
            bash_cmd('conda deactivate >/dev/null 2>&1 || true; ros2 launch ./RealSense/launch/four_realsense_640x480_30.launch.py'),
            root,
            logs / 'realsense_camera.log',
            fatal_patterns=self.config.fatal_realsense_patterns,
            on_fatal=self._on_process_fatal,
            on_exit=self._on_process_exit,
        )
        self.processes['rosbag_recorder'] = ManagedProcess(
            'rosbag_recorder',
            bash_cmd('conda deactivate >/dev/null 2>&1 || true; ros2 launch ./RealSense/launch/rosbag2_recorder.launch.py'),
            root,
            logs / 'rosbag_recorder.log',
            on_exit=self._on_process_exit,
        )
        for process in self.processes.values():
            process.start()
            self.log('process_started', name=process.name, cmd=process.cmd)

    def _start_receivers(self) -> None:
        self.zmq_receiver = ZmqTelemetryReceiver(self.config.zmq_connect, self._on_zmq_frame, self._on_zmq_error)
        self.zmq_receiver.start()
        self.ft_client.start()
        self.xense_client.start()
        self.realsense_monitor = RealSenseMetadataMonitor(self.config.realsense_metadata_topics, self._on_realsense_metadata)
        self.realsense_monitor.start()
        self.rosbag = RosbagControl()

    def _wait_startup_ready(self) -> None:
        if self.zmq_receiver is None or not self.zmq_receiver.wait_first_frame(self.config.zmq_first_frame_timeout_s):
            raise RuntimeError('ZMQ telemetry did not produce a valid first frame')
        if not self.ft_client.wait_connected(self.config.startup_timeout_s):
            raise RuntimeError('FT300S UDS did not connect')
        if not self.xense_client.wait_connected(self.config.startup_timeout_s):
            raise RuntimeError('Xense UDS did not connect')
        if not self.ft_client.wait_init_ready(self.config.init_timeout_s):
            raise RuntimeError('FT300S did not send INIT_READY')
        if not self.xense_client.wait_init_ready(self.config.init_timeout_s):
            raise RuntimeError('Xense did not send INIT_READY')
        if self.rosbag is not None and not self.rosbag.wait_ready(self.config.startup_timeout_s):
            raise RuntimeError('rosbag2 recorder services did not become ready')

    def _on_uds_event(self, event: UdsEvent) -> None:
        if event.msg_type == MsgType.ERROR:
            self.log('uds_error', sensor=event.client_name, frame_id=event.frame_id, payload=event.payload)
            print(f'[WARN] {event.client_name} ERROR: {event.payload}')
            return
        if event.msg_type != MsgType.FRAME_READY:
            self.log('uds_event', sensor=event.client_name, msg_type=event.msg_type.name, frame_id=event.frame_id, payload=event.payload)
            return

        if event.client_name == 'ft300':
            stamp_ns = _int_or_none(event.payload.get('timestamp_ns'))
            self._observe_drop('ft300', event.frame_id, stamp_ns, self.config.rate.ft300_hz)
            if self.get_state() == ControllerState.COLLECTING and self.demo_store is not None:
                self.demo_store.ft300.append(frame_id=event.frame_id, timestamp_ns=stamp_ns, recv_time_ns=event.recv_time_ns, recv_monotonic_ns=event.recv_monotonic_ns)
        elif event.client_name == 'xense':
            stamp0 = _int_or_none(event.payload.get('timestamp_ns_0'))
            stamp1 = _int_or_none(event.payload.get('timestamp_ns_1'))
            self._observe_drop('xense', event.frame_id, stamp0, self.config.rate.xense_hz)
            if self.get_state() == ControllerState.COLLECTING and self.demo_store is not None:
                self.demo_store.xense.append(frame_id=event.frame_id, timestamp_ns_0=stamp0, timestamp_ns_1=stamp1, recv_time_ns=event.recv_time_ns, recv_monotonic_ns=event.recv_monotonic_ns)

    def _on_zmq_frame(self, frame: TelemetryFrame, recv_time_ns: int, recv_monotonic_ns: int) -> None:
        stamp_ns = int(round(frame.stamp * 1_000_000_000))
        self._observe_drop(f'zmq_source_{frame.source}', frame.seq, stamp_ns, self.config.rate.zmq_hz)
        if self.get_state() == ControllerState.COLLECTING and self.demo_store is not None:
            self.demo_store.zmq.append(source=frame.source, seq=frame.seq, stamp_s=frame.stamp, valid_mask=frame.valid_mask, floats_58=frame.floats_58, gripper_gPO=frame.gripper_gPO, gripper_gCU=frame.gripper_gCU, recv_time_ns=recv_time_ns, recv_monotonic_ns=recv_monotonic_ns)

    def _on_zmq_error(self, message: str) -> None:
        self.log('zmq_error', message=message)
        print(f'[WARN] {message}')

    def _on_realsense_metadata(self, event) -> None:
        assert isinstance(event, RealSenseMetadataEvent)
        stamp_ns = event.frame_timestamp_ns or event.header_stamp_ns
        self._observe_drop(f'realsense:{event.topic}', event.frame_number, stamp_ns, self.config.rate.realsense_hz)
        if self.get_state() == ControllerState.COLLECTING and self.demo_store is not None:
            self.demo_store.realsense.append(topic=event.topic, frame_number=event.frame_number, header_stamp_ns=event.header_stamp_ns, frame_timestamp_ns=event.frame_timestamp_ns, hw_timestamp_ns=event.hw_timestamp_ns, recv_time_ns=event.recv_time_ns, recv_monotonic_ns=event.recv_monotonic_ns)

    def _observe_drop(self, stream: str, key: int | None, stamp_ns: int | None, rate_hz: float) -> None:
        expected = ns_from_hz(rate_hz)
        warning = ns_from_hz(rate_hz, self.config.rate.warning_factor)
        monitor = self.drop_monitors.get(stream)
        if monitor is None:
            monitor = DropMonitor(stream, expected, warning)
            self.drop_monitors[stream] = monitor
        for warning_event in monitor.observe(key, stamp_ns):
            self._emit_drop_warning(warning_event)

    def _emit_drop_warning(self, warning: DropWarning) -> None:
        payload = warning.__dict__
        self.log('drop_warning', **payload)
        print(f"[DROP] {warning.stream} {warning.reason} key={warning.previous_key}->{warning.current_key} interval_ns={warning.interval_ns}")

    def _sensor_command(self, client: UdsClient, msg_type: MsgType, cmd_name: str, timeout_s: float | None) -> dict[str, Any] | None:
        payload = client.send_and_wait_ack(msg_type, cmd_name, timeout_s)
        self.log('sensor_command', sensor=client.name, cmd=cmd_name, ok=payload is not None, payload=payload)
        return payload

    def _sensor_command_no_timeout(self, client: UdsClient, msg_type: MsgType, cmd_name: str) -> dict[str, Any] | None:
        return client.send_and_wait_ack(
            msg_type,
            cmd_name,
            timeout_s=None,
            progress_period_s=self.config.progress_log_period_s,
            on_progress=lambda elapsed: self.log('sensor_flush_waiting', sensor=client.name, cmd=cmd_name, elapsed_s=round(elapsed, 3)),
        )

    def _save_current_demo(self, status: str) -> None:
        if self.demo_store is None:
            return
        npz_paths = self.demo_store.save_all()
        manifest = {
            'status': status,
            'started_ns': self.demo_started_ns,
            'finished_ns': time.time_ns(),
            'rosbag_uri': None if self.rosbag_uri is None else str(self.rosbag_uri),
            'sensor_saved_files': self.sensor_saved_files,
            'npz': npz_paths,
            'drop_monitors': {name: monitor.summary() for name, monitor in self.drop_monitors.items()},
            'realsense_restart_count': self.realsense_restart_count,
            'realsense_restart_events': self.realsense_restart_events,
        }
        manifest_path = self.demo_store.write_manifest(manifest)
        self.log('demo_saved', status=status, manifest=str(manifest_path), npz=npz_paths)

    def reset_drop_baselines(self) -> None:
        """Reset every monitor baseline after pause/resume boundaries."""
        for monitor in self.drop_monitors.values():
            monitor.reset_baseline()

    def reset_realsense_drop_baselines(self) -> None:
        """Reset only RealSense monitor baselines after camera restart."""
        for name, monitor in self.drop_monitors.items():
            if name.startswith('realsense:'):
                monitor.reset_baseline()

    def reject_command(self, command: str, state: ControllerState) -> None:
        """Log and print an invalid command for the current state."""
        self.log('command_rejected', command=command, state=state.name)
        print(f'[WARN] command {command!r} ignored in state {state.name}')

    def log(self, event_type: str, **payload: Any) -> None:
        """Write one controller event."""
        self.logger.event(event_type, state=self.get_state().name, **payload)

    def set_state(self, state: ControllerState) -> None:
        """Set controller state and log transition."""
        with self.state_lock:
            previous = self.state
            self.state = state
        if previous != state:
            self.logger.event('state_transition', previous=previous.name, current=state.name)
            print(f'[state] {previous.name} -> {state.name}')

    def get_state(self) -> ControllerState:
        """Return current controller state."""
        with self.state_lock:
            return self.state

    def _on_process_fatal(self, name: str, line: str) -> None:
        self.commands.put(Command('realsense_fatal', {'process': name, 'line': line, 'time_ns': time.time_ns()}))

    def _on_process_exit(self, name: str, returncode: int) -> None:
        self.log('process_exited', process=name, returncode=returncode)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    """Parse MainController CLI arguments."""
    parser = argparse.ArgumentParser(description='MainController for multi-sensor data collection')
    parser.add_argument('--zmq-connect', default='tcp://127.0.0.1:6000')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--startup-timeout-s', type=float, default=60.0)
    parser.add_argument('--ack-timeout-s', type=float, default=2.0)
    parser.add_argument('--progress-log-period-s', type=float, default=5.0)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RuntimeConfig:
    """Build RuntimeConfig from CLI arguments."""
    output_dir = RuntimeConfig.output_dir if args.output_dir is None else Path(args.output_dir)
    return RuntimeConfig(
        output_dir=output_dir,
        zmq_connect=args.zmq_connect,
        startup_timeout_s=args.startup_timeout_s,
        ack_timeout_s=args.ack_timeout_s,
        progress_log_period_s=args.progress_log_period_s,
    )


def main() -> None:
    """Console entrypoint."""
    args = parse_args()
    controller = MainController(build_config(args))
    controller.run()


if __name__ == '__main__':
    main()
