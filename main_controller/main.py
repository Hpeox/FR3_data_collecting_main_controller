"""MainController entrypoint and orchestration state machine."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .buffers import DemoStore, JsonlLogger
from .config import RuntimeConfig, ns_from_hz, validate_repo_root
from .drop_monitor import DropMonitor, DropWarning
from .processes import ManagedProcess, bash_cmd
from .realsense_metadata import RealSenseMetadataEvent, RealSenseMetadataMonitor
from .rosbag_control import RosbagControl
from .timestamp_alignment import AlignmentOptions, align_demo_timestamps, failure_manifest_entry, update_manifest_alignment
from .uds_client import MAGIC as FT300_MAGIC
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
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = time.strftime('run_%Y%m%d_%H%M%S')
        self.logger = JsonlLogger(
            self.output_dir / f'controller_events_{self.run_id}.jsonl'
        )
        self.commands: queue.Queue[Command] = queue.Queue()
        self.state = ControllerState.BOOT
        self.state_lock = threading.RLock()
        self.demo_store: DemoStore | None = None
        self.demo_started_ns: int | None = None
        self.queued_stop_after_finalizing = False
        self.rosbag_record_started = False
        self.rosbag_uri: Path | None = None
        self.sensor_paths: dict[str, str | None] = {}
        self.realsense_restart_count = 0
        self.realsense_restart_events: list[dict[str, Any]] = []
        self.demo_realsense_restart_count = 0
        self.demo_realsense_restart_events: list[dict[str, Any]] = []
        self.realsense_readiness_manifest: dict[str, Any] | None = None
        self.realsense_postcheck_manifest: dict[str, Any] | None = None
        self.realsense_clock_domain_missing_topics: set[str] = set()

        self.drop_monitors: dict[str, DropMonitor] = {}
        self.demo_drop_monitors: dict[str, DropMonitor] = {}
        self.processes: dict[str, ManagedProcess] = {}
        self.expected_process_exits: set[str] = set()
        self.rosbag: RosbagControl | None = None
        self.realsense_monitor: RealSenseMetadataMonitor | None = None
        self.zmq_receiver: ZmqTelemetryReceiver | None = None
        self.ft_client = UdsClient('ft300', config.ft_uds_path, self._on_uds_event, magic=FT300_MAGIC, on_disconnect=self._on_uds_disconnect)
        self.xense_client = UdsClient('xense', config.xense_uds_path, self._on_uds_event, magic=b'XS', on_disconnect=self._on_uds_disconnect)

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
            state = self.get_state()
            if self._active_demo_needs_abort(state):
                self._abort_active_demo(
                    failure_stage='keyboard_interrupt',
                    failure_reason='KeyboardInterrupt during active demo',
                    abort_context={'signal': 'KeyboardInterrupt'},
                )
            else:
                self.stop_all()
        finally:
            self.logger.close()

    def startup(self) -> None:
        """Start services, connect streams, and initialize sensors."""
        try:
            self.set_state(ControllerState.STARTING_SERVICES)
            self._start_processes()
            self._start_receivers()
            self.set_state(ControllerState.INIT)
            self._wait_startup_ready()
            self.set_state(ControllerState.WAIT_START)
            self.log('ready')
            print('MainController ready.')
        except Exception as exc:
            self.log('startup_failed', error=str(exc), stage=self.get_state().name)
            if self.get_state() != ControllerState.ERROR:
                self.set_state(ControllerState.ERROR)
            self.stop_all()
            raise

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
            elif self._active_demo_needs_abort(state):
                self._abort_active_demo(
                    failure_stage='user_quit',
                    failure_reason='user requested quit during active demo',
                    abort_context={'command': 'q'},
                )
            else:
                self.stop_all()
        elif name == 'realsense_fatal':
            self.handle_realsense_fatal(command.payload or {})
        elif name == 'realsense_metadata_fatal':
            self.handle_realsense_metadata_fatal(command.payload or {})
        elif name == 'zmq_fatal':
            self.handle_zmq_fatal(command.payload or {})
        elif name == 'uds_disconnect':
            self.handle_uds_disconnect(command.payload or {})
        elif name == 'process_exit':
            self.handle_process_exit(command.payload or {})
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
            demo_dir = self._new_demo_dir()
            demo_dir.mkdir(parents=True, exist_ok=True)
            self.demo_store = DemoStore(demo_dir)
            self.demo_started_ns = time.time_ns()
            self.rosbag_record_started = False
            self.rosbag_uri = demo_dir / 'rosbag'
            self.sensor_paths = {}
            self.realsense_readiness_manifest = None
            self.realsense_postcheck_manifest = None
            self.realsense_clock_domain_missing_topics = set()
            self.demo_drop_monitors = {}
            self.demo_realsense_restart_count = 0
            self.demo_realsense_restart_events = []
            self.log('demo_created', demo_dir=str(demo_dir))

        required_sensors = [('ft300', self.ft_client), ('xense', self.xense_client)]
        rollback_target_sensors: list[tuple[str, UdsClient]] = [] if new_demo else list(required_sensors)
        acked_start_sensors: list[tuple[str, UdsClient]] = []
        if not self._sensor_command(self.ft_client, MsgType.START_REQ, 'START_REQ', self.config.ack_timeout_s):
            self._fail_start_resume_transaction(
                new_demo=new_demo,
                failure_stage='ft300_start',
                failure_reason='FT300S START_REQ failed',
                acked_start_sensors=acked_start_sensors,
                rollback_target_sensors=rollback_target_sensors or acked_start_sensors,
            )
            return
        acked_start_sensors.append(('ft300', self.ft_client))

        if not self._sensor_command(self.xense_client, MsgType.START_REQ, 'START_REQ', self.config.ack_timeout_s):
            self._fail_start_resume_transaction(
                new_demo=new_demo,
                failure_stage='xense_start',
                failure_reason='Xense START_REQ failed',
                acked_start_sensors=acked_start_sensors,
                rollback_target_sensors=rollback_target_sensors or acked_start_sensors,
            )
            return
        acked_start_sensors.append(('xense', self.xense_client))
        if new_demo:
            rollback_target_sensors = list(acked_start_sensors)

        rosbag_action: str | None = None
        try:
            if self.rosbag is not None and self.rosbag_uri is not None:
                readiness = self.rosbag.check_image_readiness(
                    self.config.realsense_image_requirements,
                    timeout_s=self.config.realsense_image_ready_timeout_s,
                    mode=self.config.realsense_capture_mode,
                )
                self.realsense_readiness_manifest = readiness.to_manifest()
                self.log('realsense_image_readiness', **self.realsense_readiness_manifest)
                if not readiness.ok:
                    self._fail_start_resume_transaction(
                        new_demo=new_demo,
                        failure_stage='realsense_image_readiness',
                        failure_reason='required RealSense image topics are not ready',
                        acked_start_sensors=acked_start_sensors,
                        rollback_target_sensors=rollback_target_sensors,
                        rosbag_state={'image_readiness': self.realsense_readiness_manifest},
                    )
                    return
                if not self.rosbag_record_started:
                    rosbag_action = 'record'
                    self.rosbag.record(self.rosbag_uri, timeout_s=self.config.rosbag_timeout_s)
                    self.rosbag_record_started = True
                rosbag_action = 'resume'
                self.rosbag.resume(timeout_s=self.config.rosbag_timeout_s)
        except Exception as exc:
            self.log('rosbag_start_failed', action=rosbag_action, error=str(exc))
            print(f'[ERROR] rosbag start/resume failed: {exc}')
            self._fail_start_resume_transaction(
                new_demo=new_demo,
                failure_stage=f'rosbag_{rosbag_action or "start"}',
                failure_reason=str(exc),
                acked_start_sensors=acked_start_sensors,
                rollback_target_sensors=rollback_target_sensors,
                rosbag_state={'failed_action': rosbag_action},
            )
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
        ft_result = self._sensor_command_result(self.ft_client, MsgType.PAUSE_REQ, 'PAUSE_REQ', self.config.ack_timeout_s)
        xense_result = self._sensor_command_result(self.xense_client, MsgType.PAUSE_REQ, 'PAUSE_REQ', self.config.ack_timeout_s)
        rosbag_pause_result: dict[str, Any] = {'ok': True, 'action': 'pause'}
        try:
            if self.rosbag is not None:
                self.rosbag.pause(timeout_s=self.config.rosbag_timeout_s)
        except Exception as exc:
            rosbag_pause_result = {'ok': False, 'action': 'pause', 'error': str(exc)}
            self.log('rosbag_pause_failed', error=str(exc))
        if not rosbag_pause_result['ok']:
            self._handle_command_transaction_failure(
                failure_stage='rosbag_pause',
                failure_reason='rosbag pause failed',
                command_results={'ft300': ft_result, 'xense': xense_result, 'rosbag_pause': rosbag_pause_result},
                clear_demo=False,
                stop_system=True,
            )
            return False
        if not ft_result['ok'] or not xense_result['ok']:
            self._handle_command_transaction_failure(
                failure_stage='pause_command',
                failure_reason='required sensor PAUSE_REQ failed',
                command_results={'ft300': ft_result, 'xense': xense_result, 'rosbag_pause': rosbag_pause_result},
                clear_demo=False,
                stop_system=True,
            )
            return False
        self.reset_drop_baselines()
        self.log('pause_done', reason=reason, ft300=ft_result['ok'], xense=xense_result['ok'])
        return True

    def finish_demo(self) -> None:
        """Finish the current demo and save all controller buffers."""
        if self.get_state() not in {ControllerState.COLLECTING, ControllerState.PAUSED}:
            self.reject_command('d', self.get_state())
            return
        self.set_state(ControllerState.FINALIZING)
        self.log('finalizing_started')

        with ThreadPoolExecutor(max_workers=3, thread_name_prefix='DemoFinalize') as executor:
            ft_future = executor.submit(
                self._sensor_command_result_with_progress,
                self.ft_client,
                MsgType.DEMO_DONE_REQ,
                'DEMO_DONE_REQ',
                self.config.sensor_flush_timeout_s,
            )
            xense_future = executor.submit(
                self._sensor_command_result_with_progress,
                self.xense_client,
                MsgType.DEMO_DONE_REQ,
                'DEMO_DONE_REQ',
                self.config.sensor_flush_timeout_s,
            )
            rosbag_stop_future = executor.submit(self._rosbag_stop_result)
            ft_result = ft_future.result()
            xense_result = xense_future.result()
            rosbag_stop_result = rosbag_stop_future.result()
        self.sensor_paths = {
            'ft300': self._sensor_path_from_payload(ft_result['payload']),
            'xense': self._sensor_path_from_payload(xense_result['payload']),
        }

        command_failed = not ft_result['ok'] or not xense_result['ok'] or not rosbag_stop_result['ok']
        postcheck_failed = False
        if command_failed:
            self.realsense_postcheck_manifest = None
            status = 'failed'
        else:
            self.realsense_postcheck_manifest = self._run_realsense_rosbag_postcheck()
            status = 'done'
        if not command_failed and self.realsense_postcheck_manifest is not None and not self.realsense_postcheck_manifest.get('ok', False):
            postcheck_failed = True
            status = 'failed'
        extra = None
        if command_failed:
            if not rosbag_stop_result['ok']:
                failure_stage = 'rosbag_stop'
                failure_reason = 'rosbag stop failed'
            else:
                failure_stage = 'finish_command'
                failure_reason = 'required sensor DEMO_DONE_REQ failed'
            extra = {
                'failure_stage': failure_stage,
                'failure_reason': failure_reason,
                'command_results': {'ft300': ft_result, 'xense': xense_result, 'rosbag_stop': rosbag_stop_result},
            }
        elif postcheck_failed:
            extra = {
                'failure_stage': 'realsense_rosbag_postcheck',
                'failure_reason': self._realsense_postcheck_failure_reason(),
                'command_results': {'ft300': ft_result, 'xense': xense_result, 'rosbag_stop': rosbag_stop_result},
            }
        manifest_path = self._save_current_demo(status=status, extra=extra)
        if status == 'done' and manifest_path is not None:
            self._run_timestamp_alignment(manifest_path)
        if command_failed or postcheck_failed:
            self.demo_store = None
            self.demo_started_ns = None
            self.rosbag_record_started = False
            self.rosbag_uri = None
            self.set_state(ControllerState.ERROR)
            self.stop_all()
            return
        self.demo_store = None
        self.rosbag_record_started = False
        self.rosbag_uri = None
        self.set_state(ControllerState.WAIT_START)
        self.log('finalizing_done')
        if self.queued_stop_after_finalizing:
            self.stop_all()

    def _rosbag_stop_result(self) -> dict[str, Any]:
        """Stop rosbag recording and return a manifest-friendly result."""
        try:
            if self.rosbag is not None:
                self.rosbag.stop(timeout_s=self.config.rosbag_timeout_s)
            return {'ok': True, 'action': 'stop'}
        except Exception as exc:
            self.log('rosbag_stop_failed', error=str(exc))
            return {'ok': False, 'action': 'stop', 'error': str(exc)}

    def discard_demo(self) -> None:
        """Discard current demo buffers and stop current recording."""
        if self.get_state() not in {ControllerState.COLLECTING, ControllerState.PAUSED}:
            self.reject_command('x', self.get_state())
            return
        self.set_state(ControllerState.DISCARDING)
        self.log('discard_started')
        ft_result = self._sensor_command_result(self.ft_client, MsgType.DEMO_DISCARD_REQ, 'DEMO_DISCARD_REQ', self.config.ack_timeout_s)
        xense_result = self._sensor_command_result(self.xense_client, MsgType.DEMO_DISCARD_REQ, 'DEMO_DISCARD_REQ', self.config.ack_timeout_s)
        rosbag_stop_result: dict[str, Any] = {'ok': True}
        try:
            if self.rosbag is not None:
                self.rosbag.stop(timeout_s=self.config.rosbag_timeout_s)
        except Exception as exc:
            rosbag_stop_result = {'ok': False, 'error': str(exc)}
            self.log('rosbag_stop_failed', error=str(exc))
        if not ft_result['ok'] or not xense_result['ok'] or not rosbag_stop_result['ok']:
            self._handle_command_transaction_failure(
                failure_stage='discard_command',
                failure_reason='user discard transaction failed',
                command_results={'ft300': ft_result, 'xense': xense_result, 'rosbag_stop': rosbag_stop_result},
                clear_demo=True,
                stop_system=True,
            )
            return
        self._write_current_demo_manifest(
            status='discarded',
            npz_paths={},
            extra={'discard_reason': 'user'},
        )
        self.demo_store = None
        self.rosbag_record_started = False
        self.rosbag_uri = None
        self.set_state(ControllerState.WAIT_START)
        self.log('discard_done')

    def stop_all(self) -> None:
        """Stop sensors, receivers, ROS helpers, and subprocesses."""
        self._stop_runtime_resources(stop_rosbag=True, send_sensor_stop=True)

    def _stop_runtime_resources(self, *, stop_rosbag: bool, send_sensor_stop: bool) -> None:
        """Stop runtime resources without writing demo manifests."""
        if self.get_state() == ControllerState.STOPPED:
            return
        self.set_state(ControllerState.STOPPING)
        self.log('stopping')
        try:
            if stop_rosbag and self.rosbag is not None:
                self.rosbag.stop(timeout_s=self.config.rosbag_timeout_s)
        except Exception as exc:
            self.log('rosbag_stop_failed', error=str(exc))
        if send_sensor_stop:
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
        self.expected_process_exits.update(self.processes)
        for process in reversed(list(self.processes.values())):
            process.stop()
        self.set_state(ControllerState.STOPPED)
        self.log('stopped')

    def _abort_active_demo(
        self,
        *,
        failure_stage: str,
        failure_reason: str,
        abort_context: dict[str, Any] | None = None,
    ) -> None:
        """Stop an active demo, save partial controller data, and stop the system."""
        self.log('active_demo_abort_started', failure_stage=failure_stage, failure_reason=failure_reason)
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix='DemoAbort') as executor:
            ft_future = executor.submit(
                self._sensor_command_result,
                self.ft_client,
                MsgType.STOP_REQ,
                'STOP_REQ',
                self.config.ack_timeout_s,
            )
            xense_future = executor.submit(
                self._sensor_command_result,
                self.xense_client,
                MsgType.STOP_REQ,
                'STOP_REQ',
                self.config.ack_timeout_s,
            )
            rosbag_stop_future = executor.submit(self._rosbag_stop_result)
            ft_result = ft_future.result()
            xense_result = xense_future.result()
            rosbag_stop_result = rosbag_stop_future.result()

        self.sensor_paths = {
            'ft300': self._sensor_path_from_payload(ft_result['payload']),
            'xense': self._sensor_path_from_payload(xense_result['payload']),
        }
        extra = {
            'failure_stage': failure_stage,
            'failure_reason': failure_reason,
            'command_results': {'ft300': ft_result, 'xense': xense_result, 'rosbag_stop': rosbag_stop_result},
            'abort_context': abort_context or {},
        }
        self._save_current_demo(status='failed', extra=extra)
        self.log('active_demo_abort_done', failure_stage=failure_stage, failure_reason=failure_reason)
        self.demo_store = None
        self.demo_started_ns = None
        self.rosbag_record_started = False
        self.rosbag_uri = None
        self._stop_runtime_resources(stop_rosbag=False, send_sensor_stop=False)

    def handle_realsense_fatal(self, payload: dict[str, Any]) -> None:
        """Pause if needed and restart the RealSense camera launch."""
        self.log('realsense_fatal_detected', **payload)
        print(f"[WARN] RealSense fatal output: {payload.get('line')}")
        if self.get_state() == ControllerState.COLLECTING:
            if not self.pause_demo(reason='realsense_fatal'):
                self.log('realsense_restart_skipped', reason='auto_pause_failed')
                return
        if self.get_state() in {ControllerState.ERROR, ControllerState.STOPPING, ControllerState.STOPPED}:
            self.log('realsense_restart_skipped', reason='controller_stopping_or_error')
            return
        if self.get_state() not in {ControllerState.PAUSED, ControllerState.WAIT_START}:
            self.log('realsense_restart_skipped', reason='invalid_state_for_restart')
            return
        process = self.processes.get('realsense_camera')
        if process is None:
            self.log('realsense_restart_skipped', reason='process_not_found')
            return
        self.realsense_restart_count += 1
        event = {'time_ns': time.time_ns(), **payload}
        self.realsense_restart_events.append(event)
        if self.demo_store is not None:
            self.demo_realsense_restart_count += 1
            self.demo_realsense_restart_events.append(event)
        self.log('realsense_restart_started', count=self.realsense_restart_count)
        self.expected_process_exits.add('realsense_camera')
        process.restart()
        self.reset_realsense_drop_baselines()
        self.log('realsense_restart_done', count=self.realsense_restart_count)

    def handle_realsense_metadata_fatal(self, payload: dict[str, Any]) -> None:
        """Treat metadata receiver termination as an unrecoverable error."""
        self.log('realsense_metadata_fatal_detected', **payload)
        print(f"[ERROR] RealSense metadata monitor fatal: {payload.get('message')}")
        self._handle_unrecoverable_fatal(
            failure_stage='realsense_metadata_fatal',
            failure_reason=str(payload.get('message') or 'RealSense metadata monitor fatal'),
            abort_context=payload,
        )

    def handle_zmq_fatal(self, payload: dict[str, Any]) -> None:
        """Treat receiver-loop termination as an unrecoverable controller error."""
        self.log('zmq_fatal_detected', **payload)
        print(f"[ERROR] ZMQ receiver fatal: {payload.get('message')}")
        self._handle_unrecoverable_fatal(
            failure_stage='zmq_fatal',
            failure_reason=str(payload.get('message') or 'ZMQ receiver fatal'),
            abort_context=payload,
        )

    def handle_uds_disconnect(self, payload: dict[str, Any]) -> None:
        """Treat required sensor UDS peer disconnect as a fatal runtime error."""
        sensor = str(payload.get('sensor') or 'unknown')
        if sensor not in {'ft300', 'xense'}:
            self.log('uds_disconnect_ignored', **payload)
            return
        self.log('uds_disconnect_detected', **payload)
        self._handle_unrecoverable_fatal(
            failure_stage=f'{sensor}_uds_disconnect',
            failure_reason=f'{sensor} UDS peer disconnected',
            abort_context=payload,
        )

    def handle_process_exit(self, payload: dict[str, Any]) -> None:
        """Treat unexpected required subprocess exit as a fatal runtime error."""
        name = str(payload.get('process') or 'unknown')
        if name not in {'ft300', 'xense', 'rosbag_recorder', 'realsense_camera'}:
            self.log('process_exit_ignored', **payload)
            return
        state = self.get_state()
        if state not in {ControllerState.WAIT_START, ControllerState.COLLECTING, ControllerState.PAUSED}:
            self.log('process_exit_ignored', reason='inactive_state', **payload)
            return
        self._handle_unrecoverable_fatal(
            failure_stage=f'{name}_process_exit',
            failure_reason=f'{name} process exited unexpectedly',
            abort_context=payload,
        )

    def _handle_unrecoverable_fatal(
        self,
        *,
        failure_stage: str,
        failure_reason: str,
        abort_context: dict[str, Any],
    ) -> None:
        active_demo = self._active_demo_needs_abort(self.get_state())
        if self.get_state() != ControllerState.ERROR:
            self.set_state(ControllerState.ERROR)
        if active_demo:
            self._abort_active_demo(
                failure_stage=failure_stage,
                failure_reason=failure_reason,
                abort_context=abort_context,
            )
        else:
            self.stop_all()

    def _start_processes(self) -> None:
        logs = self.output_dir / 'process_logs' / self.run_id
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
        started: list[ManagedProcess] = []
        try:
            for process in self.processes.values():
                process.start()
                started.append(process)
                self.log('process_started', name=process.name, cmd=process.cmd)
        except Exception as exc:
            self.log('process_start_failed', name=process.name, error=str(exc))
            for process in reversed(started):
                process.stop()
            raise

    def _start_receivers(self) -> None:
        self.zmq_receiver = ZmqTelemetryReceiver(
            self.config.zmq_connect,
            self._on_zmq_frame,
            self._on_zmq_error,
            self._on_zmq_fatal,
        )
        self.zmq_receiver.start()
        self.ft_client.start()
        self.xense_client.start()
        self.realsense_monitor = RealSenseMetadataMonitor(
            self.config.realsense_metadata_topics,
            self._on_realsense_metadata,
            self._on_realsense_metadata_fatal,
        )
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
        if self.realsense_monitor is None or not self.realsense_monitor.wait_ready(self.config.startup_timeout_s):
            error = None if self.realsense_monitor is None else self.realsense_monitor.fatal_error()
            if error is None:
                error = 'RealSense metadata monitor did not become ready'
            raise RuntimeError(error)
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

    def _on_zmq_fatal(self, message: str) -> None:
        self.commands.put(Command('zmq_fatal', {'message': message, 'time_ns': time.time_ns()}))

    def _on_realsense_metadata_fatal(self, message: str) -> None:
        self.commands.put(Command('realsense_metadata_fatal', {'message': message, 'time_ns': time.time_ns()}))

    def _on_uds_disconnect(self, name: str, pending_cmds: list[str]) -> None:
        self.commands.put(Command('uds_disconnect', {'sensor': name, 'pending_cmds': pending_cmds, 'time_ns': time.time_ns()}))

    def _on_realsense_metadata(self, event) -> None:
        assert isinstance(event, RealSenseMetadataEvent)
        if event.clock_domain is None and event.topic not in self.realsense_clock_domain_missing_topics:
            self.realsense_clock_domain_missing_topics.add(event.topic)
            self.log('realsense_clock_domain_missing', topic=event.topic, frame_number=event.frame_number)
        stamp_ns = event.frame_timestamp_ns or event.header_stamp_ns
        self._observe_drop(f'realsense:{event.topic}', event.frame_number, stamp_ns, self.config.rate.realsense_hz)
        if self.get_state() == ControllerState.COLLECTING and self.demo_store is not None:
            self.demo_store.realsense.append(topic=event.topic, frame_number=event.frame_number, header_stamp_ns=event.header_stamp_ns, frame_timestamp_ns=event.frame_timestamp_ns, hw_timestamp_ns=event.hw_timestamp_ns, clock_domain=event.clock_domain, recv_time_ns=event.recv_time_ns, recv_monotonic_ns=event.recv_monotonic_ns)

    def _observe_drop(self, stream: str, key: int | None, stamp_ns: int | None, rate_hz: float) -> None:
        expected = ns_from_hz(rate_hz)
        warning = ns_from_hz(rate_hz, self.config.rate.warning_factor)
        monitor = self.drop_monitors.get(stream)
        if monitor is None:
            monitor = DropMonitor(stream, expected, warning)
            self.drop_monitors[stream] = monitor
        for warning_event in monitor.observe(key, stamp_ns):
            self._emit_drop_warning(warning_event)
        if self.get_state() == ControllerState.COLLECTING and self.demo_store is not None:
            demo_monitor = self.demo_drop_monitors.get(stream)
            if demo_monitor is None:
                demo_monitor = DropMonitor(stream, expected, warning)
                self.demo_drop_monitors[stream] = demo_monitor
            demo_monitor.observe(key, stamp_ns)

    def _emit_drop_warning(self, warning: DropWarning) -> None:
        payload = warning.__dict__
        self.log('drop_warning', **payload)
        print(f"[DROP] {warning.stream} {warning.reason} key={warning.previous_key}->{warning.current_key} interval_ns={warning.interval_ns}")

    def _sensor_command(self, client: UdsClient, msg_type: MsgType, cmd_name: str, timeout_s: float | None) -> dict[str, Any] | None:
        payload = client.send_and_wait_ack(msg_type, cmd_name, timeout_s)
        self.log('sensor_command', sensor=client.name, cmd=cmd_name, ok=payload is not None, payload=payload)
        return payload

    def _sensor_command_result(self, client: UdsClient, msg_type: MsgType, cmd_name: str, timeout_s: float | None) -> dict[str, Any]:
        payload = self._sensor_command(client, msg_type, cmd_name, timeout_s)
        return {
            'sensor': client.name,
            'cmd': cmd_name,
            'ok': payload is not None,
            'payload': payload,
            'error': None if payload is not None else client.last_error_for(cmd_name),
        }

    def _sensor_command_with_progress(
        self,
        client: UdsClient,
        msg_type: MsgType,
        cmd_name: str,
        timeout_s: float | None,
    ) -> dict[str, Any] | None:
        return client.send_and_wait_ack(
            msg_type,
            cmd_name,
            timeout_s=timeout_s,
            progress_period_s=self.config.progress_log_period_s,
            on_progress=lambda elapsed: self.log('sensor_flush_waiting', sensor=client.name, cmd=cmd_name, elapsed_s=round(elapsed, 3)),
        )

    def _sensor_command_result_with_progress(
        self,
        client: UdsClient,
        msg_type: MsgType,
        cmd_name: str,
        timeout_s: float | None,
    ) -> dict[str, Any]:
        payload = self._sensor_command_with_progress(client, msg_type, cmd_name, timeout_s)
        self.log('sensor_command', sensor=client.name, cmd=cmd_name, ok=payload is not None, payload=payload)
        return {
            'sensor': client.name,
            'cmd': cmd_name,
            'ok': payload is not None,
            'payload': payload,
            'error': None if payload is not None else client.last_error_for(cmd_name),
        }

    def _handle_command_transaction_failure(
        self,
        *,
        failure_stage: str,
        failure_reason: str,
        command_results: dict[str, Any],
        clear_demo: bool,
        stop_system: bool,
    ) -> None:
        self._write_current_demo_manifest(
            status='failed',
            npz_paths={},
            extra={
                'failure_stage': failure_stage,
                'failure_reason': failure_reason,
                'command_results': command_results,
            },
        )
        self.log('command_transaction_failed', failure_stage=failure_stage, failure_reason=failure_reason, command_results=command_results)
        if clear_demo:
            self.demo_store = None
            self.demo_started_ns = None
            self.rosbag_record_started = False
            self.rosbag_uri = None
        if stop_system:
            if self.get_state() != ControllerState.ERROR:
                self.set_state(ControllerState.ERROR)
            self.stop_all()

    def _fail_start_resume_transaction(
        self,
        *,
        new_demo: bool,
        failure_stage: str,
        failure_reason: str,
        acked_start_sensors: list[tuple[str, UdsClient]],
        rollback_target_sensors: list[tuple[str, UdsClient]],
        rosbag_state: dict[str, Any] | None = None,
    ) -> None:
        rollback_results: dict[str, dict[str, Any]] = {}
        rollback_unconfirmed_sensors: list[str] = []
        for sensor, client in rollback_target_sensors:
            result = self._sensor_command_result(client, MsgType.DEMO_DISCARD_REQ, 'DEMO_DISCARD_REQ', self.config.ack_timeout_s)
            rollback_results[sensor] = result
            if not result['ok']:
                rollback_unconfirmed_sensors.append(sensor)

        rosbag_stop: dict[str, Any] | None = None
        if self.rosbag is not None and self.rosbag_record_started:
            try:
                self.rosbag.stop(timeout_s=self.config.rosbag_timeout_s)
                rosbag_stop = {'ok': True}
            except Exception as exc:
                rosbag_stop = {'ok': False, 'error': str(exc)}
                self.log('rosbag_stop_failed', error=str(exc))

        cleanup_unconfirmed = bool(rollback_unconfirmed_sensors) or (rosbag_stop is not None and not rosbag_stop.get('ok', False))
        failed_details = {
            'failure_stage': failure_stage,
            'failure_reason': failure_reason,
            'new_demo': new_demo,
            'acked_start_sensors': [sensor for sensor, _client in acked_start_sensors],
            'rollback_target_sensors': [sensor for sensor, _client in rollback_target_sensors],
            'rollback_unconfirmed_sensors': rollback_unconfirmed_sensors,
            'rollback_action': 'DEMO_DISCARD_REQ',
            'rollback_results': rollback_results,
            'rosbag_record_resume': {
                'record_started': self.rosbag_record_started,
                'uri': (
                    None
                    if self.rosbag_uri is None
                    else self._demo_relative_str(self.rosbag_uri)
                ),
                'stop': rosbag_stop,
                **(rosbag_state or {}),
            },
        }
        self._write_current_demo_manifest(status='failed', npz_paths={}, extra=failed_details)
        self.log('start_resume_transaction_failed', **failed_details)
        self.demo_store = None
        self.demo_started_ns = None
        self.rosbag_record_started = False
        self.rosbag_uri = None
        self.sensor_paths = {}
        self.realsense_readiness_manifest = None
        self.realsense_postcheck_manifest = None
        if cleanup_unconfirmed:
            self.set_state(ControllerState.ERROR)
            self.stop_all()
            return
        self.set_state(ControllerState.WAIT_START)

    def _save_current_demo(self, status: str, extra: dict[str, Any] | None = None) -> Path | None:
        if self.demo_store is None:
            return None
        npz_paths = self.demo_store.save_all()
        return self._write_current_demo_manifest(status=status, npz_paths=npz_paths, extra=extra)

    def _write_current_demo_manifest(
        self,
        status: str,
        npz_paths: dict[str, str],
        extra: dict[str, Any] | None = None,
    ) -> Path | None:
        if self.demo_store is None:
            return None
        manifest = {
            'status': status,
            'started_ns': self.demo_started_ns,
            'finished_ns': time.time_ns(),
            'run_id': self.run_id,
            'rosbag_uri': (
                None
                if self.rosbag_uri is None
                else self._demo_relative_str(self.rosbag_uri)
            ),
            'sensor_paths': self.sensor_paths,
            'npz': npz_paths,
            'frame_counts': self.demo_store.frame_counts(),
            'drop_monitors': {name: monitor.summary() for name, monitor in self.demo_drop_monitors.items()},
            'realsense_restart_count': self.demo_realsense_restart_count,
            'realsense_restart_events': self.demo_realsense_restart_events,
            'run_realsense_restart_count': self.realsense_restart_count,
            'run_realsense_restart_events': self.realsense_restart_events,
            'realsense_image_readiness': self.realsense_readiness_manifest,
            'realsense_rosbag_postcheck': self.realsense_postcheck_manifest,
        }
        if extra is not None:
            manifest.update(extra)
        manifest_path = self.demo_store.write_manifest(manifest)
        self.log('demo_saved', status=status, manifest=str(manifest_path), npz=npz_paths)
        return manifest_path

    def _run_timestamp_alignment(self, manifest_path: Path) -> None:
        """Run automatic timestamp alignment and update the demo manifest."""
        if self.demo_store is None:
            return
        started_ns = time.time_ns()
        demo_dir = self.demo_store.demo_dir
        options = AlignmentOptions(
            repo_root=self.config.repo_root,
            base='auto',
            alignment_base_source=self.config.alignment_base_source,
            mode=self.config.alignment_mode,
            hz=self.config.alignment_hz,
            start_trim_s=self.config.alignment_start_trim_s,
        )
        try:
            result = align_demo_timestamps(demo_dir, options)
            entry = result.to_manifest_entry(started_ns=started_ns, finished_ns=time.time_ns())
            update_manifest_alignment(manifest_path, entry)
            self.log('timestamp_alignment_done', **entry)
        except Exception as exc:
            entry = failure_manifest_entry(started_ns, exc)
            update_manifest_alignment(manifest_path, entry)
            self.log('timestamp_alignment_failed', **entry)
            print(f'[WARN] timestamp alignment failed: {exc}')

    def _run_realsense_rosbag_postcheck(self) -> dict[str, Any] | None:
        if self.rosbag is None or self.rosbag_uri is None:
            return None
        try:
            result = self.rosbag.validate_recorded_images(
                self.rosbag_uri,
                self.config.realsense_image_requirements,
                count_skew_limit=self.config.realsense_rosbag_count_skew_limit,
                mode=self.config.realsense_capture_mode,
            )
            manifest = result.to_manifest()
            manifest['rosbag_uri'] = self._demo_relative_str(self.rosbag_uri)
            self.log('realsense_rosbag_postcheck', **manifest)
            return manifest
        except Exception as exc:
            manifest = {
                'ok': False,
                'mode': self.config.realsense_capture_mode,
                'rosbag_uri': self._demo_relative_str(self.rosbag_uri),
                'required_topics': [requirement.topic for requirement in self.config.realsense_image_requirements],
                'error': str(exc),
            }
            self.log('realsense_rosbag_postcheck_failed', **manifest)
            return manifest

    def _realsense_postcheck_failure_reason(self) -> str:
        manifest = self.realsense_postcheck_manifest or {}
        if manifest.get('error'):
            return str(manifest['error'])
        missing_topics = manifest.get('missing_topics') or []
        if missing_topics:
            return f"missing required RealSense image topics: {', '.join(str(topic) for topic in missing_topics)}"
        return 'RealSense rosbag post-check failed'

    def _new_demo_dir(self) -> Path:
        """Return a unique demo directory path for rapid repeated captures."""
        base = time.strftime('demo_%Y%m%d_%H%M%S')
        demos_dir = self.output_dir / 'demos'
        candidate = demos_dir / base
        if not candidate.exists():
            return candidate
        suffix = 1
        while True:
            candidate = demos_dir / f'{base}_{suffix:03d}'
            if not candidate.exists():
                return candidate
            suffix += 1

    def _demo_relative_str(self, path: Path) -> str:
        """Return a path relative to the active demo directory."""
        if self.demo_store is None:
            return path.as_posix()
        return path.resolve().relative_to(self.demo_store.demo_dir.resolve()).as_posix()

    @staticmethod
    def _sensor_path_from_payload(payload: dict[str, Any] | None) -> str | None:
        """Convert a sensor ACK payload into a repo-relative runtime_frames path."""
        if payload is None:
            return None
        saved_file = payload.get('saved_file')
        if saved_file is None:
            return None
        return (Path('runtime_frames') / Path(saved_file).name).as_posix()

    def reset_drop_baselines(self) -> None:
        """Reset every monitor baseline after pause/resume boundaries."""
        for monitor in self.drop_monitors.values():
            monitor.reset_baseline()
        for monitor in self.demo_drop_monitors.values():
            monitor.reset_baseline()

    def reset_realsense_drop_baselines(self) -> None:
        """Reset only RealSense monitor baselines after camera restart."""
        for name, monitor in self.drop_monitors.items():
            if name.startswith('realsense:'):
                monitor.reset_baseline()
        for name, monitor in self.demo_drop_monitors.items():
            if name.startswith('realsense:'):
                monitor.reset_baseline()

    def _active_demo_needs_abort(self, state: ControllerState) -> bool:
        return self.demo_store is not None and state in {ControllerState.COLLECTING, ControllerState.PAUSED}

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
        if name in self.expected_process_exits:
            self.expected_process_exits.discard(name)
            self.log('process_exited_expected', process=name, returncode=returncode)
            return
        self.log('process_exited', process=name, returncode=returncode)
        self.commands.put(Command('process_exit', {'process': name, 'returncode': returncode, 'time_ns': time.time_ns()}))


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _float_or_none(value: str) -> float | None:
    lowered = value.strip().lower()
    if lowered in {'none', 'unbounded'}:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    """Parse MainController CLI arguments."""
    parser = argparse.ArgumentParser(description='MainController for multi-sensor data collection')
    parser.add_argument('--repo-root', default=None)
    parser.add_argument('--zmq-connect', default='tcp://127.0.0.1:6000')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--startup-timeout-s', type=float, default=60.0)
    parser.add_argument('--ack-timeout-s', type=float, default=2.0)
    parser.add_argument('--sensor-flush-timeout-s', type=_float_or_none, default=300.0)
    parser.add_argument('--progress-log-period-s', type=float, default=5.0)
    parser.add_argument('--alignment-base-source', choices=['realsense', 'xense'], default='realsense')
    parser.add_argument('--alignment-mode', choices=['causal', 'nearest'], default='causal')
    parser.add_argument('--alignment-hz', type=float, default=30.0)
    parser.add_argument('--alignment-start-trim-s', type=float, default=2.0)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RuntimeConfig:
    """Build RuntimeConfig from CLI arguments."""
    repo_root = validate_repo_root(Path(args.repo_root)) if args.repo_root is not None else None
    output_dir = None if args.output_dir is None else Path(args.output_dir)
    kwargs: dict[str, Any] = {}
    if repo_root is not None:
        kwargs['repo_root'] = repo_root
    return RuntimeConfig(
        output_dir=output_dir,
        zmq_connect=args.zmq_connect,
        startup_timeout_s=args.startup_timeout_s,
        ack_timeout_s=args.ack_timeout_s,
        sensor_flush_timeout_s=args.sensor_flush_timeout_s,
        progress_log_period_s=args.progress_log_period_s,
        alignment_base_source=args.alignment_base_source,
        alignment_mode=args.alignment_mode,
        alignment_hz=args.alignment_hz,
        alignment_start_trim_s=args.alignment_start_trim_s,
        **kwargs,
    )


def main() -> None:
    """Console entrypoint."""
    args = parse_args()
    controller = MainController(build_config(args))
    controller.run()


if __name__ == '__main__':
    main()
