"""Persistent FR3 inference session and rollout controller."""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .aligned_health import AlignedHealthReader
from .buffers import JsonlLogger, TableBuffer
from .config import XENSE_SDK_CONDA_ENVS, default_repo_root, validate_repo_root
from .inference_config import InferenceConfig
from .inference_protocol import ControlledClient, Status
from .processes import ManagedProcess, bash_cmd
from .rosbag_control import RosbagControl
from .uds_client import MsgType, UdsClient, UdsEvent
from .zmq_telemetry import TelemetryFrame, ZmqTelemetryReceiver


SOURCE_ROBOT = 2
ROBOT_TELEMETRY_FLAG_RESETTING = 1 << 0
ROBOT_TELEMETRY_FLAG_JUMP_HOLD = 1 << 1


class InferenceState(Enum):
    """MainController-owned inference lifecycle states."""

    CREATED = auto()
    STARTING_SERVICES = auto()
    WAIT_START = auto()
    INITIALIZING = auto()
    READY = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING_ROLLOUT = auto()
    ABORTING = auto()
    SHUTTING_DOWN = auto()
    FAIL_STOPPING = auto()
    STOPPED = auto()


@dataclass(frozen=True)
class InferenceCommand:
    """One action admitted against the state observed at receipt time."""

    name: str
    payload: dict[str, Any] | None = None


@dataclass
class InferenceRolloutStore:
    """Raw MainController-owned buffers for one inference rollout."""

    rollout_dir: Path

    def __post_init__(self) -> None:
        self.ft300 = TableBuffer(('frame_id', 'timestamp_ns', 'recv_time_ns', 'recv_monotonic_ns'))
        self.xense = TableBuffer(
            ('frame_id', 'timestamp_ns_0', 'timestamp_ns_1', 'recv_time_ns', 'recv_monotonic_ns')
        )
        self.zmq = TableBuffer(
            (
                'source', 'flags', 'seq', 'stamp_s', 'valid_mask', 'floats_58',
                'gripper_gPO', 'gripper_gCU', 'recv_time_ns', 'recv_monotonic_ns',
            )
        )

    def save(self) -> dict[str, str]:
        """Persist raw low-dimensional buffers."""
        paths = {
            'ft300': self.ft300.save_npz(self.rollout_dir / 'ft300_timestamps.npz'),
            'xense': self.xense.save_npz(self.rollout_dir / 'xense_timestamps.npz'),
            'zmq': self.zmq.save_npz(self.rollout_dir / 'zmq_telemetry.npz'),
        }
        return {name: path.relative_to(self.rollout_dir).as_posix() for name, path in paths.items()}

    def counts(self) -> dict[str, int]:
        return {'ft300': len(self.ft300), 'xense': len(self.xense), 'zmq': len(self.zmq)}


class InferenceInputThread:
    """Read terminal input and validate it synchronously through the controller."""

    def __init__(self, controller: InferenceMainController):
        self.controller = controller

    def start(self) -> None:
        threading.Thread(target=self._run, name='InferenceInput', daemon=True).start()

    def _run(self) -> None:
        aliases = {'i': 'initialize', 's': 'start', 'd': 'stop', 'a': 'abort', 'q': 'shutdown'}
        while self.controller.get_state() != InferenceState.STOPPED:
            try:
                line = input().strip().lower()
            except EOFError:
                return
            if line:
                self.controller.admit_user_action(aliases.get(line, line))


class InferenceMainController:
    """Own one persistent inference session and repeated explicit rollouts."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.run_id = time.strftime('inference_%Y%m%d_%H%M%S')
        self.session_dir = config.runtime_sessions_dir / self.run_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.logger = JsonlLogger(self.session_dir / 'controller_events.jsonl')
        self.state = InferenceState.CREATED
        self.state_lock = threading.RLock()
        self.commands: queue.Queue[InferenceCommand] = queue.Queue()
        self.processes: dict[str, ManagedProcess] = {}
        self.expected_process_exits: set[str] = set()
        self.control: ControlledClient | None = None
        self.aligned_reader: AlignedHealthReader | None = None
        self.zmq_receiver: ZmqTelemetryReceiver | None = None
        self.rosbag: RosbagControl | None = None
        self.ft_client = UdsClient(
            'ft300', config.ft_uds_path, self._on_uds_event,
            on_disconnect=self._on_uds_disconnect,
        )
        self.xense_client = UdsClient(
            'xense', config.xense_uds_path, self._on_uds_event,
            magic=b'XS', on_disconnect=self._on_uds_disconnect,
        )
        self.rollout_store: InferenceRolloutStore | None = None
        self.rollout_started_ns: int | None = None
        self.rosbag_uri: Path | None = None
        self.recording_active = False
        self.rollout_index = 0
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._termination_thread: threading.Thread | None = None
        self._termination_lock = threading.Lock()
        self.termination_mode: str | None = None
        self.termination_reason: str | None = None
        self._resources_stopped = False
        self._logger_closed = False

    def run(self) -> None:
        """Start the inference session and process admitted actions."""
        try:
            self.startup()
            InferenceInputThread(self).start()
            while self.get_state() != InferenceState.STOPPED:
                try:
                    command = self.commands.get(timeout=0.2)
                except queue.Empty:
                    continue
                self.handle_command(command)
        except KeyboardInterrupt:
            self.admit_user_action('shutdown')
        except BaseException as exc:
            self.request_fail_stop('main_controller_internal_error', str(exc))
        finally:
            thread = self._termination_thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=self.config.worker_exit_timeout_s + 5.0)
            if self.get_state() != InferenceState.STOPPED:
                self.request_fail_stop('main_controller_exit', 'controller run loop exited')
                thread = self._termination_thread
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout=self.config.worker_exit_timeout_s + 5.0)

    def startup(self) -> None:
        """Start all required workstation processes, then connect persistent LeRobot."""
        self.set_state(InferenceState.STARTING_SERVICES)
        try:
            self._start_receivers()
            self._start_processes()
            self._wait_services_ready()
            self._start_lerobot()
            assert self.control is not None
            ready = self.control.wait_for_status({'READY'})
            if ready.phase != 'WAIT_INITIALIZE':
                raise RuntimeError(f'unexpected LeRobot READY phase: {ready.phase}')
            self.aligned_reader = AlignedHealthReader(self.config.observation_shm_name)
            health = self.aligned_reader.read()
            if health.fatal:
                raise RuntimeError(
                    f'aligned observation reports fatal {health.status_code}: {health.message}'
                )
            if not health.ready:
                raise RuntimeError('aligned observation SHM is not ready')
            self.set_state(InferenceState.WAIT_START)
            self.log('session_ready')
        except BaseException:
            self._stop_runtime_resources()
            raise

    def admit_user_action(self, action: str) -> bool:
        """Validate and reserve one user action at its actual receipt time."""
        transitions = {
            ('initialize', InferenceState.WAIT_START): InferenceState.INITIALIZING,
            ('start', InferenceState.READY): InferenceState.STARTING,
            ('stop', InferenceState.RUNNING): InferenceState.STOPPING_ROLLOUT,
            ('abort', InferenceState.RUNNING): InferenceState.ABORTING,
            ('shutdown', InferenceState.WAIT_START): InferenceState.SHUTTING_DOWN,
            ('shutdown', InferenceState.READY): InferenceState.SHUTTING_DOWN,
        }
        with self.state_lock:
            current = self.state
            target = transitions.get((action, current))
            if target is None:
                self.log('command_rejected', command=action, received_state=current.name)
                return False
            self._set_state_locked(target)
            self.commands.put(InferenceCommand(action, {'received_state': current.name}))
            self.log('command_admitted', command=action, received_state=current.name)
            return True

    def request_rollout_abort(self, reason: str, details: dict[str, Any] | None = None) -> bool:
        """Atomically admit one recoverable abort only while RUNNING."""
        with self.state_lock:
            if self.state != InferenceState.RUNNING:
                return False
            self._set_state_locked(InferenceState.ABORTING)
            self.commands.put(InferenceCommand('recoverable_abort', {'reason': reason, **(details or {})}))
            return True

    def request_fail_stop(self, reason: str, message: str) -> None:
        """Establish a session-fatal intent once and execute it asynchronously."""
        with self._termination_lock:
            if self.termination_mode is not None:
                return
            self.termination_mode = 'FAIL_STOP'
            self.termination_reason = f'{reason}: {message}'
            with self.state_lock:
                if self.state != InferenceState.STOPPED:
                    self._set_state_locked(InferenceState.FAIL_STOPPING)
            self.log('session_fail_stop_requested', reason=reason, message=message)
            self._termination_thread = threading.Thread(
                target=self._execute_fail_stop,
                name='InferenceFailStop',
                daemon=True,
            )
            self._termination_thread.start()

    def handle_command(self, command: InferenceCommand) -> None:
        """Execute one already-admitted action without reinterpreting it later."""
        if command.name == 'initialize':
            self._initialize_rollout()
        elif command.name == 'start':
            self._start_rollout()
        elif command.name == 'stop':
            self._stop_rollout()
        elif command.name in {'abort', 'recoverable_abort'}:
            payload = command.payload or {}
            reason = payload.get('reason', 'user_abort')
            self._abort_rollout(str(reason), payload)
        elif command.name == 'shutdown':
            self._execute_shutdown()
        elif command.name == 'rollout_completed':
            if self.get_state() == InferenceState.STOPPING_ROLLOUT:
                try:
                    self._finish_recording('done', {'completion_status': 'COMPLETED'})
                    self.set_state(InferenceState.WAIT_START)
                except BaseException as exc:
                    self.request_fail_stop('recording_finalize_failure', str(exc))

    def _initialize_rollout(self) -> None:
        try:
            result = self._require_control().transact('INITIALIZE', {'INITIALIZED'})
            if not result.ack.accepted:
                self.log('initialize_rejected', code=result.ack.code, phase=result.ack.phase)
                self.set_state(InferenceState.WAIT_START)
                return
            assert result.completion is not None
            self.log('initialize_completed', status=result.completion.status)
            if self.get_state() == InferenceState.INITIALIZING:
                self.set_state(InferenceState.READY)
        except BaseException as exc:
            self.request_fail_stop('lerobot_initialize_failure', str(exc))

    def _start_rollout(self) -> None:
        try:
            self._begin_recording()
            result = self._require_control().transact('START', {'STARTED'})
            if not result.ack.accepted:
                raise RuntimeError(
                    f'LeRobot rejected START: {result.ack.code} in {result.ack.phase}'
                )
            assert result.completion is not None
            self.log('rollout_started', status=result.completion.status)
            if self.get_state() == InferenceState.STARTING:
                self.set_state(InferenceState.RUNNING)
                self._start_watchdog()
        except BaseException as exc:
            self._fail_recording_best_effort('start_failure', str(exc))
            self.request_fail_stop('rollout_start_failure', str(exc))

    def _stop_rollout(self) -> None:
        self._finish_rollout_transaction('STOP', 'STOPPED', 'done', 'user_stop')

    def _abort_rollout(self, reason: str, details: dict[str, Any]) -> None:
        self._finish_rollout_transaction('ABORT', 'ABORTED', 'failed', reason, details)

    def _finish_rollout_transaction(
        self,
        operation: str,
        completion: str,
        recording_status: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._stop_watchdog()
        try:
            result = self._require_control().transact(operation, {completion})
            if not result.ack.accepted:
                raise RuntimeError(
                    f'LeRobot rejected {operation}: {result.ack.code} in {result.ack.phase}'
                )
            self._finish_recording(recording_status, {'reason': reason, **(details or {})})
            self.log('rollout_finished', operation=operation, outcome=completion, reason=reason)
            if self.get_state() in {InferenceState.ABORTING, InferenceState.STOPPING_ROLLOUT}:
                self.set_state(InferenceState.WAIT_START)
        except BaseException as exc:
            self._fail_recording_best_effort('rollout_finish_failure', str(exc))
            self.request_fail_stop('rollout_finish_failure', str(exc))

    def _on_control_status(self, status: Status) -> None:
        self.log(
            'lerobot_status', status=status.status, phase=status.phase,
            code=status.code, message=status.message,
        )
        if status.status == 'ERROR':
            self.request_fail_stop('lerobot_fatal', f'{status.code}: {status.message}')
            return
        if status.status == 'COMPLETED':
            with self.state_lock:
                if self.state != InferenceState.RUNNING:
                    return
                self._set_state_locked(InferenceState.STOPPING_ROLLOUT)
                self.commands.put(InferenceCommand('rollout_completed'))

    def _on_control_disconnect(self, exc: BaseException) -> None:
        process = self.processes.get('lerobot')
        if process is not None and process.poll() is not None:
            self.request_fail_stop('lerobot_exit', f'exit code {process.poll()}')
        else:
            self.request_fail_stop('lerobot_control_disconnect', str(exc))

    def _begin_recording(self) -> None:
        self.rollout_index += 1
        rollout_dir = self.session_dir / 'rollouts' / f'rollout_{self.rollout_index:04d}'
        rollout_dir.mkdir(parents=True, exist_ok=False)
        self.rollout_store = InferenceRolloutStore(rollout_dir)
        self.rollout_started_ns = time.time_ns()
        self.rosbag_uri = rollout_dir / 'rosbag'
        started_sensors: list[UdsClient] = []
        try:
            for client in (self.ft_client, self.xense_client):
                self._sensor_command(client, MsgType.START_REQ, 'START_REQ', self.config.sensor_ack_timeout_s)
                started_sensors.append(client)
            if self.rosbag is None:
                raise RuntimeError('rosbag recorder control is unavailable')
            self.rosbag.record(self.rosbag_uri, timeout_s=self.config.rosbag_timeout_s)
            self.recording_active = True
            self.log('recording_started', rollout_dir=str(rollout_dir))
        except BaseException:
            for client in started_sensors:
                try:
                    self._sensor_command(
                        client, MsgType.STOP_REQ, 'STOP_REQ', self.config.sensor_flush_timeout_s
                    )
                except BaseException:
                    pass
            raise

    def _finish_recording(self, status: str, extra: dict[str, Any]) -> Path | None:
        store = self.rollout_store
        if store is None:
            return None
        sensor_payloads: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        message_type = MsgType.DEMO_DONE_REQ if status == 'done' else MsgType.STOP_REQ
        command_name = 'DEMO_DONE_REQ' if status == 'done' else 'STOP_REQ'
        sensor_timeout_s = (
            self.config.sensor_flush_timeout_s
            if status == 'done'
            else self.config.sensor_ack_timeout_s
        )
        for client in (self.ft_client, self.xense_client):
            try:
                sensor_payloads[client.name] = self._sensor_command(
                    client, message_type, command_name, sensor_timeout_s
                )
            except BaseException as exc:
                errors.append(f'{client.name}: {exc}')
        if self.rosbag is not None and self.recording_active:
            try:
                self.rosbag.stop(timeout_s=self.config.rosbag_timeout_s)
            except BaseException as exc:
                errors.append(f'rosbag: {exc}')
        self.recording_active = False
        npz = store.save()
        final_status = status if not errors else 'failed'
        manifest = {
            'status': final_status,
            'started_ns': self.rollout_started_ns,
            'finished_ns': time.time_ns(),
            'rollout_index': self.rollout_index,
            'rosbag_uri': None if self.rosbag_uri is None else self.rosbag_uri.relative_to(store.rollout_dir).as_posix(),
            'sensor_results': sensor_payloads,
            'npz': npz,
            'frame_counts': store.counts(),
            **extra,
        }
        if errors:
            manifest['recording_errors'] = errors
        path = store.rollout_dir / 'manifest.json'
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding='utf-8')
        self.log('recording_finished', manifest=str(path), status=final_status, errors=errors)
        self.rollout_store = None
        self.rosbag_uri = None
        self.rollout_started_ns = None
        if errors:
            raise RuntimeError('; '.join(errors))
        return path

    def _fail_recording_best_effort(self, stage: str, reason: str) -> None:
        try:
            self._finish_recording('failed', {'failure_stage': stage, 'failure_reason': reason})
        except BaseException as exc:
            self.log('recording_failure_cleanup_failed', error=str(exc))

    def _sensor_command(
        self,
        client: UdsClient,
        message_type: MsgType,
        command_name: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        payload = client.send_and_wait_ack(message_type, command_name, timeout_s)
        if payload is None:
            error = client.last_error_for(command_name)
            raise RuntimeError(f'{client.name} {command_name} failed: {error}')
        self.log('sensor_command', sensor=client.name, command=command_name, payload=payload)
        return payload

    def _start_watchdog(self) -> None:
        self._stop_watchdog()
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name='AlignedObservationWatchdog',
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        last_sequence: int | None = None
        last_advance = time.monotonic()
        while not self._watchdog_stop.wait(self.config.aligned_poll_interval_s):
            if self.get_state() != InferenceState.RUNNING:
                return
            try:
                if self.aligned_reader is None:
                    raise RuntimeError('aligned health reader is unavailable')
                health = self.aligned_reader.read()
            except BaseException as exc:
                self.request_fail_stop('aligned_health_read_failure', str(exc))
                return
            if health.fatal:
                self.request_fail_stop(
                    'sensorhub_fatal_observed',
                    f'{health.status_code}: {health.message}',
                )
                return
            if not health.ready:
                self.request_rollout_abort('aligned_not_ready')
                return
            if health.latest_sequence != last_sequence:
                last_sequence = health.latest_sequence
                last_advance = time.monotonic()
                continue
            stalled_s = time.monotonic() - last_advance
            if stalled_s > self.config.aligned_stall_timeout_s:
                self.request_rollout_abort(
                    'aligned_sequence_stall',
                    {'latest_sequence': last_sequence, 'stalled_s': stalled_s},
                )
                return

    def _stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        thread, self._watchdog_thread = self._watchdog_thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _execute_shutdown(self) -> None:
        with self._termination_lock:
            if self.termination_mode is not None:
                return
            self.termination_mode = 'SHUTDOWN'
            self.termination_reason = 'user_requested'
        self.log('session_shutdown_requested')
        try:
            control = self._require_control()
            sequence = control.send('SHUTDOWN')
            ack = control.wait_for_ack(sequence)
        except BaseException as exc:
            self.termination_mode = 'FAIL_STOP'
            self.termination_reason = f'shutdown_failure: {exc}'
            self.set_state(InferenceState.FAIL_STOPPING)
            self.log('shutdown_failed', error=str(exc))
            self._execute_fail_stop()
            return
        if not ack.accepted:
            self.termination_mode = None
            self.termination_reason = None
            self.set_state(InferenceState.WAIT_START)
            self.log('shutdown_rejected', code=ack.code, phase=ack.phase)
            return
        self._wait_for_worker_exit()
        self._stop_runtime_resources()
        self._write_session_manifest()
        self.set_state(InferenceState.STOPPED)
        self._close_logger()

    def _execute_fail_stop(self) -> None:
        self._stop_watchdog()
        self._fail_recording_best_effort('session_fail_stop', self.termination_reason or 'unknown')
        process = self.processes.get('lerobot')
        control = self.control
        delivery_confirmed = False
        try:
            while process is not None and process.poll() is None and not delivery_confirmed:
                if control is None or control.disconnected:
                    break
                sequence = control.send('FAIL_STOP')
                deadline = time.monotonic() + self.config.fail_stop_retry_interval_s
                while time.monotonic() < deadline:
                    ack = control.ack_if_received(sequence)
                    if ack is not None and ack.accepted:
                        delivery_confirmed = True
                        break
                    if any(item.status in {'FAIL_STOPPING', 'ERROR'} for item in control.statuses()):
                        delivery_confirmed = True
                        break
                    if process.poll() is not None:
                        delivery_confirmed = True
                        break
                    time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
            if process is not None and process.poll() is None:
                self._wait_for_worker_exit()
        except BaseException as exc:
            self.log('fail_stop_delivery_error', error=str(exc))
            if process is not None and process.poll() is None:
                self._wait_for_worker_exit()
        finally:
            self._stop_runtime_resources()
            self._write_session_manifest()
            self.set_state(InferenceState.STOPPED)
            self._close_logger()

    def _wait_for_worker_exit(self) -> None:
        process = self.processes.get('lerobot')
        if process is None:
            return
        self.expected_process_exits.add('lerobot')
        deadline = time.monotonic() + self.config.worker_exit_timeout_s
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if process.poll() is None:
            self.log('lerobot_force_terminate')
            process.stop(grace_s=1.0)

    def _start_receivers(self) -> None:
        self.zmq_receiver = ZmqTelemetryReceiver(
            self.config.zmq_connect,
            self._on_zmq_frame,
            self._on_zmq_error,
            lambda message: self.request_fail_stop('zmq_receiver_fatal', message),
            destroy_context_on_stop=True,
        )
        self.zmq_receiver.start()
        self.ft_client.start()
        self.xense_client.start()

    def _start_processes(self) -> None:
        root = self.config.repo_root
        logs = self.session_dir / 'process_logs'
        xense_env = XENSE_SDK_CONDA_ENVS[self.config.xense_sdk_version]
        specs = {
            'ft300': (
                ['conda', 'run', '-n', 'modbus314', 'python', '-m', 'FT300S.app',
                 '--uds-path', self.config.ft_uds_path, '--shm-name', self.config.ft_shm_name,
                 '--fps', str(self.config.ft_fps), '--save-dir', str(self.config.runtime_frames_dir)],
                (),
            ),
            'xense': (
                ['conda', 'run', '-n', xense_env, 'python', '-m', 'XenseTacSensor.app',
                 '--uds-path', self.config.xense_uds_path, '--shm-name', self.config.xense_shm_name,
                 '--fps', str(self.config.xense_fps), '--save-dir', str(self.config.runtime_frames_dir)],
                (),
            ),
            'realsense': (
                bash_cmd('conda deactivate >/dev/null 2>&1 || true; ros2 launch ./RealSense/launch/four_realsense_shm_runtime.launch.py'),
                self.config.fatal_realsense_patterns,
            ),
            'rosbag_recorder': (
                bash_cmd('conda deactivate >/dev/null 2>&1 || true; ros2 launch ./RealSense/launch/rosbag2_recorder.launch.py'),
                (),
            ),
        }
        for name, (command, fatal_patterns) in specs.items():
            process = ManagedProcess(
                name, command, root, logs / f'{name}.log',
                fatal_patterns=fatal_patterns,
                on_fatal=lambda process_name, line: self.request_fail_stop(
                    f'{process_name}_fatal', line
                ),
                on_exit=self._on_process_exit,
            )
            self.processes[name] = process
            process.start()
            self.log('process_started', process=name, command=command)

    def _wait_services_ready(self) -> None:
        if not self.ft_client.wait_connected(self.config.startup_timeout_s):
            raise RuntimeError('FT300S UDS did not connect')
        if not self.xense_client.wait_connected(self.config.startup_timeout_s):
            raise RuntimeError('Xense UDS did not connect')
        if not self.ft_client.wait_init_ready(self.config.init_timeout_s):
            raise RuntimeError('FT300S did not report INIT_READY')
        if not self.xense_client.wait_init_ready(self.config.init_timeout_s):
            raise RuntimeError('Xense did not report INIT_READY')
        self.rosbag = RosbagControl(node_name=f'inference_rosbag_{os.getpid()}')
        if not self.rosbag.wait_ready(self.config.startup_timeout_s):
            raise RuntimeError('rosbag recorder services did not become ready')
        self.rosbag.stop(timeout_s=self.config.rosbag_timeout_s)

    def _start_lerobot(self) -> None:
        process = ManagedProcess(
            'lerobot', self.config.lerobot_command(), self.config.repo_root / 'LeRobotFR3',
            self.session_dir / 'process_logs' / 'lerobot.log',
            on_exit=self._on_process_exit,
        )
        self.processes['lerobot'] = process
        process.start()
        self.log('process_started', process='lerobot', command=process.cmd)
        self.control = ControlledClient(
            self.config.control_socket_path,
            on_status=self._on_control_status,
            on_disconnect=self._on_control_disconnect,
        )
        self.control.connect(self.config.startup_timeout_s)

    def _stop_runtime_resources(self) -> None:
        with self._termination_lock:
            if self._resources_stopped:
                return
            self._resources_stopped = True
        self._stop_watchdog()
        if self.control is not None:
            self.control.close()
        self.ft_client.stop()
        self.xense_client.stop()
        if self.zmq_receiver is not None:
            self.zmq_receiver.stop()
        if self.aligned_reader is not None:
            self.aligned_reader.close()
        for name in ('lerobot', 'rosbag_recorder', 'realsense', 'xense', 'ft300'):
            process = self.processes.get(name)
            if process is not None:
                self.expected_process_exits.add(name)
                process.stop()
        if self.rosbag is not None:
            try:
                self.rosbag.close()
            except BaseException as exc:
                self.log('rosbag_close_failed', error=str(exc))

    def _on_uds_event(self, event: UdsEvent) -> None:
        if event.msg_type == MsgType.ERROR:
            self.request_fail_stop(
                f'{event.client_name}_runtime_error',
                json.dumps(event.payload, ensure_ascii=True),
            )
            return
        if event.msg_type != MsgType.FRAME_READY or not self.recording_active:
            return
        store = self.rollout_store
        if store is None:
            return
        if event.client_name == 'ft300':
            store.ft300.append(
                frame_id=event.frame_id,
                timestamp_ns=_int_or_none(event.payload.get('timestamp_ns')),
                recv_time_ns=event.recv_time_ns,
                recv_monotonic_ns=event.recv_monotonic_ns,
            )
        elif event.client_name == 'xense':
            store.xense.append(
                frame_id=event.frame_id,
                timestamp_ns_0=_int_or_none(event.payload.get('timestamp_ns_0')),
                timestamp_ns_1=_int_or_none(event.payload.get('timestamp_ns_1')),
                recv_time_ns=event.recv_time_ns,
                recv_monotonic_ns=event.recv_monotonic_ns,
            )

    def _on_uds_disconnect(self, name: str, pending_commands: list[str]) -> None:
        self.request_fail_stop(
            f'{name}_uds_disconnect',
            f'pending_commands={pending_commands}',
        )

    def _on_zmq_frame(
        self,
        frame: TelemetryFrame,
        recv_time_ns: int,
        recv_monotonic_ns: int,
    ) -> None:
        if frame.source == SOURCE_ROBOT and frame.flags & ROBOT_TELEMETRY_FLAG_JUMP_HOLD:
            self.request_rollout_abort(
                'jump_hold',
                {'telemetry_sequence': frame.seq, 'telemetry_flags': frame.flags},
            )
        if not self.recording_active or self.rollout_store is None:
            return
        self.rollout_store.zmq.append(
            source=frame.source,
            flags=frame.flags,
            seq=frame.seq,
            stamp_s=frame.stamp,
            valid_mask=frame.valid_mask,
            floats_58=frame.floats_58,
            gripper_gPO=frame.gripper_gPO,
            gripper_gCU=frame.gripper_gCU,
            recv_time_ns=recv_time_ns,
            recv_monotonic_ns=recv_monotonic_ns,
        )

    def _on_zmq_error(self, message: str) -> None:
        self.log('zmq_frame_rejected', message=message)

    def _on_process_exit(self, name: str, returncode: int) -> None:
        if name in self.expected_process_exits:
            self.expected_process_exits.discard(name)
            self.log('process_exited_expected', process=name, returncode=returncode)
            return
        self.log('process_exited_unexpected', process=name, returncode=returncode)
        self.request_fail_stop(f'{name}_unexpected_exit', f'returncode={returncode}')

    def _write_session_manifest(self) -> None:
        payload = {
            'run_id': self.run_id,
            'finished_ns': time.time_ns(),
            'termination_mode': self.termination_mode,
            'termination_reason': self.termination_reason,
            'rollout_count': self.rollout_index,
        }
        (self.session_dir / 'session_manifest.json').write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding='utf-8',
        )

    def _require_control(self) -> ControlledClient:
        if self.control is None:
            raise RuntimeError('LeRobot control channel is unavailable')
        return self.control

    def set_state(self, state: InferenceState) -> None:
        with self.state_lock:
            self._set_state_locked(state)

    def _set_state_locked(self, state: InferenceState) -> None:
        previous = self.state
        self.state = state
        if previous != state:
            self.logger.event('state_transition', previous=previous.name, current=state.name)

    def get_state(self) -> InferenceState:
        with self.state_lock:
            return self.state

    def log(self, event: str, **payload: Any) -> None:
        if self._logger_closed:
            return
        self.logger.event(event, state=self.get_state().name, **payload)

    def _close_logger(self) -> None:
        if not self._logger_closed:
            self._logger_closed = True
            self.logger.close()


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def parse_inference_args() -> argparse.Namespace:
    """Parse the inference-specific entrypoint arguments."""
    parser = argparse.ArgumentParser(description='Persistent FR3 inference MainController')
    parser.add_argument('--policy-path', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--repo-root', default=None)
    parser.add_argument('--runtime-root', default=None)
    parser.add_argument('--control-socket-path', default=None)
    parser.add_argument('--zmq-connect', default='tcp://192.168.1.37:6000')
    parser.add_argument('--robot-command-endpoint', default='tcp://192.168.1.37:6001')
    parser.add_argument('--robot-telemetry-endpoint', default='tcp://192.168.1.37:6000')
    parser.add_argument('--aligned-stall-timeout-s', type=float, default=0.075)
    parser.add_argument('--lerobot-aligned-max-age-ms', type=int, default=100)
    parser.add_argument('--xense-sdk-version', choices=sorted(XENSE_SDK_CONDA_ENVS), default='2.0.1')
    return parser.parse_args()


def build_inference_config(args: argparse.Namespace) -> InferenceConfig:
    """Build an inference config from parsed CLI arguments."""
    root = validate_repo_root(Path(args.repo_root)) if args.repo_root else default_repo_root()
    values: dict[str, Any] = {
        'policy_path': args.policy_path,
        'task': args.task,
        'repo_root': root,
        'runtime_root': None if args.runtime_root is None else Path(args.runtime_root),
        'zmq_connect': args.zmq_connect,
        'robot_command_endpoint': args.robot_command_endpoint,
        'robot_telemetry_endpoint': args.robot_telemetry_endpoint,
        'aligned_stall_timeout_s': args.aligned_stall_timeout_s,
        'lerobot_aligned_max_age_ms': args.lerobot_aligned_max_age_ms,
        'xense_sdk_version': args.xense_sdk_version,
    }
    if args.control_socket_path is not None:
        values['control_socket_path'] = args.control_socket_path
    return InferenceConfig(**values)


def main() -> None:
    """Run the inference-specific controller."""
    controller = InferenceMainController(build_inference_config(parse_inference_args()))
    controller.run()
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == '__main__':
    main()
