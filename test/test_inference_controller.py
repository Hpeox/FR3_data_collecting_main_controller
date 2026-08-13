from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main_controller.inference_config import InferenceConfig
from main_controller.inference_controller import (
    InferenceCommand,
    InferenceMainController,
    InferenceState,
    ROBOT_TELEMETRY_FLAG_JUMP_HOLD,
    ROBOT_TELEMETRY_FLAG_RESETTING,
)
from main_controller.inference_protocol import Ack, Status, TransactionResult
from main_controller.zmq_telemetry import TelemetryFrame


def config(tmp_path, **overrides):
    for relative in ('FT300S', 'XenseTacSensor', 'RealSense/launch', 'LeRobotFR3'):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    for relative in (
        'RealSense/launch/four_realsense_640x480_30.launch.py',
        'RealSense/launch/four_realsense_shm_runtime.launch.py',
        'RealSense/launch/rosbag2_recorder.launch.py',
    ):
        (tmp_path / relative).touch()
    values = {
        'policy_path': 'policy',
        'task': 'task',
        'repo_root': tmp_path,
        'aligned_stall_timeout_s': 0.03,
        'lerobot_aligned_max_age_ms': 100,
        'aligned_poll_interval_s': 0.005,
        'worker_exit_timeout_s': 0.03,
        'fail_stop_retry_interval_s': 0.01,
    }
    values.update(overrides)
    return InferenceConfig(**values)


def ack(sequence, operation, accepted=True):
    return Ack(
        sequence=sequence,
        operation=operation,
        accepted=accepted,
        code='accepted' if accepted else 'invalid_phase',
        phase='phase',
        message='',
    )


def status(sequence, name, phase):
    return Status(
        sequence=sequence,
        status=name,
        phase=phase,
        code='ok',
        message='',
        timestamp_ns=time.time_ns(),
    )


class FakeControl:
    def __init__(self):
        self.transactions = []
        self.sent = []
        self.disconnected = False
        self.ack_after = 1
        self.ack_accepted = True

    def transact(self, operation, completion_statuses):
        self.transactions.append(operation)
        completion = next(iter(completion_statuses))
        return TransactionResult(
            ack=ack(len(self.transactions), operation),
            completion=status(len(self.transactions), completion, 'phase'),
        )

    def send(self, operation):
        self.sent.append(operation)
        return len(self.sent)

    def wait_for_ack(self, sequence):
        return ack(sequence, self.sent[sequence - 1], accepted=self.ack_accepted)

    def ack_if_received(self, sequence):
        if sequence < self.ack_after:
            return None
        return ack(sequence, self.sent[sequence - 1])

    def statuses(self):
        return ()

    def close(self):
        self.disconnected = True


class FakeSensor:
    def __init__(self, name):
        self.name = name
        self.commands = []

    def send_and_wait_ack(self, message_type, command_name, timeout_s):
        self.commands.append(command_name)
        return {'cmd': command_name}

    def last_error_for(self, command_name):
        return None

    def stop(self):
        pass


class FakeRosbag:
    def __init__(self):
        self.calls = []

    def record(self, uri, timeout_s):
        self.calls.append(('record', Path(uri)))

    def stop(self, timeout_s):
        self.calls.append(('stop', None))

    def close(self):
        pass


def controller(tmp_path):
    instance = InferenceMainController(config(tmp_path))
    instance.control = FakeControl()
    instance.ft_client = FakeSensor('ft300')
    instance.xense_client = FakeSensor('xense')
    instance.rosbag = FakeRosbag()
    instance._start_watchdog = lambda: None
    instance._stop_watchdog = lambda: None
    return instance


def test_input_is_validated_at_receipt_and_never_deferred(tmp_path):
    instance = controller(tmp_path)
    instance.set_state(InferenceState.WAIT_START)
    assert instance.admit_user_action('initialize')
    assert instance.get_state() == InferenceState.INITIALIZING
    assert not instance.admit_user_action('start')
    queued = instance.commands.get_nowait()
    assert queued.name == 'initialize'
    assert instance.commands.empty()


def test_repeated_explicit_rollouts_keep_one_worker(tmp_path):
    instance = controller(tmp_path)
    instance.set_state(InferenceState.WAIT_START)
    for _ in range(2):
        assert instance.admit_user_action('initialize')
        instance.handle_command(instance.commands.get_nowait())
        assert instance.get_state() == InferenceState.READY
        assert instance.admit_user_action('start')
        instance.handle_command(instance.commands.get_nowait())
        assert instance.get_state() == InferenceState.RUNNING
        assert instance.admit_user_action('stop')
        instance.handle_command(instance.commands.get_nowait())
        assert instance.get_state() == InferenceState.WAIT_START
    assert instance.control.transactions == [
        'INITIALIZE', 'START', 'STOP', 'INITIALIZE', 'START', 'STOP'
    ]
    assert instance.rollout_index == 2
    for index in (1, 2):
        manifest = json.loads(
            (instance.session_dir / 'rollouts' / f'rollout_{index:04d}' / 'manifest.json').read_text()
        )
        assert manifest['status'] == 'done'


def test_jump_hold_is_immediate_recoverable_abort_and_flags_are_recorded(tmp_path):
    instance = controller(tmp_path)
    instance.set_state(InferenceState.RUNNING)
    instance._begin_recording()
    frame = TelemetryFrame(
        source=2,
        flags=ROBOT_TELEMETRY_FLAG_JUMP_HOLD,
        seq=44,
        stamp=1.0,
        valid_mask=2,
        floats_58=(0.0,) * 58,
        gripper_gPO=0,
        gripper_gCU=0,
    )
    instance._on_zmq_frame(frame, 10, 11)
    assert instance.get_state() == InferenceState.ABORTING
    admitted = instance.commands.get_nowait()
    assert admitted.name == 'recoverable_abort'
    instance.handle_command(admitted)
    assert instance.get_state() == InferenceState.WAIT_START
    manifest = json.loads(
        (instance.session_dir / 'rollouts' / 'rollout_0001' / 'manifest.json').read_text()
    )
    assert manifest['status'] == 'failed'
    assert manifest['reason'] == 'jump_hold'
    import numpy as np
    arrays = np.load(instance.session_dir / 'rollouts' / 'rollout_0001' / 'zmq_telemetry.npz')
    assert arrays['flags'].tolist() == [ROBOT_TELEMETRY_FLAG_JUMP_HOLD]


def test_raw_resetting_never_completes_initialize(tmp_path):
    instance = controller(tmp_path)
    instance.set_state(InferenceState.INITIALIZING)
    frame = TelemetryFrame(
        source=2,
        flags=ROBOT_TELEMETRY_FLAG_RESETTING,
        seq=45,
        stamp=1.0,
        valid_mask=2,
        floats_58=(0.0,) * 58,
        gripper_gPO=0,
        gripper_gCU=0,
    )
    instance._on_zmq_frame(frame, 10, 11)
    assert instance.get_state() == InferenceState.INITIALIZING
    assert instance.commands.empty()


def test_aligned_stall_requests_recoverable_abort(tmp_path):
    instance = InferenceMainController(config(tmp_path))
    instance.aligned_reader = SimpleNamespace(
        read=lambda: SimpleNamespace(
            ready=True, latest_sequence=7, fatal=False, status_code=0, message=''
        )
    )
    instance.set_state(InferenceState.RUNNING)
    instance._start_watchdog()
    deadline = time.monotonic() + 0.3
    while instance.get_state() == InferenceState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.005)
    assert instance.get_state() == InferenceState.ABORTING
    command = instance.commands.get_nowait()
    assert command.payload['reason'] == 'aligned_sequence_stall'
    instance._stop_watchdog()


class FakeProcess:
    def __init__(self):
        self.stopped = False

    def poll(self):
        return 0 if self.stopped else None

    def stop(self, grace_s=5.0):
        self.stopped = True


def test_fail_stop_retries_with_fresh_sequences_then_waits_for_exit(tmp_path):
    instance = InferenceMainController(config(tmp_path))
    control = FakeControl()
    control.ack_after = 3
    process = FakeProcess()
    instance.control = control
    instance.processes['lerobot'] = process
    instance.termination_mode = 'FAIL_STOP'
    instance.termination_reason = 'test fault'
    instance._stop_runtime_resources = lambda: None
    instance._write_session_manifest = lambda: None
    instance._execute_fail_stop()
    assert control.sent == ['FAIL_STOP', 'FAIL_STOP', 'FAIL_STOP']
    assert process.stopped
    assert instance.get_state() == InferenceState.STOPPED


def test_shutdown_is_rejected_during_initialize_without_wire_send(tmp_path):
    instance = controller(tmp_path)
    instance.set_state(InferenceState.INITIALIZING)
    assert not instance.admit_user_action('shutdown')
    assert instance.control.sent == []


def test_worker_shutdown_rejection_keeps_session_alive(tmp_path):
    instance = controller(tmp_path)
    instance.control.ack_accepted = False
    instance.set_state(InferenceState.SHUTTING_DOWN)
    instance._execute_shutdown()
    assert instance.get_state() == InferenceState.WAIT_START
    assert instance.termination_mode is None
    assert not instance._resources_stopped


def test_shutdown_from_running_settles_rollout_before_wire_shutdown(tmp_path):
    instance = controller(tmp_path)
    instance.set_state(InferenceState.RUNNING)
    instance._begin_recording()
    assert instance.admit_user_action('shutdown')
    instance.handle_command(instance.commands.get_nowait())
    assert instance.control.transactions == ['STOP']
    assert instance.control.sent == ['SHUTDOWN']
    assert instance.get_state() == InferenceState.STOPPED
    manifest = json.loads(
        (instance.session_dir / 'rollouts' / 'rollout_0001' / 'manifest.json').read_text()
    )
    assert manifest['status'] == 'done'
    assert manifest['reason'] == 'user_shutdown'


def test_required_process_exit_is_session_fatal_without_restart(tmp_path):
    instance = controller(tmp_path)
    observed = []
    instance.request_fail_stop = lambda reason, message: observed.append((reason, message))
    instance._on_process_exit('realsense', 7)
    assert observed == [('realsense_unexpected_exit', 'returncode=7')]
    assert not hasattr(instance.processes.get('realsense'), 'restart')


def test_fault_promotes_in_progress_shutdown_to_fail_stop(tmp_path):
    instance = controller(tmp_path)
    started = threading.Event()
    instance._execute_fail_stop = started.set
    instance.termination_mode = 'SHUTDOWN'
    instance.termination_reason = 'user_requested'
    instance.set_state(InferenceState.SHUTTING_DOWN)
    instance.request_fail_stop('realsense_unexpected_exit', 'returncode=7')
    assert started.wait(timeout=1.0)
    assert instance.termination_mode == 'FAIL_STOP'
    assert instance.termination_reason == 'realsense_unexpected_exit: returncode=7'
    assert instance.get_state() == InferenceState.FAIL_STOPPING
    assert instance._termination_thread is not None
    instance._termination_thread.join(timeout=1.0)
