from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main_controller.inference_controller as inference_controller_module
from main_controller.inference_config import InferenceConfig
from main_controller.inference_controller import (
    InferenceCommand,
    InferenceMainController,
    InferenceState,
    ROBOT_TELEMETRY_FLAG_JUMP_HOLD,
    ROBOT_TELEMETRY_FLAG_RESETTING,
)
from main_controller.inference_protocol import Ack, Status, TransactionResult
from main_controller.uds_client import MsgType
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


class EventControl(FakeControl):
    def __init__(self, events, *, start_accepted=True):
        super().__init__()
        self.events = events
        self.start_accepted = start_accepted

    def transact(self, operation, completion_statuses):
        self.events.append(('lerobot', operation))
        result = super().transact(operation, completion_statuses)
        if operation == 'START' and not self.start_accepted:
            return TransactionResult(
                ack=ack(len(self.transactions), operation, accepted=False),
                completion=None,
            )
        return result


class StartupControl:
    def __init__(self, events, *, ready=None, error=None):
        self.events = events
        self.ready = ready or status(1, 'READY', 'WAIT_INITIALIZE')
        self.error = error

    def wait_for_status(self, expected):
        self.events.append(('lerobot', 'READY'))
        if self.error is not None:
            raise self.error
        return self.ready

    def close(self):
        pass


class FakeSensor:
    def __init__(self, name, *, events=None, fail_commands=()):
        self.name = name
        self.commands = []
        self.events = events
        self.fail_commands = set(fail_commands)

    def send_and_wait_ack(self, message_type, command_name, timeout_s):
        self.commands.append(command_name)
        if self.events is not None:
            self.events.append((self.name, command_name))
        if command_name in self.fail_commands:
            return None
        return {'cmd': command_name}

    def last_error_for(self, command_name):
        if command_name in self.fail_commands:
            return {'error': 'injected_failure', 'cmd': command_name}
        return None

    def stop(self):
        pass


class FakeRosbag:
    def __init__(self, *, events=None):
        self.calls = []
        self.image_readiness_ok = True
        self.events = events

    def record(self, uri, timeout_s):
        self.calls.append(('record', Path(uri)))
        if self.events is not None:
            self.events.append(('rosbag', 'record'))

    def stop(self, timeout_s):
        self.calls.append(('stop', None))

    def check_image_readiness(self, requirements, timeout_s, mode):
        self.calls.append(('check_image_readiness', tuple(requirements)))
        return SimpleNamespace(
            ok=self.image_readiness_ok,
            to_manifest=lambda: {
                'ok': self.image_readiness_ok,
                'mode': mode,
                'required_topics': [requirement.topic for requirement in requirements],
            },
        )

    def close(self):
        pass


def aligned_health(sequence, *, ready=True, fatal=False, message=''):
    return SimpleNamespace(
        ready=ready,
        latest_sequence=sequence,
        fatal=fatal,
        status_code=1 if fatal else 0,
        message=message,
    )


class AdvancingAlignedReader:
    def __init__(self, start=0, *, events=None):
        self.sequence = start
        self.events = events

    def read(self):
        self.sequence += 1
        if self.events is not None:
            self.events.append(('aligned', 'read', self.sequence))
        return aligned_health(self.sequence)

    def close(self):
        pass


class SequenceAlignedReader:
    def __init__(self, values, *, events=None):
        self.values = list(values)
        self.events = events
        self.last = self.values[-1]

    def read(self):
        if self.values:
            self.last = self.values.pop(0)
        if isinstance(self.last, BaseException):
            raise self.last
        if self.events is not None:
            self.events.append(('aligned', 'read', self.last.latest_sequence))
        return self.last

    def close(self):
        pass


def controller(tmp_path):
    instance = InferenceMainController(config(tmp_path))
    instance.control = FakeControl()
    instance.ft_client = FakeSensor('ft300')
    instance.xense_client = FakeSensor('xense')
    instance.rosbag = FakeRosbag()
    instance.aligned_reader = AdvancingAlignedReader()
    instance._start_watchdog = lambda: None
    instance._stop_watchdog = lambda: None
    return instance


def startup_controller(tmp_path, events):
    instance = InferenceMainController(config(tmp_path))
    instance.ft_client = FakeSensor('ft300', events=events)
    instance.xense_client = FakeSensor('xense', events=events)
    instance.rosbag = FakeRosbag(events=events)
    instance._start_receivers = lambda: None
    instance._start_processes = lambda: None
    instance._wait_services_ready = lambda: None
    instance._wait_realsense_startup_ready = lambda: None

    def start_lerobot():
        events.append(('lerobot', 'launch'))
        instance.control = StartupControl(events)

    instance._start_lerobot = start_lerobot

    def register_aligned_reader():
        events.append(('aligned', 'attach'))
        instance.aligned_reader = SequenceAlignedReader(
            [aligned_health(10)],
            events=events,
        )

    instance._register_aligned_reader = register_aligned_reader
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
    assert instance.ft_client.commands == [
        'START_REQ', 'DEMO_DONE_REQ', 'START_REQ', 'DEMO_DONE_REQ'
    ]
    assert instance.xense_client.commands == [
        'START_REQ', 'DEMO_DONE_REQ', 'START_REQ', 'DEMO_DONE_REQ'
    ]
    assert instance.rollout_index == 2
    for index in (1, 2):
        manifest = json.loads(
            (instance.session_dir / 'rollouts' / f'rollout_{index:04d}' / 'manifest.json').read_text()
        )
        assert manifest['status'] == 'done'


def test_rollout_rearms_aligned_stream_before_rosbag_and_lerobot_start(tmp_path):
    events = []
    instance = controller(tmp_path)
    instance.ft_client = FakeSensor('ft300', events=events)
    instance.xense_client = FakeSensor('xense', events=events)
    instance.rosbag = FakeRosbag(events=events)
    instance.control = EventControl(events)
    instance.aligned_reader = SequenceAlignedReader(
        [aligned_health(40), aligned_health(40), aligned_health(41)],
        events=events,
    )
    instance.set_state(InferenceState.READY)

    assert instance.admit_user_action('start')
    instance.handle_command(instance.commands.get_nowait())

    assert instance.get_state() == InferenceState.RUNNING
    assert events == [
        ('aligned', 'read', 40),
        ('ft300', 'START_REQ'),
        ('xense', 'START_REQ'),
        ('aligned', 'read', 40),
        ('aligned', 'read', 41),
        ('rosbag', 'record'),
        ('lerobot', 'START'),
    ]


def test_xense_start_failure_discards_only_acked_ft_without_stop(tmp_path):
    events = []
    instance = controller(tmp_path)
    instance.ft_client = FakeSensor('ft300', events=events)
    instance.xense_client = FakeSensor(
        'xense',
        events=events,
        fail_commands={'START_REQ'},
    )

    with pytest.raises(RuntimeError, match='xense START_REQ failed'):
        instance._start_sensor_acquisition()

    assert events == [
        ('ft300', 'START_REQ'),
        ('xense', 'START_REQ'),
        ('ft300', 'DEMO_DISCARD_REQ'),
    ]
    assert instance._active_sensor_clients == {}


@pytest.mark.parametrize(
    ('rearm_health', 'expected_reason'),
    [
        (aligned_health(10, ready=False), 'not ready'),
        (aligned_health(10, fatal=True, message='source failed'), 'is fatal'),
    ],
)
def test_unhealthy_aligned_rearm_discards_sensors_before_fail_stop(
    tmp_path,
    rearm_health,
    expected_reason,
):
    events = []
    instance = controller(tmp_path)
    instance.ft_client = FakeSensor('ft300', events=events)
    instance.xense_client = FakeSensor('xense', events=events)
    instance.rosbag = FakeRosbag(events=events)
    instance.aligned_reader = SequenceAlignedReader(
        [aligned_health(10), rearm_health],
        events=events,
    )
    fail_stops = []
    instance.request_fail_stop = lambda reason, message: fail_stops.append((reason, message))
    instance.set_state(InferenceState.STARTING)

    instance._start_rollout()

    assert expected_reason in fail_stops[0][1]
    assert ('rosbag', 'record') not in events
    assert instance.control.transactions == []
    assert instance.ft_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert instance.xense_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert 'STOP_REQ' not in instance.ft_client.commands + instance.xense_client.commands


def test_aligned_rearm_timeout_discards_sensors_without_downstream_start(tmp_path):
    events = []
    instance = controller(tmp_path)
    instance.config = config(
        tmp_path,
        startup_timeout_s=0.01,
        aligned_poll_interval_s=0.001,
    )
    instance.ft_client = FakeSensor('ft300', events=events)
    instance.xense_client = FakeSensor('xense', events=events)
    instance.rosbag = FakeRosbag(events=events)
    instance.aligned_reader = SequenceAlignedReader(
        [aligned_health(10)],
        events=events,
    )
    fail_stops = []
    instance.request_fail_stop = lambda reason, message: fail_stops.append((reason, message))
    instance.set_state(InferenceState.STARTING)

    instance._start_rollout()

    assert 'did not advance' in fail_stops[0][1]
    assert ('rosbag', 'record') not in events
    assert instance.control.transactions == []
    assert instance.ft_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert instance.xense_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']


def test_rollout_start_preserves_aligned_and_discard_failures(tmp_path):
    instance = controller(tmp_path)
    instance.ft_client = FakeSensor('ft300')
    instance.xense_client = FakeSensor(
        'xense',
        fail_commands={'DEMO_DISCARD_REQ'},
    )
    instance.aligned_reader = SequenceAlignedReader(
        [aligned_health(10), aligned_health(10, ready=False)]
    )
    fail_stops = []
    instance.request_fail_stop = lambda reason, message: fail_stops.append((reason, message))
    instance.set_state(InferenceState.STARTING)

    instance._start_rollout()

    assert 'not ready' in fail_stops[0][1]
    assert 'xense DEMO_DISCARD_REQ failed' in fail_stops[0][1]
    assert instance.ft_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert instance.xense_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert set(instance._active_sensor_clients) == {'xense'}


def test_lerobot_start_rejection_discards_sensors_and_stops_rosbag(tmp_path):
    events = []
    instance = controller(tmp_path)
    instance.ft_client = FakeSensor('ft300', events=events)
    instance.xense_client = FakeSensor('xense', events=events)
    instance.rosbag = FakeRosbag(events=events)
    instance.control = EventControl(events, start_accepted=False)
    fail_stops = []
    instance.request_fail_stop = lambda reason, message: fail_stops.append((reason, message))
    instance.set_state(InferenceState.STARTING)

    instance._start_rollout()

    assert fail_stops[0][0] == 'rollout_start_failure'
    assert instance.ft_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert instance.xense_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert [call[0] for call in instance.rosbag.calls] == ['record', 'stop']


def test_rosbag_start_failure_discards_sensors_without_lerobot_start(tmp_path):
    events = []
    instance = controller(tmp_path)
    instance.ft_client = FakeSensor('ft300', events=events)
    instance.xense_client = FakeSensor('xense', events=events)

    class FailingRosbag(FakeRosbag):
        def record(self, uri, timeout_s):
            super().record(uri, timeout_s)
            raise RuntimeError('record failed')

    instance.rosbag = FailingRosbag(events=events)
    fail_stops = []
    instance.request_fail_stop = lambda reason, message: fail_stops.append((reason, message))
    instance.set_state(InferenceState.STARTING)

    instance._start_rollout()

    assert 'record failed' in fail_stops[0][1]
    assert instance.control.transactions == []
    assert instance.ft_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert instance.xense_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert [call[0] for call in instance.rosbag.calls] == ['record', 'stop']


def test_abort_discards_sensors_and_allows_another_rollout(tmp_path):
    instance = controller(tmp_path)
    instance.set_state(InferenceState.READY)
    assert instance.admit_user_action('start')
    instance.handle_command(instance.commands.get_nowait())
    assert instance.admit_user_action('abort')
    instance.handle_command(instance.commands.get_nowait())
    assert instance.get_state() == InferenceState.WAIT_START

    assert instance.admit_user_action('initialize')
    instance.handle_command(instance.commands.get_nowait())
    assert instance.admit_user_action('start')
    instance.handle_command(instance.commands.get_nowait())

    assert instance.get_state() == InferenceState.RUNNING
    assert instance.ft_client.commands == [
        'START_REQ', 'DEMO_DISCARD_REQ', 'START_REQ'
    ]
    assert instance.xense_client.commands == [
        'START_REQ', 'DEMO_DISCARD_REQ', 'START_REQ'
    ]


def test_jump_hold_is_immediate_recoverable_abort_and_flags_are_recorded(tmp_path):
    instance = controller(tmp_path)
    instance.set_state(InferenceState.STARTING)
    instance._begin_recording()
    instance.set_state(InferenceState.RUNNING)
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


class FakeRestartableProcess(FakeProcess):
    def __init__(self, *, restart_error=None):
        super().__init__()
        self.restart_count = 0
        self.restart_error = restart_error

    def restart(self, grace_s=5.0):
        self.restart_count += 1
        if self.restart_error is not None:
            raise self.restart_error
        self.stopped = False


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


def test_shutdown_rejection_cannot_clear_concurrent_fail_stop(tmp_path):
    instance = controller(tmp_path)
    ack_waiting = threading.Event()
    release_ack = threading.Event()
    fail_stop_started = threading.Event()

    def wait_for_rejected_ack(sequence):
        ack_waiting.set()
        assert release_ack.wait(timeout=1.0)
        return ack(sequence, 'SHUTDOWN', accepted=False)

    instance.control.wait_for_ack = wait_for_rejected_ack
    instance._execute_fail_stop = fail_stop_started.set
    instance.set_state(InferenceState.SHUTTING_DOWN)
    shutdown_thread = threading.Thread(target=instance._execute_shutdown)
    shutdown_thread.start()
    assert ack_waiting.wait(timeout=1.0)

    instance.request_fail_stop('realsense_unexpected_exit', 'returncode=7')
    assert fail_stop_started.wait(timeout=1.0)
    release_ack.set()
    shutdown_thread.join(timeout=1.0)

    assert not shutdown_thread.is_alive()
    assert instance.termination_mode == 'FAIL_STOP'
    assert instance.termination_reason == 'realsense_unexpected_exit: returncode=7'
    assert instance.get_state() == InferenceState.FAIL_STOPPING


def test_shutdown_from_running_settles_rollout_before_wire_shutdown(tmp_path):
    instance = controller(tmp_path)
    instance.set_state(InferenceState.STARTING)
    instance._begin_recording()
    instance.set_state(InferenceState.RUNNING)
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


def test_cleanup_marks_managed_exits_expected_before_closing_uds_clients(tmp_path):
    instance = controller(tmp_path)
    managed_names = {'lerobot', 'rosbag_recorder', 'realsense', 'xense', 'ft300'}
    instance.processes = {name: FakeProcess() for name in managed_names}
    instance.termination_mode = 'FAIL_STOP'
    instance.termination_reason = 'original startup failure'
    expected_snapshots = []
    fail_stops = []

    class ExitOnClientClose(FakeSensor):
        def stop(self):
            expected_snapshots.append(set(instance.expected_process_exits))
            instance._on_process_exit(self.name, 1)

    instance.ft_client = ExitOnClientClose('ft300')
    instance.xense_client = ExitOnClientClose('xense')
    instance.request_fail_stop = lambda reason, message: fail_stops.append((reason, message))

    instance._stop_runtime_resources()

    assert expected_snapshots[0] == managed_names
    assert 'xense' in expected_snapshots[1]
    assert fail_stops == []
    assert instance.termination_reason == 'original startup failure'


def test_cleanup_sends_sensor_q_before_closing_either_uds_client(tmp_path):
    instance = controller(tmp_path)
    events = []
    fail_stops = []

    class OrderedSensor(FakeSensor):
        def send_and_wait_ack(self, message_type, command_name, timeout_s):
            events.append((self.name, 'send', message_type, command_name))
            instance._on_uds_disconnect(self.name, [command_name])
            return {'cmd': command_name}

        def stop(self):
            events.append((self.name, 'stop'))

    instance.ft_client = OrderedSensor('ft300')
    instance.xense_client = OrderedSensor('xense')
    instance.request_fail_stop = lambda reason, message: fail_stops.append((reason, message))

    instance._stop_runtime_resources()

    assert events == [
        ('ft300', 'send', MsgType.STOP_REQ, 'STOP_REQ'),
        ('xense', 'send', MsgType.STOP_REQ, 'STOP_REQ'),
        ('ft300', 'stop'),
        ('xense', 'stop'),
    ]
    assert fail_stops == []


def test_sensor_q_failure_does_not_skip_remaining_cleanup(tmp_path):
    instance = controller(tmp_path)
    events = []

    class FailingStopSensor(FakeSensor):
        def send_and_wait_ack(self, message_type, command_name, timeout_s):
            events.append((self.name, 'send'))
            return None if self.name == 'ft300' else {'cmd': command_name}

        def last_error_for(self, command_name):
            return {'error': 'send_failed'}

        def stop(self):
            events.append((self.name, 'stop'))

    instance.ft_client = FailingStopSensor('ft300')
    instance.xense_client = FailingStopSensor('xense')

    instance._stop_runtime_resources()

    assert events == [
        ('ft300', 'send'),
        ('xense', 'send'),
        ('ft300', 'stop'),
        ('xense', 'stop'),
    ]


def test_recoverable_realsense_startup_fatal_restarts_without_fail_stop(tmp_path):
    instance = controller(tmp_path)
    process = FakeRestartableProcess()
    instance.processes['realsense'] = process
    instance.set_state(InferenceState.STARTING_SERVICES)
    fail_stops = []
    instance.request_fail_stop = lambda reason, message: fail_stops.append((reason, message))

    instance._on_process_fatal('realsense', 'Depth stream start failure, Hardware Error')
    instance._recover_realsense_startup_if_needed()

    assert process.restart_count == 1
    assert fail_stops == []
    assert instance.termination_mode is None


def test_duplicate_realsense_fatal_during_restart_does_not_consume_retry(tmp_path):
    instance = controller(tmp_path)
    fail_stops = []

    class DuplicateFatalDuringRestart(FakeProcess):
        def __init__(self):
            super().__init__()
            self.restart_count = 0

        def restart(self, grace_s=5.0):
            self.restart_count += 1
            instance._on_process_fatal(
                'realsense',
                'Hardware Notification:Depth stream start failure, Hardware Error',
            )
            instance._on_process_exit('realsense', -2)

    process = DuplicateFatalDuringRestart()
    instance.processes['realsense'] = process
    instance.set_state(InferenceState.STARTING_SERVICES)
    instance.request_fail_stop = lambda reason, message: fail_stops.append((reason, message))

    instance._on_process_fatal(
        'realsense',
        'XXX Hardware Notification:Depth stream start failure, Hardware Error',
    )
    instance._recover_realsense_startup_if_needed()

    assert process.restart_count == 1
    assert instance._realsense_startup_restart_count == 1
    assert fail_stops == []
    assert 'realsense' not in instance.expected_process_exits


def test_realsense_startup_allows_five_real_restart_attempts(tmp_path):
    instance = InferenceMainController(
        config(tmp_path, realsense_startup_stabilization_s=0.001)
    )

    class CyclingProcess(FakeProcess):
        def __init__(self):
            super().__init__()
            self.restart_count = 0

        def restart(self, grace_s=5.0):
            self.restart_count += 1
            instance._on_process_exit('realsense', -2)

    process = CyclingProcess()
    instance.processes['realsense'] = process
    instance.set_state(InferenceState.STARTING_SERVICES)

    for attempt in range(5):
        instance._on_process_fatal('realsense', f'Hardware Error attempt {attempt + 1}')
        instance._recover_realsense_startup_if_needed()

    assert process.restart_count == 5
    assert instance._realsense_startup_restart_count == 5

    instance._on_process_fatal('realsense', 'Hardware Error attempt 6')
    with pytest.raises(RuntimeError, match='recovery budget exhausted'):
        instance._recover_realsense_startup_if_needed()

    assert process.restart_count == 5


def test_successful_realsense_startup_recovery_allows_startup_to_continue(tmp_path):
    instance = InferenceMainController(
        config(tmp_path, realsense_startup_stabilization_s=0.001)
    )
    process = FakeRestartableProcess()
    instance.processes['realsense'] = process
    instance._start_receivers = lambda: None
    instance._start_processes = lambda: None
    instance._wait_services_ready = lambda: instance._on_process_fatal(
        'realsense', 'Depth stream start failure, Hardware Error'
    )
    instance._wait_realsense_nodes_up = lambda *, start_position: True
    instance._wait_realsense_images_ready = lambda: True
    instance._start_lerobot = lambda: None
    instance.ft_client = FakeSensor('ft300')
    instance.xense_client = FakeSensor('xense')
    instance.control = SimpleNamespace(
        wait_for_status=lambda expected: status(1, 'READY', 'WAIT_INITIALIZE'),
        close=lambda: None,
    )
    instance._register_aligned_reader = lambda: setattr(
        instance,
        'aligned_reader',
        SimpleNamespace(
            read=lambda: SimpleNamespace(
                ready=True,
                latest_sequence=1,
                fatal=False,
                status_code=0,
                message='',
            )
        ),
    )

    instance.startup()

    assert process.restart_count == 1
    assert instance.get_state() == InferenceState.WAIT_START
    assert instance.termination_mode is None


def test_startup_warmup_orders_sensor_start_lerobot_aligned_and_discard(tmp_path):
    events = []
    instance = startup_controller(tmp_path, events)

    instance.startup()

    assert instance.get_state() == InferenceState.WAIT_START
    assert events == [
        ('ft300', 'START_REQ'),
        ('xense', 'START_REQ'),
        ('lerobot', 'launch'),
        ('lerobot', 'READY'),
        ('aligned', 'attach'),
        ('aligned', 'read', 10),
        ('ft300', 'DEMO_DISCARD_REQ'),
        ('xense', 'DEMO_DISCARD_REQ'),
    ]
    assert instance._active_sensor_clients == {}


@pytest.mark.parametrize(
    ('failure_stage', 'expected_reason'),
    [
        ('lerobot_launch', 'launch failed'),
        ('lerobot_ready', 'READY failed'),
        ('aligned_attach', 'attach failed'),
        ('aligned_read', 'read failed'),
        ('aligned_not_ready', 'not ready'),
        ('aligned_fatal', 'is fatal'),
    ],
)
def test_startup_failure_after_warmup_discards_all_active_sensors(
    tmp_path,
    failure_stage,
    expected_reason,
):
    events = []
    instance = startup_controller(tmp_path, events)
    instance._stop_runtime_resources = lambda: events.append(('controller', 'terminal_cleanup'))

    if failure_stage == 'lerobot_launch':
        instance._start_lerobot = lambda: (_ for _ in ()).throw(RuntimeError('launch failed'))
    elif failure_stage == 'lerobot_ready':
        def start_lerobot():
            events.append(('lerobot', 'launch'))
            instance.control = StartupControl(events, error=RuntimeError('READY failed'))

        instance._start_lerobot = start_lerobot
    elif failure_stage == 'aligned_attach':
        instance._register_aligned_reader = lambda: (_ for _ in ()).throw(
            RuntimeError('attach failed')
        )
    elif failure_stage == 'aligned_read':
        instance._register_aligned_reader = lambda: setattr(
            instance,
            'aligned_reader',
            SequenceAlignedReader([RuntimeError('read failed')]),
        )
    elif failure_stage == 'aligned_not_ready':
        instance._register_aligned_reader = lambda: setattr(
            instance,
            'aligned_reader',
            SequenceAlignedReader([aligned_health(10, ready=False)]),
        )
    elif failure_stage == 'aligned_fatal':
        instance._register_aligned_reader = lambda: setattr(
            instance,
            'aligned_reader',
            SequenceAlignedReader(
                [aligned_health(10, fatal=True, message='source failed')]
            ),
        )

    with pytest.raises(RuntimeError, match=expected_reason):
        instance.startup()

    assert instance.ft_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert instance.xense_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert events[-3:] == [
        ('ft300', 'DEMO_DISCARD_REQ'),
        ('xense', 'DEMO_DISCARD_REQ'),
        ('controller', 'terminal_cleanup'),
    ]


def test_startup_preserves_primary_and_all_discard_failures(tmp_path):
    events = []
    instance = startup_controller(tmp_path, events)
    instance.ft_client.fail_commands.add('DEMO_DISCARD_REQ')
    instance.xense_client.fail_commands.add('DEMO_DISCARD_REQ')
    instance._start_lerobot = lambda: (_ for _ in ()).throw(RuntimeError('launch failed'))
    instance._stop_runtime_resources = lambda: None

    with pytest.raises(RuntimeError) as exc_info:
        instance.startup()

    message = str(exc_info.value)
    assert 'launch failed' in message
    assert 'ft300 DEMO_DISCARD_REQ failed' in message
    assert 'xense DEMO_DISCARD_REQ failed' in message
    assert instance.ft_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert instance.xense_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert set(instance._active_sensor_clients) == {'ft300', 'xense'}


def test_startup_final_discard_failure_continues_other_sensor_and_fails(tmp_path):
    events = []
    instance = startup_controller(tmp_path, events)
    instance.ft_client.fail_commands.add('DEMO_DISCARD_REQ')
    instance._stop_runtime_resources = lambda: events.append(('controller', 'terminal_cleanup'))

    with pytest.raises(RuntimeError, match='ft300 DEMO_DISCARD_REQ failed'):
        instance.startup()

    assert instance.ft_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert instance.xense_client.commands == ['START_REQ', 'DEMO_DISCARD_REQ']
    assert set(instance._active_sensor_clients) == {'ft300'}
    assert instance.get_state() == InferenceState.STARTING_SERVICES


def test_concurrent_startup_fail_stop_waits_for_warmup_discard_before_stop(tmp_path):
    events = []
    instance = startup_controller(tmp_path, events)
    instance._execute_fail_stop = instance._stop_runtime_resources

    def start_lerobot():
        events.append(('lerobot', 'launch'))
        instance.control = StartupControl(events)
        instance.request_fail_stop('injected_startup_fatal', 'fatal during startup')

    instance._start_lerobot = start_lerobot

    with pytest.raises(RuntimeError, match='session termination began'):
        instance.startup()

    assert instance._termination_thread is not None
    instance._termination_thread.join(timeout=1.0)
    assert not instance._termination_thread.is_alive()
    ft_discard = events.index(('ft300', 'DEMO_DISCARD_REQ'))
    xense_discard = events.index(('xense', 'DEMO_DISCARD_REQ'))
    ft_stop = events.index(('ft300', 'STOP_REQ'))
    xense_stop = events.index(('xense', 'STOP_REQ'))
    assert ft_discard < xense_discard < ft_stop < xense_stop


def test_four_node_lines_are_only_launch_gate_and_images_are_authoritative(tmp_path):
    instance = controller(tmp_path)
    process = FakeProcess()
    process.log_path = tmp_path / 'realsense.log'
    process.log_path.write_text(
        ''.join(
            f'[{camera}.camera]: RealSense Node Is Up!\n'
            for camera in ('cam1', 'cam2', 'cam3', 'cam4')
        ),
        encoding='utf-8',
    )
    instance.processes['realsense'] = process
    instance.rosbag.image_readiness_ok = False

    assert instance._wait_realsense_nodes_up(start_position=0)
    with pytest.raises(RuntimeError, match='image topics are not ready'):
        instance._wait_realsense_images_ready()

    assert [call[0] for call in instance.rosbag.calls] == ['check_image_readiness']


def test_recoverable_fatal_during_image_readiness_restarts_and_regates(tmp_path):
    instance = controller(tmp_path)
    instance.config = config(
        tmp_path,
        realsense_startup_stabilization_s=0.001,
    )
    log_path = tmp_path / 'realsense.log'

    class RestartingProcess(FakeRestartableProcess):
        def __init__(self):
            super().__init__()
            self.log_path = log_path

        def restart(self, grace_s=5.0):
            super().restart(grace_s=grace_s)
            with self.log_path.open('a', encoding='utf-8') as log_fp:
                for camera in ('cam1', 'cam2', 'cam3', 'cam4'):
                    log_fp.write(f'[{camera}.camera]: RealSense Node Is Up!\n')

    process = RestartingProcess()
    process.log_path.write_text(
        ''.join(
            f'[{camera}.camera]: RealSense Node Is Up!\n'
            for camera in ('cam1', 'cam2', 'cam3', 'cam4')
        ),
        encoding='utf-8',
    )
    instance.processes['realsense'] = process
    instance.set_state(InferenceState.STARTING_SERVICES)
    readiness_calls = 0

    def check_image_readiness(requirements, timeout_s, mode):
        nonlocal readiness_calls
        readiness_calls += 1
        if readiness_calls == 1:
            instance._on_process_fatal(
                'realsense',
                'Hardware Notification:Depth stream start failure, Hardware Error',
            )
        ok = readiness_calls > 1
        return SimpleNamespace(
            ok=ok,
            to_manifest=lambda: {'ok': ok, 'mode': mode, 'required_topics': []},
        )

    instance.rosbag.check_image_readiness = check_image_readiness
    instance._wait_realsense_startup_ready()

    assert process.restart_count == 1
    assert readiness_calls == 2
    assert instance.termination_mode is None


def test_realsense_restart_node_gate_ignores_previous_generation_lines(tmp_path):
    instance = controller(tmp_path)
    instance.config = config(tmp_path, startup_timeout_s=0.01)
    process = FakeProcess()
    process.log_path = tmp_path / 'realsense.log'
    process.log_path.write_text(
        ''.join(
            f'[{camera}.camera]: RealSense Node Is Up!\n'
            for camera in ('cam1', 'cam2', 'cam3', 'cam4')
        ),
        encoding='utf-8',
    )
    restart_position = process.log_path.stat().st_size
    with process.log_path.open('a', encoding='utf-8') as log_fp:
        for camera in ('cam1', 'cam2', 'cam3'):
            log_fp.write(f'[{camera}.camera]: RealSense Node Is Up!\n')
    instance.processes['realsense'] = process

    with pytest.raises(RuntimeError, match=r"\['cam4'\]"):
        instance._wait_realsense_nodes_up(start_position=restart_position)


def test_failed_realsense_startup_recovery_escalates_to_fail_stop(tmp_path):
    instance = InferenceMainController(
        config(tmp_path, realsense_startup_stabilization_s=0.001)
    )
    process = FakeRestartableProcess(restart_error=RuntimeError('restart failed'))
    instance.processes['realsense'] = process
    instance._start_receivers = lambda: None
    instance._start_processes = lambda: None
    instance._wait_services_ready = lambda: instance._on_process_fatal(
        'realsense', 'Depth stream start failure, Hardware Error'
    )
    instance._execute_fail_stop = instance._finalize_session

    exit_code = instance.run()

    assert exit_code == 1
    assert process.restart_count == 1
    assert instance.termination_mode == 'FAIL_STOP'
    assert 'RealSense startup recovery attempt 1 failed: restart failed' in instance.termination_reason
    assert instance.get_state() == InferenceState.STOPPED


def test_non_realsense_required_process_fatal_still_requests_fail_stop(tmp_path):
    instance = controller(tmp_path)
    instance.set_state(InferenceState.STARTING_SERVICES)
    observed = []
    instance.request_fail_stop = lambda reason, message: observed.append((reason, message))

    instance._on_process_fatal('xense', 'runtime failure')

    assert observed == [('xense_fatal', 'runtime failure')]


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


def test_startup_does_not_create_resources_after_fail_stop_is_established(
    tmp_path,
    monkeypatch,
):
    instance = InferenceMainController(config(tmp_path))
    instance.termination_mode = 'FAIL_STOP'
    instance._fail_stop_requested.set()
    created = []

    def unexpected_process_creation(*args, **kwargs):
        created.append((args, kwargs))
        raise AssertionError('startup created a process after FAIL_STOP')

    monkeypatch.setattr(
        inference_controller_module,
        'ManagedProcess',
        unexpected_process_creation,
    )

    with pytest.raises(RuntimeError, match='session termination began'):
        instance._start_processes()
    assert created == []
    assert instance.processes == {}


def test_fatal_process_callback_during_startup_stops_progress_and_owned_process(
    tmp_path,
    monkeypatch,
):
    instance = InferenceMainController(config(tmp_path))
    created = []
    started = []
    callback_threads = []
    later_steps = []

    class FatalDuringStartProcess:
        def __init__(self, name, cmd, cwd, log_path, **kwargs):
            self.name = name
            self.cmd = cmd
            self.on_exit = kwargs['on_exit']
            self.stopped = False
            created.append(self)

        def start(self):
            started.append(self.name)
            if self.name == 'ft300':
                thread = threading.Thread(target=lambda: self.on_exit(self.name, 7))
                callback_threads.append(thread)
                thread.start()
                assert instance._fail_stop_requested.wait(timeout=1.0)

        def poll(self):
            return 0 if self.stopped else None

        def stop(self, grace_s=5.0):
            self.stopped = True

    monkeypatch.setattr(
        inference_controller_module,
        'ManagedProcess',
        FatalDuringStartProcess,
    )
    monkeypatch.setattr(instance, '_start_receivers', lambda: None)
    monkeypatch.setattr(instance, '_wait_services_ready', lambda: later_steps.append('services'))
    monkeypatch.setattr(instance, '_start_lerobot', lambda: later_steps.append('lerobot'))
    monkeypatch.setattr(instance, '_execute_fail_stop', instance._stop_runtime_resources)

    with pytest.raises(RuntimeError, match='session termination began during ft300 process'):
        instance.startup()

    for thread in callback_threads:
        thread.join(timeout=1.0)
    if instance._termination_thread is not None:
        instance._termination_thread.join(timeout=1.0)

    assert started == ['ft300']
    assert later_steps == []
    assert set(instance.processes.values()) == set(created)
    assert all(process.stopped for process in created)
    assert instance._resources_stopped


def test_keyboard_interrupt_requests_fail_stop_directly(tmp_path):
    instance = controller(tmp_path)

    def interrupt_startup():
        raise KeyboardInterrupt

    instance.startup = interrupt_startup
    exit_code = instance.run()

    assert exit_code == 1
    assert instance.termination_mode == 'FAIL_STOP'
    assert instance.termination_reason == 'keyboard_interrupt: KeyboardInterrupt/SIGINT received'
    assert instance.get_state() == InferenceState.STOPPED


def test_state_and_ready_messages_are_printed_to_console(tmp_path, capsys):
    instance = controller(tmp_path)

    instance.set_state(InferenceState.STARTING_SERVICES)
    instance._mark_session_ready()

    output = capsys.readouterr().out
    assert '[STARTING_SERVICES] [state] CREATED -> STARTING_SERVICES' in output
    assert '[WAIT_START] [state] STARTING_SERVICES -> WAIT_START' in output
    assert '[WAIT_START] Inference MainController ready.' in output


def test_fail_stop_prints_reason_to_stderr_only_once(tmp_path, capsys):
    instance = controller(tmp_path)
    instance._execute_fail_stop = lambda: None

    instance.request_fail_stop('test_failure', 'specific reason')
    assert instance._termination_thread is not None
    instance._termination_thread.join(timeout=1.0)
    instance.request_fail_stop('duplicate_failure', 'duplicate reason')

    error_output = capsys.readouterr().err
    assert error_output.count('FAIL_STOP: test_failure: specific reason') == 1
    assert 'duplicate reason' not in error_output


def test_graceful_shutdown_run_returns_success(tmp_path, monkeypatch):
    instance = controller(tmp_path)

    def admit_shutdown_during_startup():
        instance.set_state(InferenceState.WAIT_START)
        assert instance.admit_user_action('shutdown')

    instance.startup = admit_shutdown_during_startup
    monkeypatch.setattr(inference_controller_module.InferenceInputThread, 'start', lambda self: None)

    assert instance.run() == 0
    assert instance.termination_mode == 'SHUTDOWN'
    assert instance.get_state() == InferenceState.STOPPED


def test_main_propagates_fail_stop_as_nonzero_exit(monkeypatch):
    fake_controller = SimpleNamespace(run=lambda: 1)
    monkeypatch.setattr(inference_controller_module, 'parse_inference_args', lambda: object())
    monkeypatch.setattr(inference_controller_module, 'build_inference_config', lambda args: object())
    monkeypatch.setattr(
        inference_controller_module,
        'InferenceMainController',
        lambda config: fake_controller,
    )

    with pytest.raises(SystemExit) as exc_info:
        inference_controller_module.main()

    assert exc_info.value.code == 1


def test_main_preserves_graceful_success_exit(monkeypatch):
    fake_controller = SimpleNamespace(run=lambda: 0)
    monkeypatch.setattr(inference_controller_module, 'parse_inference_args', lambda: object())
    monkeypatch.setattr(inference_controller_module, 'build_inference_config', lambda args: object())
    monkeypatch.setattr(
        inference_controller_module,
        'InferenceMainController',
        lambda config: fake_controller,
    )

    assert inference_controller_module.main() is None
