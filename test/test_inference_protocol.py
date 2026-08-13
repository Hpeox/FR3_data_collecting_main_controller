from __future__ import annotations

import json
import mmap
import queue
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main_controller.aligned_health import (
    ABI_VERSION,
    GLOBAL_HEADER,
    GLOBAL_HEADER_SIZE,
    MAGIC,
    AlignedHealthReader,
)
from main_controller.inference_protocol import (
    Ack,
    ControlledClient,
    ControlledClientProtocolError,
    Status,
    make_command,
    parse_response,
)
from main_controller.inference_config import InferenceConfig


def response(**values) -> bytes:
    base = {'protocol_version': 1, **values}
    return json.dumps(base, separators=(',', ':')).encode()


def ack(sequence: int, operation: str, accepted: bool = True) -> bytes:
    return response(
        type='ACK', sequence=sequence, operation=operation, accepted=accepted,
        code='accepted' if accepted else 'invalid_phase', phase='WAIT_INITIALIZE',
        message='',
    )


def status(sequence: int, name: str, phase: str) -> bytes:
    return response(
        type='STATUS', sequence=sequence, status=name, phase=phase,
        code='ok', message='', timestamp_ns=time.time_ns(),
    )


class MemoryPacketEndpoint:
    def __init__(self):
        self.incoming = queue.Queue()
        self.peer = None
        self.closed = False

    def sendall(self, packet):
        if self.closed or self.peer is None:
            raise OSError('closed')
        self.peer.incoming.put(bytes(packet))

    def recv(self, _size):
        return self.incoming.get()

    def shutdown(self, _how):
        self.closed = True
        if self.peer is not None:
            self.peer.incoming.put(b'')

    def close(self):
        self.closed = True


def memory_packet_pair():
    first = MemoryPacketEndpoint()
    second = MemoryPacketEndpoint()
    first.peer = second
    second.peer = first
    return first, second


def test_protocol_strict_round_trip():
    command = json.loads(make_command(7, 'INITIALIZE'))
    assert command == {
        'protocol_version': 1,
        'type': 'COMMAND',
        'sequence': 7,
        'operation': 'INITIALIZE',
    }
    assert isinstance(parse_response(ack(7, 'INITIALIZE')), Ack)
    assert isinstance(parse_response(status(1, 'INITIALIZED', 'WAIT_START')), Status)
    malformed = json.loads(ack(7, 'INITIALIZE'))
    malformed['extra'] = True
    with pytest.raises(ControlledClientProtocolError, match='fields'):
        parse_response(json.dumps(malformed).encode())


def test_transaction_waits_for_completion_status_and_rejection_does_not():
    client_socket, server_socket = memory_packet_pair()
    client = ControlledClient('/unused')
    client._socket = client_socket
    client._receiver = threading.Thread(target=client._receive_loop, daemon=True)
    client._receiver.start()

    def worker():
        first = json.loads(server_socket.recv(1025))
        server_socket.sendall(ack(first['sequence'], 'INITIALIZE'))
        server_socket.sendall(status(1, 'INITIALIZING', 'INITIALIZING'))
        time.sleep(0.02)
        server_socket.sendall(status(2, 'INITIALIZED', 'WAIT_START'))
        second = json.loads(server_socket.recv(1025))
        server_socket.sendall(ack(second['sequence'], 'START', accepted=False))

    thread = threading.Thread(target=worker)
    thread.start()
    initialized = client.transact('INITIALIZE', {'INITIALIZED'})
    assert initialized.ack.accepted
    assert initialized.completion is not None
    assert initialized.completion.status == 'INITIALIZED'
    rejected = client.transact('START', {'STARTED'})
    assert not rejected.ack.accepted
    assert rejected.completion is None
    thread.join()
    client.close()
    server_socket.close()


def test_ordinary_lock_serializes_transactions():
    client_socket, server_socket = memory_packet_pair()
    client = ControlledClient('/unused')
    client._socket = client_socket
    client._receiver = threading.Thread(target=client._receive_loop, daemon=True)
    client._receiver.start()
    received = []

    def server():
        first = json.loads(server_socket.recv(1025))
        received.append(first['operation'])
        time.sleep(0.03)
        server_socket.sendall(ack(first['sequence'], first['operation']))
        completion = 'INITIALIZED' if first['operation'] == 'INITIALIZE' else 'STARTED'
        server_socket.sendall(status(1, completion, 'WAIT_START'))
        second = json.loads(server_socket.recv(1025))
        received.append(second['operation'])
        server_socket.sendall(ack(second['sequence'], second['operation']))
        completion = 'INITIALIZED' if second['operation'] == 'INITIALIZE' else 'STARTED'
        server_socket.sendall(status(2, completion, 'RUNNING'))

    server_thread = threading.Thread(target=server)
    server_thread.start()
    first = threading.Thread(target=lambda: client.transact('INITIALIZE', {'INITIALIZED'}))
    second = threading.Thread(target=lambda: client.transact('START', {'STARTED'}))
    first.start()
    time.sleep(0.005)
    second.start()
    first.join()
    second.join()
    server_thread.join()
    assert received == ['INITIALIZE', 'START']
    client.close()
    server_socket.close()


def test_aligned_health_reader_reads_header_only(tmp_path):
    shm_file = tmp_path / 'fr3_test'
    total_size = GLOBAL_HEADER_SIZE + 4096
    with shm_file.open('wb') as output:
        output.truncate(total_size)
    with shm_file.open('r+b') as output:
        mapping = mmap.mmap(output.fileno(), total_size)
        GLOBAL_HEADER.pack_into(
            mapping, 0, MAGIC, ABI_VERSION, 1, GLOBAL_HEADER_SIZE, 160,
            total_size, 2, 640, 480, 4, 2048, 42, 1, 9,
            b'sensor fatal\0' + b'\0' * 235,
        )
        mapping.close()
    with AlignedHealthReader('/fr3_test', shm_root=tmp_path) as reader:
        health = reader.read()
    assert health.ready
    assert health.latest_sequence == 42
    assert health.fatal
    assert health.status_code == 9
    assert health.message == 'sensor fatal'


def test_inference_config_enforces_watchdog_order_and_no_reset_target(tmp_path):
    for relative in ('FT300S', 'XenseTacSensor', 'RealSense/launch'):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    for relative in (
        'RealSense/launch/four_realsense_640x480_30.launch.py',
        'RealSense/launch/rosbag2_recorder.launch.py',
    ):
        (tmp_path / relative).touch()
    config = InferenceConfig(policy_path='policy', task='task', repo_root=tmp_path)
    command = config.lerobot_command()
    assert command[:5] == ['conda', 'run', '-n', 'lerobot-fr3-312', 'lerobot-rollout']
    assert '--strategy.type=controlled' in command
    assert not any('q_reset' in item or 'rollout_init_delta' in item for item in command)
    with pytest.raises(ValueError, match='shorter'):
        InferenceConfig(
            policy_path='policy', task='task', repo_root=tmp_path,
            aligned_stall_timeout_s=0.1, lerobot_aligned_max_age_ms=100,
        )
