"""Client-side protocol for the persistent LeRobot Controlled worker."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1
MAX_PACKET_SIZE = 1024
ORDINARY_OPERATIONS = frozenset({'INITIALIZE', 'START', 'STOP', 'ABORT'})
TERMINATION_OPERATIONS = frozenset({'SHUTDOWN', 'FAIL_STOP'})
OPERATIONS = ORDINARY_OPERATIONS | TERMINATION_OPERATIONS
FATAL_STATUSES = frozenset({'ERROR'})


class ControlledClientError(RuntimeError):
    """Base error for the LeRobot control channel."""


class ControlledClientDisconnected(ControlledClientError):
    """Raised when the persistent worker control channel disconnects."""


class ControlledClientProtocolError(ControlledClientError):
    """Raised when LeRobot sends a malformed response."""


@dataclass(frozen=True)
class Ack:
    """One validated command acknowledgement."""

    sequence: int
    operation: str
    accepted: bool
    code: str
    phase: str
    message: str


@dataclass(frozen=True)
class Status:
    """One validated lifecycle status."""

    sequence: int
    status: str
    phase: str
    code: str
    message: str
    timestamp_ns: int


@dataclass(frozen=True)
class TransactionResult:
    """Result of one ordinary lifecycle transaction."""

    ack: Ack
    completion: Status | None


def make_command(sequence: int, operation: str) -> bytes:
    """Encode one strict Controlled UDS command."""
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError('sequence must be a non-negative integer')
    if operation not in OPERATIONS:
        raise ValueError(f'unsupported operation: {operation!r}')
    payload = {
        'protocol_version': PROTOCOL_VERSION,
        'type': 'COMMAND',
        'sequence': sequence,
        'operation': operation,
    }
    encoded = json.dumps(payload, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    if len(encoded) > MAX_PACKET_SIZE:
        raise ValueError('Controlled UDS command is too large')
    return encoded


def parse_response(packet: bytes) -> Ack | Status:
    """Decode and strictly validate one ACK or STATUS response."""
    if len(packet) > MAX_PACKET_SIZE:
        raise ControlledClientProtocolError('Controlled UDS response is too large')
    try:
        value = json.loads(packet.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlledClientProtocolError(f'invalid Controlled UDS JSON: {exc}') from exc
    if not isinstance(value, dict) or value.get('protocol_version') != PROTOCOL_VERSION:
        raise ControlledClientProtocolError('unsupported Controlled UDS response')
    response_type = value.get('type')
    if response_type == 'ACK':
        required = {
            'protocol_version', 'type', 'sequence', 'operation', 'accepted',
            'code', 'phase', 'message',
        }
        if set(value) != required:
            raise ControlledClientProtocolError('Controlled ACK fields are invalid')
        _validate_nonnegative_int(value['sequence'], 'ACK sequence')
        if value['operation'] not in OPERATIONS:
            raise ControlledClientProtocolError('Controlled ACK operation is invalid')
        if not isinstance(value['accepted'], bool):
            raise ControlledClientProtocolError('Controlled ACK accepted must be bool')
        _validate_strings(value, ('code', 'phase', 'message'), 'ACK')
        return Ack(
            sequence=value['sequence'],
            operation=value['operation'],
            accepted=value['accepted'],
            code=value['code'],
            phase=value['phase'],
            message=value['message'],
        )
    if response_type == 'STATUS':
        required = {
            'protocol_version', 'type', 'sequence', 'status', 'phase',
            'code', 'message', 'timestamp_ns',
        }
        if set(value) != required:
            raise ControlledClientProtocolError('Controlled STATUS fields are invalid')
        _validate_nonnegative_int(value['sequence'], 'STATUS sequence')
        _validate_nonnegative_int(value['timestamp_ns'], 'STATUS timestamp_ns')
        _validate_strings(value, ('status', 'phase', 'code', 'message'), 'STATUS')
        return Status(
            sequence=value['sequence'],
            status=value['status'],
            phase=value['phase'],
            code=value['code'],
            message=value['message'],
            timestamp_ns=value['timestamp_ns'],
        )
    raise ControlledClientProtocolError('Controlled response must be ACK or STATUS')


def _validate_nonnegative_int(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ControlledClientProtocolError(f'{field} must be a non-negative integer')


def _validate_strings(value: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    if any(not isinstance(value[field], str) for field in fields):
        raise ControlledClientProtocolError(f'Controlled {label} string fields are invalid')


class ControlledClient:
    """Thread-safe response-multiplexing client for one persistent worker.

    A receive thread is used on the MainController side so a session-level
    ``FAIL_STOP`` intent can be transmitted while an ordinary synchronous
    operation is still executing in LeRobot. Ordinary transactions remain
    serialized by ``_ordinary_lock``.
    """

    def __init__(
        self,
        socket_path: str,
        *,
        on_status: Callable[[Status], None] | None = None,
        on_disconnect: Callable[[BaseException], None] | None = None,
    ):
        self.socket_path = socket_path
        self.on_status = on_status
        self.on_disconnect = on_disconnect
        self._socket: socket.socket | None = None
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._ordinary_lock = threading.Lock()
        self._condition = threading.Condition()
        self._acks: dict[int, Ack] = {}
        self._statuses: list[Status] = []
        self._stop = threading.Event()
        self._disconnected = threading.Event()
        self._error: BaseException | None = None
        self._receiver: threading.Thread | None = None

    @property
    def disconnected(self) -> bool:
        """Return whether the channel has terminated."""
        return self._disconnected.is_set()

    def connect(self, timeout_s: float) -> None:
        """Connect to a worker socket that may appear during process startup."""
        if self._socket is not None:
            return
        deadline = time.monotonic() + timeout_s
        last_error: OSError | None = None
        while time.monotonic() < deadline and not self._stop.is_set():
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            try:
                sock.connect(self.socket_path)
            except OSError as exc:
                last_error = exc
                sock.close()
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
                continue
            self._socket = sock
            self._receiver = threading.Thread(
                target=self._receive_loop,
                name='LeRobotControlReceiver',
                daemon=True,
            )
            self._receiver.start()
            return
        raise TimeoutError(
            f'LeRobot control socket did not connect: {self.socket_path}: {last_error}'
        )

    def close(self) -> None:
        """Close the client and wake every waiter."""
        self._stop.set()
        sock, self._socket = self._socket, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        with self._condition:
            self._condition.notify_all()
        receiver, self._receiver = self._receiver, None
        if receiver is not None and receiver is not threading.current_thread():
            receiver.join(timeout=2.0)

    def wait_for_status(self, expected: set[str], *, after_index: int = 0) -> Status:
        """Wait without a protocol liveness timeout for a matching status."""
        with self._condition:
            index = after_index
            while True:
                for item in self._statuses[index:]:
                    if item.status in FATAL_STATUSES:
                        raise ControlledClientError(
                            f'LeRobot reported {item.status}: {item.code}: {item.message}'
                        )
                    if item.status in expected:
                        return item
                index = len(self._statuses)
                self._raise_if_disconnected()
                self._condition.wait()

    def status_count(self) -> int:
        """Return the current receive-order status count."""
        with self._condition:
            return len(self._statuses)

    def transact(self, operation: str, completion_statuses: set[str]) -> TransactionResult:
        """Execute one serialized ordinary REQUEST -> ACK -> STATUS transaction."""
        if operation not in ORDINARY_OPERATIONS:
            raise ValueError(f'{operation} is not an ordinary operation')
        with self._ordinary_lock:
            after_index = self.status_count()
            sequence = self.send(operation)
            ack = self.wait_for_ack(sequence)
            if not ack.accepted:
                return TransactionResult(ack=ack, completion=None)
            completion = self.wait_for_status(completion_statuses, after_index=after_index)
            return TransactionResult(ack=ack, completion=completion)

    def send(self, operation: str) -> int:
        """Send one command using a newly allocated monotonically increasing sequence."""
        with self._sequence_lock:
            self._sequence += 1
            sequence = self._sequence
        packet = make_command(sequence, operation)
        sock = self._socket
        if sock is None:
            raise ControlledClientDisconnected('LeRobot control socket is not connected')
        try:
            with self._send_lock:
                sock.sendall(packet)
        except OSError as exc:
            self._mark_disconnected(ControlledClientDisconnected(f'LeRobot control send failed: {exc}'))
            raise ControlledClientDisconnected(f'LeRobot control send failed: {exc}') from exc
        return sequence

    def wait_for_ack(self, sequence: int, timeout_s: float | None = None) -> Ack:
        """Wait for one ACK; ``None`` intentionally means no liveness timeout."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            while sequence not in self._acks:
                self._raise_if_disconnected()
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'LeRobot ACK timed out for sequence {sequence}')
                self._condition.wait(timeout=remaining)
            return self._acks[sequence]

    def ack_if_received(self, sequence: int) -> Ack | None:
        """Return an ACK without waiting."""
        with self._condition:
            return self._acks.get(sequence)

    def statuses(self) -> tuple[Status, ...]:
        """Return a receive-order status snapshot."""
        with self._condition:
            return tuple(self._statuses)

    def _receive_loop(self) -> None:
        try:
            while not self._stop.is_set():
                sock = self._socket
                if sock is None:
                    return
                packet = sock.recv(MAX_PACKET_SIZE + 1)
                if not packet:
                    raise ControlledClientDisconnected('LeRobot control socket disconnected')
                response = parse_response(packet)
                callback_status: Status | None = None
                with self._condition:
                    if isinstance(response, Ack):
                        self._acks[response.sequence] = response
                    else:
                        self._statuses.append(response)
                        callback_status = response
                    self._condition.notify_all()
                if callback_status is not None and self.on_status is not None:
                    self.on_status(callback_status)
        except BaseException as exc:
            if not self._stop.is_set():
                self._mark_disconnected(exc)

    def _mark_disconnected(self, exc: BaseException) -> None:
        first = not self._disconnected.is_set()
        self._error = exc
        self._disconnected.set()
        with self._condition:
            self._condition.notify_all()
        if first and self.on_disconnect is not None:
            self.on_disconnect(exc)

    def _raise_if_disconnected(self) -> None:
        if self._disconnected.is_set():
            if isinstance(self._error, BaseException):
                raise ControlledClientDisconnected(str(self._error)) from self._error
            raise ControlledClientDisconnected('LeRobot control socket disconnected')

