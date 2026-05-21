"""Unix domain socket client for FT300S and XenseTacSensor services."""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


MAGIC = b'F3'
HEADER_FMT = '<2sBBHIq'
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class MsgType(IntEnum):
    """Message type identifiers shared with sensor services."""

    INIT_REQ = 1
    INIT_READY = 2
    START_REQ = 3
    FRAME_READY = 4
    PAUSE_REQ = 5
    STOP_REQ = 6
    ACK = 7
    ERROR = 8
    DEMO_DONE_REQ = 9
    DEMO_DISCARD_REQ = 10


@dataclass(frozen=True)
class UdsEvent:
    """One decoded UDS event from a sensor service."""

    client_name: str
    msg_type: MsgType
    frame_id: int
    payload: dict[str, Any]
    recv_time_ns: int
    recv_monotonic_ns: int


def pack_message(
    msg_type: MsgType,
    frame_id: int = -1,
    payload: dict[str, Any] | None = None,
    version: int = 1,
    flags: int = 0,
) -> bytes:
    """Serialize a UDS protocol message."""
    payload_bytes = b''
    if payload is not None:
        payload_bytes = json.dumps(payload, ensure_ascii=True).encode('utf-8')
    header = struct.pack(HEADER_FMT, MAGIC, version, int(msg_type), flags, len(payload_bytes), int(frame_id))
    return header + payload_bytes


def unpack_header(header_bytes: bytes) -> tuple[int, MsgType, int, int, int]:
    """Deserialize and validate a UDS header."""
    magic, version, msg_type, flags, payload_len, frame_id = struct.unpack(HEADER_FMT, header_bytes)
    if magic != MAGIC:
        raise ValueError('invalid UDS magic')
    return int(version), MsgType(msg_type), int(flags), int(payload_len), int(frame_id)


def decode_payload(payload_bytes: bytes) -> dict[str, Any]:
    """Decode a JSON payload."""
    if not payload_bytes:
        return {}
    return json.loads(payload_bytes.decode('utf-8'))


class UdsClient:
    """Reconnecting UDS client with background receive loop."""

    def __init__(
        self,
        name: str,
        socket_path: str,
        on_event: Callable[[UdsEvent], None],
        protocol_version: int = 1,
        retry_interval_s: float = 0.5,
        recv_timeout_s: float = 0.2,
    ):
        self.name = name
        self.socket_path = socket_path
        self.on_event = on_event
        self.protocol_version = protocol_version
        self.retry_interval_s = retry_interval_s
        self.recv_timeout_s = recv_timeout_s
        self.init_ready = threading.Event()
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._ack_lock = threading.Lock()
        self._ack_events: dict[str, threading.Event] = {}
        self._ack_payloads: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        """Start background connection and receive handling."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=f'UdsClient:{self.name}', daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the client and close the socket."""
        self._stop.set()
        self._close_socket()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def wait_connected(self, timeout_s: float) -> bool:
        """Wait for a socket connection."""
        return self._connected.wait(timeout=timeout_s)

    def send_msg(self, msg_type: MsgType, frame_id: int = -1, payload: dict[str, Any] | None = None) -> bool:
        """Send one message if connected."""
        data = pack_message(msg_type, frame_id=frame_id, payload=payload, version=self.protocol_version)
        with self._send_lock:
            sock = self._sock
            if sock is None:
                return False
            try:
                sock.sendall(data)
                return True
            except OSError:
                self._mark_disconnected()
                return False

    def send_and_wait_ack(
        self,
        msg_type: MsgType,
        cmd_name: str,
        timeout_s: float | None,
        progress_period_s: float = 5.0,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict[str, Any] | None:
        """Send a command and wait for ACK(cmd). timeout_s=None means no hard timeout."""
        event = self._ack_event(cmd_name)
        event.clear()
        with self._ack_lock:
            self._ack_payloads.pop(cmd_name, None)

        if not self.send_msg(msg_type):
            return None

        start = time.monotonic()
        while not self._stop.is_set():
            wait_s = progress_period_s
            if timeout_s is not None:
                remaining = timeout_s - (time.monotonic() - start)
                if remaining <= 0:
                    return None
                wait_s = min(wait_s, remaining)
            if event.wait(timeout=wait_s):
                with self._ack_lock:
                    return self._ack_payloads.get(cmd_name, {})
            if on_progress is not None:
                on_progress(time.monotonic() - start)
        return None

    def wait_init_ready(self, timeout_s: float) -> bool:
        """Send INIT_REQ and wait for INIT_READY."""
        if self.init_ready.is_set():
            return True
        self.send_msg(MsgType.INIT_REQ)
        return self.init_ready.wait(timeout=timeout_s)

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._ensure_connected():
                time.sleep(self.retry_interval_s)
                continue
            try:
                header = self._recv_exact(HEADER_SIZE)
                version, msg_type, _flags, payload_len, frame_id = unpack_header(header)
                if version != self.protocol_version:
                    raise ValueError(f'protocol version mismatch: {version}')
                payload_bytes = self._recv_exact(payload_len) if payload_len else b''
                event = UdsEvent(
                    client_name=self.name,
                    msg_type=msg_type,
                    frame_id=frame_id,
                    payload=decode_payload(payload_bytes),
                    recv_time_ns=time.time_ns(),
                    recv_monotonic_ns=time.monotonic_ns(),
                )
                self._handle_event(event)
                self.on_event(event)
            except socket.timeout:
                continue
            except Exception:
                self._mark_disconnected()
                time.sleep(self.retry_interval_s)

    def _ensure_connected(self) -> bool:
        if self._sock is not None:
            return True
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.recv_timeout_s)
        try:
            sock.connect(self.socket_path)
        except OSError:
            sock.close()
            return False
        with self._state_lock:
            self._sock = sock
            self._connected.set()
        return True

    def _recv_exact(self, size: int) -> bytes:
        sock = self._sock
        if sock is None:
            raise ConnectionError('not connected')
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ConnectionError('UDS peer closed')
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def _handle_event(self, event: UdsEvent) -> None:
        if event.msg_type == MsgType.INIT_READY:
            self.init_ready.set()
            return
        if event.msg_type != MsgType.ACK:
            return
        cmd = event.payload.get('cmd')
        if not isinstance(cmd, str):
            return
        ack_event = self._ack_event(cmd)
        with self._ack_lock:
            self._ack_payloads[cmd] = event.payload
        ack_event.set()

    def _ack_event(self, cmd_name: str) -> threading.Event:
        with self._ack_lock:
            event = self._ack_events.get(cmd_name)
            if event is None:
                event = threading.Event()
                self._ack_events[cmd_name] = event
            return event

    def _mark_disconnected(self) -> None:
        self._connected.clear()
        self._close_socket()

    def _close_socket(self) -> None:
        with self._state_lock:
            sock = self._sock
            self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
