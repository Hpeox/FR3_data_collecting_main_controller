"""ZMQ telemetry binary protocol and receiver."""

from __future__ import annotations

import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import zmq


MAGIC = b'FGT1'
VERSION = 1
GELLO_FLOAT_COUNT = 8
ROBOT_FLOAT_COUNT = 50
FLOAT_COUNT = GELLO_FLOAT_COUNT + ROBOT_FLOAT_COUNT
FRAME_STRUCT = struct.Struct('<4sBBHQdQ58dBB6x')
FRAME_SIZE = FRAME_STRUCT.size


@dataclass(frozen=True)
class TelemetryFrame:
    """Decoded telemetry frame from the remote relay."""

    source: int
    seq: int
    stamp: float
    valid_mask: int
    floats_58: tuple[float, ...]
    gripper_gPO: int
    gripper_gCU: int


def unpack_frame(payload: bytes) -> TelemetryFrame:
    """Decode one telemetry payload."""
    if len(payload) != FRAME_SIZE:
        raise ValueError(f'expected telemetry frame size {FRAME_SIZE}, got {len(payload)}')
    unpacked = FRAME_STRUCT.unpack(payload)
    magic, version, source, _flags, seq, stamp, valid_mask = unpacked[:7]
    if magic != MAGIC:
        raise ValueError(f'invalid telemetry magic {magic!r}')
    if version != VERSION:
        raise ValueError(f'unsupported telemetry version {version}')
    floats = tuple(float(item) for item in unpacked[7 : 7 + FLOAT_COUNT])
    return TelemetryFrame(
        source=int(source),
        seq=int(seq),
        stamp=float(stamp),
        valid_mask=int(valid_mask),
        floats_58=floats,
        gripper_gPO=int(unpacked[-2]),
        gripper_gCU=int(unpacked[-1]),
    )


class ZmqTelemetryReceiver:
    """Always-on ZMQ receiver that drains telemetry from the remote relay."""

    def __init__(
        self,
        endpoint: str,
        on_frame: Callable[[TelemetryFrame, int, int], None],
        on_error: Callable[[str], None] | None = None,
        on_fatal: Callable[[str], None] | None = None,
        rcv_hwm: int = 1000,
        context: zmq.Context | None = None,
        destroy_context_on_stop: bool = False,
    ):
        self.endpoint = endpoint
        self.on_frame = on_frame
        self.on_error = on_error
        self.on_fatal = on_fatal
        self.rcv_hwm = rcv_hwm
        self._context = context if context is not None else zmq.Context.instance()
        self._destroy_context_on_stop = destroy_context_on_stop
        self.first_frame = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the receiver thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name='ZmqTelemetryReceiver', daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the receiver thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def wait_first_frame(self, timeout_s: float) -> bool:
        """Wait until at least one valid frame is decoded."""
        return self.first_frame.wait(timeout=timeout_s)

    def _run(self) -> None:
        socket = self._context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        socket.setsockopt(zmq.RCVHWM, self.rcv_hwm)
        poller = zmq.Poller()
        try:
            socket.connect(self.endpoint)
            poller.register(socket, zmq.POLLIN)
            while not self._stop.is_set():
                events = dict(poller.poll(250))
                if socket not in events:
                    continue
                payload = socket.recv()
                recv_time_ns = time.time_ns()
                recv_monotonic_ns = time.monotonic_ns()
                try:
                    frame = unpack_frame(payload)
                except Exception as exc:
                    if self.on_error is not None:
                        self.on_error(f'invalid ZMQ frame: {exc}')
                    continue
                self.first_frame.set()
                self.on_frame(frame, recv_time_ns, recv_monotonic_ns)
        except Exception as exc:
            if not self._stop.is_set():
                message = f'ZMQ receiver failed: {exc}'
                if self.on_fatal is not None:
                    self.on_fatal(message)
                elif self.on_error is not None:
                    self.on_error(message)
        finally:
            try:
                poller.unregister(socket)
            except Exception:
                pass
            socket.close(linger=0)
            if self._destroy_context_on_stop:
                self._context.destroy(linger=0)
