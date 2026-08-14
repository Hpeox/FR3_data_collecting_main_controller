"""Read only the FR3 aligned-observation global SHM header."""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path


MAGIC = b'FR3OBS2\0'
ABI_VERSION = 2
GLOBAL_HEADER_SIZE = 320
GLOBAL_HEADER = struct.Struct('<8sIIIIQIIIIQQII248s')


@dataclass(frozen=True)
class AlignedHealth:
    """Header-level health required by MainController."""

    ready: bool
    latest_sequence: int
    fatal: bool
    status_code: int
    message: str


class AlignedHealthReader:
    """Version-pinned read-only representation of the stable FR3OBS2 ABI.

    Keeping this intentional process-boundary representation in a small module
    prevents the rollout watchdog from copying image and tactile payloads or
    importing policy/robot implementation state.
    """

    def __init__(self, shm_name: str, *, shm_root: Path = Path('/dev/shm')):
        clean = shm_name.removeprefix('/')
        if not clean or '/' in clean:
            raise ValueError('aligned observation SHM name must be simple')
        self.shm_name = shm_name
        self.path = shm_root / clean
        self._fd = os.open(self.path, os.O_RDONLY)
        self._size = os.fstat(self._fd).st_size
        if self._size < GLOBAL_HEADER_SIZE:
            os.close(self._fd)
            raise ValueError('aligned observation SHM is smaller than its global header')
        self._mapping = mmap.mmap(
            self._fd,
            GLOBAL_HEADER_SIZE,
            access=mmap.ACCESS_READ,
        )
        try:
            self.read()
        except Exception:
            self.close()
            raise

    def read(self) -> AlignedHealth:
        """Return one global-header snapshot without accessing any slot payload."""
        fields = GLOBAL_HEADER.unpack_from(self._mapping, 0)
        magic, version, ready, global_size, slot_header_size, total_size = fields[:6]
        if magic != MAGIC or version != ABI_VERSION or global_size != GLOBAL_HEADER_SIZE:
            raise ValueError('aligned observation SHM ABI is unsupported')
        slot_count, width, height, camera_count, slot_stride = fields[6:11]
        if (
            total_size != self._size
            or slot_count != 2
            or width != 640
            or height != 480
            or camera_count <= 0
            or slot_header_size <= 0
            or slot_stride <= slot_header_size
        ):
            raise ValueError('aligned observation SHM metadata is invalid')
        latest_sequence = fields[11]
        fatal = fields[12]
        status_code = fields[13]
        message = fields[14].split(b'\0', 1)[0].decode('utf-8', errors='replace')
        return AlignedHealth(
            ready=ready == 1,
            latest_sequence=latest_sequence,
            fatal=fatal != 0,
            status_code=status_code,
            message=message,
        )

    def close(self) -> None:
        """Release the read-only mapping."""
        mapping, self._mapping = getattr(self, '_mapping', None), None
        if mapping is not None:
            mapping.close()
        fd, self._fd = getattr(self, '_fd', None), None
        if fd is not None:
            os.close(fd)

    def __enter__(self) -> AlignedHealthReader:
        return self

    def __exit__(self, *_args) -> None:
        self.close()
