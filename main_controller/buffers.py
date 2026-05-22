"""Session logging and demo buffer persistence."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class JsonlLogger:
    """Thread-safe JSONL logger for low-rate controller events."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open('a', encoding='utf-8')
        self._lock = threading.Lock()

    def event(self, event_type: str, **payload: Any) -> None:
        """Append one timestamped event."""
        entry = {'time_ns': time.time_ns(), 'event': event_type, **payload}
        line = json.dumps(entry, ensure_ascii=True, separators=(',', ':'))
        with self._lock:
            self._fp.write(line + '\n')
            self._fp.flush()

    def close(self) -> None:
        """Close the underlying file."""
        with self._lock:
            self._fp.close()


class TableBuffer:
    """Simple list-backed table buffer saved as a NumPy archive."""

    def __init__(self, fields: tuple[str, ...]):
        self.fields = fields
        self._rows: dict[str, list[Any]] = {field: [] for field in fields}
        self._lock = threading.Lock()

    def append(self, **values: Any) -> None:
        """Append one row, filling missing fields with None."""
        with self._lock:
            for field in self.fields:
                self._rows[field].append(values.get(field))

    def __len__(self) -> int:
        return len(self._rows[self.fields[0]])

    def clear(self) -> None:
        """Clear all buffered rows."""
        with self._lock:
            for values in self._rows.values():
                values.clear()

    def save_npz(self, path: Path) -> Path:
        """Persist buffered rows to an uncompressed .npz file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            arrays = {field: np.asarray(values) for field, values in self._rows.items()}
        np.savez(path, **arrays)
        return path

    def snapshot_lengths(self) -> dict[str, int]:
        """Return field lengths for diagnostics."""
        with self._lock:
            return {field: len(values) for field, values in self._rows.items()}


@dataclass
class DemoStore:
    """Buffers for one active demo."""

    demo_dir: Path
    ft300: TableBuffer = field(default_factory=lambda: TableBuffer(('frame_id', 'timestamp_ns', 'recv_time_ns', 'recv_monotonic_ns')))
    xense: TableBuffer = field(default_factory=lambda: TableBuffer(('frame_id', 'timestamp_ns_0', 'timestamp_ns_1', 'recv_time_ns', 'recv_monotonic_ns')))
    realsense: TableBuffer = field(default_factory=lambda: TableBuffer(('topic', 'frame_number', 'header_stamp_ns', 'frame_timestamp_ns', 'hw_timestamp_ns', 'recv_time_ns', 'recv_monotonic_ns')))
    zmq: TableBuffer = field(default_factory=lambda: TableBuffer(('source', 'seq', 'stamp_s', 'valid_mask', 'floats_58', 'gripper_gPO', 'gripper_gCU', 'recv_time_ns', 'recv_monotonic_ns')))

    def save_all(self) -> dict[str, str]:
        """Save every high-rate buffer and return file paths."""
        paths = {
            'ft300': self.ft300.save_npz(self.demo_dir / 'ft300_timestamps.npz'),
            'xense': self.xense.save_npz(self.demo_dir / 'xense_timestamps.npz'),
            'realsense': self.realsense.save_npz(self.demo_dir / 'realsense_metadata.npz'),
            'zmq': self.zmq.save_npz(self.demo_dir / 'zmq_telemetry.npz'),
        }
        return {name: str(path) for name, path in paths.items()}

    def frame_counts(self) -> dict[str, int]:
        """Return primary row counts for each stream buffer."""
        return {
            'ft300': len(self.ft300),
            'xense': len(self.xense),
            'realsense': len(self.realsense),
            'zmq': len(self.zmq),
        }

    def write_manifest(self, payload: dict[str, Any]) -> Path:
        """Write the demo manifest."""
        path = self.demo_dir / 'manifest.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')
        return path
