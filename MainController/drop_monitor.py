"""Drop and timing-gap monitoring helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DropWarning:
    """A detected stream continuity or interval warning."""

    stream: str
    reason: str
    previous_key: int | None
    current_key: int | None
    previous_stamp_ns: int | None
    current_stamp_ns: int | None
    interval_ns: int | None
    expected_interval_ns: int
    warning_interval_ns: int


class DropMonitor:
    """Track one stream's frame/sequence continuity and timing gaps."""

    def __init__(self, stream: str, expected_interval_ns: int, warning_interval_ns: int):
        self.stream = stream
        self.expected_interval_ns = expected_interval_ns
        self.warning_interval_ns = warning_interval_ns
        self.previous_key: int | None = None
        self.previous_stamp_ns: int | None = None
        self.warning_count = 0
        self.missing_frame_count = 0
        self.max_interval_ns = 0

    def reset_baseline(self) -> None:
        """Reset baseline after pause/resume or a known stream restart."""
        self.previous_key = None
        self.previous_stamp_ns = None

    def observe(self, key: int | None, stamp_ns: int | None) -> list[DropWarning]:
        """Observe one sample and return warnings, if any."""
        warnings: list[DropWarning] = []
        prev_key = self.previous_key
        prev_stamp = self.previous_stamp_ns

        if key is not None and prev_key is not None and key != prev_key + 1:
            self.missing_frame_count += max(0, key - prev_key - 1)
            warnings.append(self._warning('non_contiguous_key', prev_key, key, prev_stamp, stamp_ns, None))

        if stamp_ns is not None and prev_stamp is not None:
            interval = stamp_ns - prev_stamp
            self.max_interval_ns = max(self.max_interval_ns, interval)
            if interval > self.warning_interval_ns:
                warnings.append(self._warning('large_interval', prev_key, key, prev_stamp, stamp_ns, interval))

        if key is not None:
            self.previous_key = key
        if stamp_ns is not None:
            self.previous_stamp_ns = stamp_ns

        self.warning_count += len(warnings)
        return warnings

    def summary(self) -> dict[str, int | str]:
        """Return a manifest-friendly monitor summary."""
        return {
            'stream': self.stream,
            'warning_count': self.warning_count,
            'missing_frame_count': self.missing_frame_count,
            'max_interval_ns': self.max_interval_ns,
            'expected_interval_ns': self.expected_interval_ns,
            'warning_interval_ns': self.warning_interval_ns,
        }

    def _warning(
        self,
        reason: str,
        prev_key: int | None,
        key: int | None,
        prev_stamp: int | None,
        stamp_ns: int | None,
        interval: int | None,
    ) -> DropWarning:
        return DropWarning(
            stream=self.stream,
            reason=reason,
            previous_key=prev_key,
            current_key=key,
            previous_stamp_ns=prev_stamp,
            current_stamp_ns=stamp_ns,
            interval_ns=interval,
            expected_interval_ns=self.expected_interval_ns,
            warning_interval_ns=self.warning_interval_ns,
        )
