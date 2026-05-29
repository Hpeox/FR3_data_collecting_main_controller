"""Subprocess management for sensor and ROS launch processes."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path


class ManagedProcess:
    """A subprocess with log capture and fatal-pattern detection."""

    def __init__(
        self,
        name: str,
        cmd: list[str],
        cwd: Path,
        log_path: Path,
        fatal_patterns: tuple[str, ...] = (),
        on_fatal: Callable[[str, str], None] | None = None,
        on_exit: Callable[[str, int], None] | None = None,
    ):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.log_path = log_path
        self.fatal_patterns = fatal_patterns
        self.on_fatal = on_fatal
        self.on_exit = on_exit
        self.process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the subprocess."""
        if self.process is not None and self.process.poll() is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = self.log_path.open('a', encoding='utf-8')
        self.process = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        log_fp.write(f'[start] pid={self.process.pid} cmd={self.cmd}\n')
        log_fp.flush()
        self._reader_thread = threading.Thread(
            target=self._read_output,
            args=(log_fp,),
            name=f'ProcessLog:{self.name}',
            daemon=True,
        )
        self._reader_thread.start()

    def stop(self, grace_s: float = 5.0) -> None:
        """Stop the process with SIGINT, then stronger signals if needed."""
        proc = self.process
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def restart(self, grace_s: float = 5.0) -> None:
        """Stop and start the process using the same command."""
        self.stop(grace_s=grace_s)
        time.sleep(0.5)
        self.start()

    def poll(self) -> int | None:
        """Return subprocess poll result."""
        if self.process is None:
            return None
        return self.process.poll()

    def _read_output(self, log_fp) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                log_fp.write(line)
                log_fp.flush()
                if self.on_fatal is not None and any(pattern in line for pattern in self.fatal_patterns):
                    self.on_fatal(self.name, line.rstrip())
            rc = self.process.wait()
            if self.on_exit is not None:
                self.on_exit(self.name, rc)
        finally:
            log_fp.close()


def bash_cmd(script: str) -> list[str]:
    """Return a bash -lc command."""
    return ['bash', '-lc', script]
