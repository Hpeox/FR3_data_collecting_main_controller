"""Configuration for the persistent FR3 inference MainController."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from .config import XENSE_SDK_CONDA_ENVS, validate_repo_root


EXPECTED_REALSENSE_SHM_NAMES = (
    '/realsense_cam1', '/realsense_cam2', '/realsense_cam3', '/realsense_cam4',
)


def default_control_socket_path() -> str:
    """Return the per-user LeRobot Controlled socket path."""
    return f'/run/user/{os.getuid()}/lerobot_controlled.sock'


@dataclass(frozen=True)
class InferenceConfig:
    """Session-scoped configuration with explicit watchdog ordering."""

    policy_path: str
    task: str
    repo_root: Path
    runtime_root: Path | None = None
    runtime_sessions_dir: Path = field(init=False)
    runtime_frames_dir: Path = field(init=False)
    control_socket_path: str = field(default_factory=default_control_socket_path)
    observation_shm_name: str = '/fr3_aligned_observation'
    realsense_shm_names: tuple[str, ...] = EXPECTED_REALSENSE_SHM_NAMES
    zmq_connect: str = 'tcp://192.168.1.37:6000'
    robot_command_endpoint: str = 'tcp://192.168.1.37:6001'
    robot_telemetry_endpoint: str = 'tcp://192.168.1.37:6000'
    ft_uds_path: str = '/tmp/ft300_sensor.sock'
    xense_uds_path: str = '/tmp/xense_sensor.sock'
    ft_shm_name: str = 'ft300_sensor_frame'
    xense_shm_name: str = 'xense_sensor_frame'
    ft_fps: float = 100.0
    xense_fps: float = 30.0
    xense_sdk_version: str = '2.0.1'
    startup_timeout_s: float = 60.0
    init_timeout_s: float = 15.0
    sensor_ack_timeout_s: float = 2.0
    sensor_flush_timeout_s: float = 300.0
    rosbag_timeout_s: float = 15.0
    aligned_poll_interval_s: float = 0.01
    aligned_stall_timeout_s: float = 0.075
    lerobot_aligned_max_age_ms: int = 100
    fail_stop_retry_interval_s: float = 0.2
    worker_exit_timeout_s: float = 15.0
    realsense_startup_max_restarts: int = 1
    realsense_startup_stabilization_s: float = 1.5
    lerobot_conda_env: str = 'lerobot-fr3-312'
    fatal_realsense_patterns: tuple[str, ...] = (
        'Hardware Error',
        'Depth stream start failure',
    )

    def __post_init__(self) -> None:
        """Normalize paths and reject unsafe or contradictory settings."""
        if not isinstance(self.policy_path, str) or not self.policy_path.strip():
            raise ValueError('policy_path must be a non-empty string')
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError('task must be a non-empty string')
        if '\0' in self.task:
            raise ValueError('task must not contain NUL characters')
        if not self.control_socket_path:
            raise ValueError('control_socket_path must be non-empty')
        root = validate_repo_root(self.repo_root)
        inference_requirements = (
            root / 'LeRobotFR3',
            root / 'RealSense' / 'launch' / 'four_realsense_shm_runtime.launch.py',
        )
        missing = [str(path) for path in inference_requirements if not path.exists()]
        if missing:
            raise ValueError(f'inference repo requirements are missing: {missing}')
        runtime_root = root if self.runtime_root is None else Path(self.runtime_root).expanduser().resolve()
        if self.xense_sdk_version not in XENSE_SDK_CONDA_ENVS:
            raise ValueError(f'unsupported xense_sdk_version: {self.xense_sdk_version}')
        positive = {
            'ft_fps': self.ft_fps,
            'xense_fps': self.xense_fps,
            'startup_timeout_s': self.startup_timeout_s,
            'init_timeout_s': self.init_timeout_s,
            'sensor_ack_timeout_s': self.sensor_ack_timeout_s,
            'sensor_flush_timeout_s': self.sensor_flush_timeout_s,
            'rosbag_timeout_s': self.rosbag_timeout_s,
            'aligned_poll_interval_s': self.aligned_poll_interval_s,
            'aligned_stall_timeout_s': self.aligned_stall_timeout_s,
            'lerobot_aligned_max_age_ms': self.lerobot_aligned_max_age_ms,
            'fail_stop_retry_interval_s': self.fail_stop_retry_interval_s,
            'worker_exit_timeout_s': self.worker_exit_timeout_s,
            'realsense_startup_stabilization_s': self.realsense_startup_stabilization_s,
        }
        invalid = [name for name, value in positive.items() if not math.isfinite(value) or value <= 0]
        if invalid:
            raise ValueError(f'inference timeout/rate values must be positive: {invalid}')
        if (
            isinstance(self.realsense_startup_max_restarts, bool)
            or not isinstance(self.realsense_startup_max_restarts, int)
            or self.realsense_startup_max_restarts < 0
        ):
            raise ValueError('realsense_startup_max_restarts must be a non-negative integer')
        if self.aligned_stall_timeout_s * 1000 >= self.lerobot_aligned_max_age_ms:
            raise ValueError(
                'aligned_stall_timeout_s must be shorter than '
                'lerobot_aligned_max_age_ms'
            )
        if self.realsense_shm_names != EXPECTED_REALSENSE_SHM_NAMES:
            raise ValueError(
                'realsense_shm_names must match the current four-camera RealSense runtime'
            )
        object.__setattr__(self, 'repo_root', root)
        object.__setattr__(self, 'runtime_root', runtime_root)
        object.__setattr__(self, 'runtime_sessions_dir', runtime_root / 'runtime_sessions')
        object.__setattr__(self, 'runtime_frames_dir', runtime_root / 'runtime_frames')

    def lerobot_command(self) -> list[str]:
        """Build the persistent worker command without owning policy/reset state."""
        return [
            'conda', 'run', '-n', self.lerobot_conda_env,
            'lerobot-rollout',
            '--strategy.type=controlled',
            f'--strategy.control_socket_path={self.control_socket_path}',
            f'--policy.path={self.policy_path}',
            '--robot.type=fr3',
            f'--robot.command_endpoint={self.robot_command_endpoint}',
            f'--robot.telemetry_endpoint={self.robot_telemetry_endpoint}',
            f'--robot.observation_shm_name={self.observation_shm_name}',
            f'--robot.realsense_shm_names={json.dumps(self.realsense_shm_names)}',
            f'--robot.xense_shm_name={self.xense_shm_name}',
            f'--robot.ft300s_shm_name={self.ft_shm_name}',
            f'--robot.max_snapshot_age_ms={self.lerobot_aligned_max_age_ms}',
            f'--task={self.task}',
            '--duration=0',
        ]
