"""Runtime configuration for the MainController package."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .realsense_image_guard import (
    ImageTopicRequirement,
    formal_image_requirements,
    select_image_requirements,
)


REQUIRED_REPO_DIRS = (
    Path('FT300S'),
    Path('XenseTacSensor'),
    Path('RealSense') / 'launch',
)
REQUIRED_REALSENSE_LAUNCHES = (
    Path('RealSense') / 'launch' / 'four_realsense_640x480_30.launch.py',
    Path('RealSense') / 'launch' / 'rosbag2_recorder.launch.py',
)
XENSE_SDK_CONDA_ENVS = {
    '1.x': 'Xense310',
    '2.0': 'xense2_bak',
    '2.0.1': 'xense2',
}
TASK_NAME_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
WEIGHT_SUM_ABS_TOL = 1e-9


def validate_task_name(task_name: str) -> str:
    """Validate and return a task name safe for use as one filename stem."""
    if not isinstance(task_name, str):
        raise TypeError('task_name must be a string')
    if not TASK_NAME_PATTERN.fullmatch(task_name) or '..' in task_name:
        raise ValueError(
            'task_name must start with an ASCII letter or digit, contain only '
            'ASCII letters, digits, ".", "_", or "-", and must not contain ".."'
        )
    return task_name


def _instruction_text(value: Any, index: int) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f'instructions[{index}].text must be a string')
    text = value.strip()
    if not text:
        raise RuntimeError(f'instructions[{index}].text must not be empty')
    if '\x00' in text:
        raise RuntimeError(
            f'instructions[{index}].text must not contain NUL characters'
        )
    return text


def _instruction_weight(value: Any, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f'instructions[{index}].weight must be a number')
    weight = float(value)
    if not math.isfinite(weight):
        raise RuntimeError(f'instructions[{index}].weight must be finite')
    if weight != -1.0 and not 0.0 < weight < 1.0:
        raise RuntimeError(
            f'instructions[{index}].weight must be -1 or satisfy 0 < weight < 1'
        )
    return weight


def parse_task_instruction_payload(
    payload: Any,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Validate a task instruction JSON payload and resolve automatic weights."""
    if not isinstance(payload, dict) or set(payload) != {'instructions'}:
        raise RuntimeError(
            'task instruction JSON must be an object containing only "instructions"'
        )
    items = payload['instructions']
    if not isinstance(items, list) or not items:
        raise RuntimeError('task instruction "instructions" must be a non-empty array')

    texts: list[str] = []
    raw_weights: list[float] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != {'text', 'weight'}:
            raise RuntimeError(
                f'instructions[{index}] must be an object containing only '
                '"text" and "weight"'
            )
        texts.append(_instruction_text(item['text'], index))
        raw_weights.append(_instruction_weight(item['weight'], index))

    explicit_sum = sum(weight for weight in raw_weights if weight != -1.0)
    automatic_count = sum(weight == -1.0 for weight in raw_weights)
    if automatic_count:
        if explicit_sum >= 1.0:
            raise RuntimeError(
                'explicit instruction weights must sum to less than 1 when '
                'automatic weight entries are present'
            )
        automatic_weight = (1.0 - explicit_sum) / automatic_count
        weights = tuple(
            automatic_weight if weight == -1.0 else weight
            for weight in raw_weights
        )
    else:
        if not math.isclose(
            explicit_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=WEIGHT_SUM_ABS_TOL,
        ):
            raise RuntimeError(
                'explicit instruction weights must sum to 1 when no automatic '
                'weight entries are present'
            )
        weights = tuple(raw_weights)
    return tuple(texts), weights


def load_task_instruction_config(
    repo_root: Path,
    task_name: str,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Load one task instruction file anchored at the integrated repo root."""
    task_name = validate_task_name(task_name)
    path = repo_root / 'TaskInstruction' / f'{task_name}.json'
    if not path.exists():
        raise RuntimeError(f'task instruction file does not exist: {path}')
    if not path.is_file():
        raise RuntimeError(f'task instruction path is not a file: {path}')
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f'task instruction file is not valid UTF-8: {path}'
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f'task instruction file is not valid JSON: {path}: {exc}'
        ) from exc
    except OSError as exc:
        raise RuntimeError(f'cannot read task instruction file {path}: {exc}') from exc
    return parse_task_instruction_payload(payload)


def build_time_repo_root_hint() -> Path | None:
    """Return the repo root recorded during package build, if present."""
    try:
        from ._repo_root_hint import REPO_ROOT_HINT
    except Exception:
        return None
    return Path(REPO_ROOT_HINT)


def validate_repo_root(repo_root: Path) -> Path:
    """Resolve and validate the integrated repository root."""
    root = repo_root.expanduser().resolve()
    required = (*REQUIRED_REPO_DIRS, *REQUIRED_REALSENSE_LAUNCHES)
    missing = [str(path) for path in required if not (root / path).exists()]
    if missing:
        joined = ', '.join(missing)
        raise RuntimeError(f'invalid repo root {root}: missing {joined}')
    return root


def default_repo_root() -> Path:
    """Return the build-time repo root hint, or raise with an actionable message."""
    hint = build_time_repo_root_hint()
    if hint is None:
        raise RuntimeError(
            'repo root is not configured; pass --repo-root PATH or rebuild '
            'MainController with colcon'
        )
    return validate_repo_root(hint)


@dataclass(frozen=True)
class RateConfig:
    """Target stream rates and drop-warning intervals."""

    ft300_hz: float = 100.0
    xense_hz: float = 30.0
    zmq_hz: float = 50.0
    realsense_hz: float = 30.0
    warning_factor: float = 2.0


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration values shared by controller components."""

    task_name: str
    task_instructions: tuple[str, ...]
    task_instruction_weights: tuple[float, ...]
    repo_root: Path = field(default_factory=default_repo_root)
    runtime_root: Path | None = None
    runtime_sessions_dir: Path = field(init=False)
    runtime_frames_dir: Path = field(init=False)
    zmq_connect: str = 'tcp://127.0.0.1:6000'
    ft_uds_path: str = '/tmp/ft300_sensor.sock'
    xense_uds_path: str = '/tmp/xense_sensor.sock'
    ft_shm_name: str = 'ft300_sensor_frame'
    xense_shm_name: str = 'xense_sensor_frame'
    ft_fps: float = 100.0
    xense_fps: float = 30.0
    xense_sdk_version: str = '2.0.1'
    startup_timeout_s: float = 60.0
    init_timeout_s: float = 15.0
    ack_timeout_s: float = 2.0
    sensor_flush_timeout_s: float | None = 300.0
    progress_log_period_s: float = 5.0
    alignment_base: str = 'realsense:bundle'
    alignment_mode: str = 'causal'
    alignment_hz: float = 30.0
    alignment_start_trim_s: float = 2.0
    alignment_end_trim_s: float = 0.0
    gripper_plot_timeout_s: float = 30.0
    xense_tactile_zero_force_mean_tolerance: float = 0.1
    xense_tactile_edge_warning_threshold: float = 0.5
    xense_tactile_edge_window_samples: int = 15
    zmq_first_frame_timeout_s: float = 5.0
    rosbag_timeout_s: float = 15.0
    realsense_image_ready_timeout_s: float = 30.0
    realsense_capture_mode: str = 'formal'
    realsense_debug_image_topics: tuple[str, ...] = ()
    realsense_rosbag_count_skew_limit_percent: float = 0.5
    rate: RateConfig = field(default_factory=RateConfig)

    cameras: tuple[str, ...] = ('cam1', 'cam2', 'cam3', 'cam4')
    # Current-site RealSense image baseline from the checked-in launch profile.
    # If RealSense launch parameters change, update these values in the same
    # change so readiness and rosbag post-checks stay aligned with recording.
    realsense_image_message_type: str = 'sensor_msgs/msg/Image'
    realsense_color_width: int = 640
    realsense_color_height: int = 480
    realsense_color_encoding: str = 'rgb8'
    realsense_color_step: int = 1920
    realsense_depth_width: int = 640
    realsense_depth_height: int = 480
    realsense_depth_encoding: str = '16UC1'
    realsense_depth_step: int = 1280

    fatal_realsense_patterns: tuple[str, ...] = (
        'Hardware Error',
        'Depth stream start failure',
    )

    def __post_init__(self) -> None:
        """Normalize path settings after dataclass initialization."""
        task_name = validate_task_name(self.task_name)
        if not self.task_instructions:
            raise ValueError('task_instructions must not be empty')
        if len(self.task_instructions) != len(self.task_instruction_weights):
            raise ValueError(
                'task_instructions and task_instruction_weights must have equal length'
            )
        task_instructions = tuple(
            _instruction_text(text, index)
            for index, text in enumerate(self.task_instructions)
        )
        for index, weight in enumerate(self.task_instruction_weights):
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError(
                    f'task_instruction_weights[{index}] must be a number'
                )
            if not math.isfinite(float(weight)) or float(weight) <= 0.0:
                raise ValueError(
                    f'task_instruction_weights[{index}] must be finite and positive'
                )
        if not math.isclose(
            sum(float(weight) for weight in self.task_instruction_weights),
            1.0,
            rel_tol=0.0,
            abs_tol=WEIGHT_SUM_ABS_TOL,
        ):
            raise ValueError('task_instruction_weights must sum to 1')
        repo_root = validate_repo_root(self.repo_root)
        if self.xense_sdk_version not in XENSE_SDK_CONDA_ENVS:
            allowed = ', '.join(sorted(XENSE_SDK_CONDA_ENVS))
            raise ValueError(f'unsupported xense_sdk_version {self.xense_sdk_version!r}; expected one of: {allowed}')
        runtime_root = (
            repo_root
            if self.runtime_root is None
            else Path(self.runtime_root).expanduser().resolve()
        )
        object.__setattr__(self, 'task_name', task_name)
        object.__setattr__(self, 'task_instructions', task_instructions)
        object.__setattr__(
            self,
            'task_instruction_weights',
            tuple(float(weight) for weight in self.task_instruction_weights),
        )
        object.__setattr__(self, 'repo_root', repo_root)
        object.__setattr__(self, 'runtime_root', runtime_root)
        object.__setattr__(self, 'runtime_sessions_dir', runtime_root / 'runtime_sessions')
        object.__setattr__(self, 'runtime_frames_dir', runtime_root / 'runtime_frames')

    @property
    def realsense_metadata_topics(self) -> tuple[str, ...]:
        """Return metadata topics used for RealSense timing checks."""
        topics: list[str] = []
        for camera in self.cameras:
            topics.append(f'/{camera}/camera/color/metadata')
            topics.append(f'/{camera}/camera/depth/metadata')
        return tuple(topics)

    @property
    def formal_realsense_image_requirements(self) -> tuple[ImageTopicRequirement, ...]:
        """Return the formal RealSense image recording requirements."""
        return formal_image_requirements(
            cameras=self.cameras,
            image_message_type=self.realsense_image_message_type,
            color_width=self.realsense_color_width,
            color_height=self.realsense_color_height,
            color_encoding=self.realsense_color_encoding,
            color_step=self.realsense_color_step,
            depth_width=self.realsense_depth_width,
            depth_height=self.realsense_depth_height,
            depth_encoding=self.realsense_depth_encoding,
            depth_step=self.realsense_depth_step,
        )

    @property
    def realsense_image_requirements(self) -> tuple[ImageTopicRequirement, ...]:
        """Return image topics required for this run's capture mode."""
        return select_image_requirements(
            mode=self.realsense_capture_mode,
            formal_requirements=self.formal_realsense_image_requirements,
            debug_topics=self.realsense_debug_image_topics,
        )


def ns_from_hz(rate_hz: float, factor: float = 1.0) -> int:
    """Convert a frequency to an integer nanosecond interval."""
    return int(round((1.0 / rate_hz) * factor * 1_000_000_000))
