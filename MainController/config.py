"""Runtime configuration for the MainController package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .realsense_image_guard import ImageTopicRequirement, formal_image_requirements, select_image_requirements


REPO_ROOT = Path(__file__).resolve().parents[4]


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

    repo_root: Path = REPO_ROOT
    output_dir: Path = REPO_ROOT / 'runtime_sessions'
    zmq_connect: str = 'tcp://127.0.0.1:6000'
    ft_uds_path: str = '/tmp/ft300_sensor.sock'
    xense_uds_path: str = '/tmp/xense_sensor.sock'
    ft_shm_name: str = 'ft300_sensor_frame'
    xense_shm_name: str = 'xense_sensor_frame'
    ft_fps: float = 100.0
    xense_fps: float = 30.0
    startup_timeout_s: float = 60.0
    init_timeout_s: float = 15.0
    ack_timeout_s: float = 2.0
    progress_log_period_s: float = 5.0
    zmq_first_frame_timeout_s: float = 5.0
    rosbag_timeout_s: float = 15.0
    realsense_image_ready_timeout_s: float = 5.0
    realsense_capture_mode: str = 'formal'
    realsense_debug_image_topics: tuple[str, ...] = ()
    realsense_rosbag_count_skew_limit: int = 3
    rate: RateConfig = field(default_factory=RateConfig)
    cameras: tuple[str, ...] = ('cam1', 'cam2', 'cam3', 'cam4')
    fatal_realsense_patterns: tuple[str, ...] = (
        'Hardware Error',
        'Depth stream start failure',
    )

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
        return formal_image_requirements(self.cameras)

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
