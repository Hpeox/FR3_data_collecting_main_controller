"""RealSense image-topic readiness and rosbag metadata validation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageTopicRequirement:
    """Stable schema expected for one RealSense image topic."""

    topic: str
    message_type: str
    width: int
    height: int
    encoding: str
    step: int
    stream_role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'topic': self.topic,
            'message_type': self.message_type,
            'width': self.width,
            'height': self.height,
            'encoding': self.encoding,
            'step': self.step,
            'stream_role': self.stream_role,
        }


@dataclass(frozen=True)
class ImageTopicBaseline:
    """Observed stable schema for one image topic."""

    topic: str
    message_type: str
    width: int
    height: int
    encoding: str
    step: int
    stream_role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'topic': self.topic,
            'message_type': self.message_type,
            'width': self.width,
            'height': self.height,
            'encoding': self.encoding,
            'step': self.step,
            'stream_role': self.stream_role,
        }


@dataclass(frozen=True)
class ImageReadinessResult:
    """Result of pre-record image-topic readiness validation."""

    ok: bool
    mode: str
    required_topics: tuple[str, ...]
    baselines: tuple[ImageTopicBaseline, ...] = ()
    missing_topics: tuple[str, ...] = ()
    mismatches: tuple[dict[str, Any], ...] = ()

    def to_manifest(self) -> dict[str, Any]:
        return {
            'ok': self.ok,
            'mode': self.mode,
            'required_topics': list(self.required_topics),
            'baselines': [baseline.to_dict() for baseline in self.baselines],
            'missing_topics': list(self.missing_topics),
            'mismatches': list(self.mismatches),
        }


@dataclass(frozen=True)
class RosbagImagePostcheckResult:
    """Result of post-record rosbag image-topic validation."""

    ok: bool
    mode: str
    rosbag_uri: str
    required_topics: tuple[str, ...]
    topic_counts: dict[str, int]
    missing_topics: tuple[str, ...] = ()
    zero_count_topics: tuple[str, ...] = ()
    type_mismatches: tuple[dict[str, Any], ...] = ()
    count_skew: int | None = None
    count_skew_limit: int = 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            'ok': self.ok,
            'mode': self.mode,
            'rosbag_uri': self.rosbag_uri,
            'required_topics': list(self.required_topics),
            'topic_counts': self.topic_counts,
            'missing_topics': list(self.missing_topics),
            'zero_count_topics': list(self.zero_count_topics),
            'type_mismatches': list(self.type_mismatches),
            'count_skew': self.count_skew,
            'count_skew_limit': self.count_skew_limit,
        }


def formal_image_requirements(
    *,
    cameras: tuple[str, ...],
    image_message_type: str,
    color_width: int,
    color_height: int,
    color_encoding: str,
    color_step: int,
    depth_width: int,
    depth_height: int,
    depth_encoding: str,
    depth_step: int,
) -> tuple[ImageTopicRequirement, ...]:
    """Return the authoritative formal 4-camera / 8-image-topic requirement."""
    requirements: list[ImageTopicRequirement] = []
    for camera in cameras:
        requirements.append(
            ImageTopicRequirement(
                topic=f'/{camera}/camera/color/image_raw',
                message_type=image_message_type,
                width=color_width,
                height=color_height,
                encoding=color_encoding,
                step=color_step,
                stream_role='color',
            )
        )
        requirements.append(
            ImageTopicRequirement(
                topic=f'/{camera}/camera/aligned_depth_to_color/image_raw',
                message_type=image_message_type,
                width=depth_width,
                height=depth_height,
                encoding=depth_encoding,
                step=depth_step,
                stream_role='aligned_depth',
            )
        )
    return tuple(requirements)


def select_image_requirements(
    *,
    mode: str,
    formal_requirements: tuple[ImageTopicRequirement, ...],
    debug_topics: tuple[str, ...],
) -> tuple[ImageTopicRequirement, ...]:
    """Return required image topics for formal or explicit debug/degraded mode."""
    if mode == 'formal':
        return formal_requirements
    if mode != 'debug_degraded':
        raise ValueError(f'unsupported RealSense capture mode: {mode}')
    by_topic = {requirement.topic: requirement for requirement in formal_requirements}
    missing = [topic for topic in debug_topics if topic not in by_topic]
    if missing:
        raise ValueError(f'debug RealSense image topics are not in the formal baseline: {missing}')
    if not debug_topics:
        raise ValueError('debug_degraded RealSense capture mode requires at least one image topic')
    return tuple(by_topic[topic] for topic in debug_topics)


def evaluate_readiness(
    *,
    mode: str,
    requirements: tuple[ImageTopicRequirement, ...],
    observed: dict[str, ImageTopicBaseline],
) -> ImageReadinessResult:
    """Compare observed image baselines to required topic schema."""
    missing: list[str] = []
    mismatches: list[dict[str, Any]] = []
    baselines: list[ImageTopicBaseline] = []
    for requirement in requirements:
        baseline = observed.get(requirement.topic)
        if baseline is None:
            missing.append(requirement.topic)
            continue
        baselines.append(baseline)
        expected = requirement.to_dict()
        actual = baseline.to_dict()
        for field in ('message_type', 'width', 'height', 'encoding', 'step', 'stream_role'):
            if actual[field] != expected[field]:
                mismatches.append(
                    {
                        'topic': requirement.topic,
                        'field': field,
                        'expected': expected[field],
                        'actual': actual[field],
                    }
                )
    return ImageReadinessResult(
        ok=not missing and not mismatches,
        mode=mode,
        required_topics=tuple(requirement.topic for requirement in requirements),
        baselines=tuple(baselines),
        missing_topics=tuple(missing),
        mismatches=tuple(mismatches),
    )


def validate_rosbag_image_metadata(
    *,
    mode: str,
    rosbag_uri: Path,
    requirements: tuple[ImageTopicRequirement, ...],
    topic_metadata: dict[str, dict[str, Any]],
    count_skew_limit: int,
) -> RosbagImagePostcheckResult:
    """Validate required image topics in rosbag metadata."""
    missing: list[str] = []
    zero_count: list[str] = []
    type_mismatches: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for requirement in requirements:
        metadata = topic_metadata.get(requirement.topic)
        if metadata is None:
            missing.append(requirement.topic)
            continue
        count = int(metadata.get('count', 0))
        counts[requirement.topic] = count
        if count <= 0:
            zero_count.append(requirement.topic)
        message_type = metadata.get('message_type')
        if message_type != requirement.message_type:
            type_mismatches.append(
                {
                    'topic': requirement.topic,
                    'expected': requirement.message_type,
                    'actual': message_type,
                }
            )
    count_skew = None
    if counts:
        values = list(counts.values())
        count_skew = max(values) - min(values)
    ok = (
        not missing
        and not zero_count
        and not type_mismatches
        and count_skew is not None
        and count_skew <= count_skew_limit
    )
    return RosbagImagePostcheckResult(
        ok=ok,
        mode=mode,
        rosbag_uri=str(rosbag_uri),
        required_topics=tuple(requirement.topic for requirement in requirements),
        topic_counts=counts,
        missing_topics=tuple(missing),
        zero_count_topics=tuple(zero_count),
        type_mismatches=tuple(type_mismatches),
        count_skew=count_skew,
        count_skew_limit=count_skew_limit,
    )


def check_ros_image_topic_readiness(node, rclpy, requirements: tuple[ImageTopicRequirement, ...], timeout_s: float, mode: str) -> ImageReadinessResult:
    """Collect one Image message per required topic and compare stable schema."""
    try:
        from sensor_msgs.msg import Image
    except Exception as exc:  # pragma: no cover - depends on ROS environment.
        raise RuntimeError(f'ROS Image message dependency is unavailable: {exc}') from exc

    observed: dict[str, ImageTopicBaseline] = {}
    subscriptions = []

    def make_callback(requirement: ImageTopicRequirement):
        def callback(message) -> None:
            observed[requirement.topic] = ImageTopicBaseline(
                topic=requirement.topic,
                message_type=requirement.message_type,
                width=int(message.width),
                height=int(message.height),
                encoding=str(message.encoding),
                step=int(message.step),
                stream_role=requirement.stream_role,
            )

        return callback

    for requirement in requirements:
        subscriptions.append(node.create_subscription(Image, requirement.topic, make_callback(requirement), 10))

    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and len(observed) < len(requirements):
            rclpy.spin_once(node, timeout_sec=0.05)
        return evaluate_readiness(mode=mode, requirements=requirements, observed=observed)
    finally:
        for subscription in subscriptions:
            try:
                node.destroy_subscription(subscription)
            except Exception:
                pass


def read_rosbag_topic_metadata(rosbag_uri: Path) -> dict[str, dict[str, Any]]:
    """Read topic counts and types from rosbag2 metadata."""
    try:
        import rosbag2_py
    except Exception as exc:  # pragma: no cover - depends on ROS environment.
        raise RuntimeError(f'rosbag2_py dependency is unavailable: {exc}') from exc

    metadata = rosbag2_py.Info().read_metadata(str(rosbag_uri))
    result: dict[str, dict[str, Any]] = {}
    for item in metadata.topics_with_message_count:
        topic_metadata = item.topic_metadata
        result[topic_metadata.name] = {
            'message_type': topic_metadata.type,
            'count': int(item.message_count),
        }
    return result
