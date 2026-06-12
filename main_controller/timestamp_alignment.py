"""Offline timestamp alignment for one completed MainController demo."""

from __future__ import annotations

from itertools import product
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


NSEC_PER_SEC = 1_000_000_000
ZMQ_CLOCK_OFFSET_WARN_MS = 100.0
ZMQ_CLOCK_OFFSET_CHECK_HINT = 'check chrony/NTP sync: run `chronyc sources -v`; expected first line: ^*192.168.10.1'
REALSENSE_BUNDLE_INITIAL_SEARCH_RADIUS = 2
REALSENSE_BUNDLE_FALLBACK_SEARCH_RADIUS = 1
REALSENSE_BUNDLE_SPAN_WARN_NS = 20_000_000


@dataclass(frozen=True)
class AlignmentOptions:
    """Configuration for timestamp-index generation."""

    repo_root: Path | None = None
    output_dir: Path | None = None
    base: str = 'realsense:bundle'
    mode: str = 'causal'
    hz: float = 30.0
    start_trim_s: float = 0.0
    end_trim_s: float = 0.0
    allow_degraded: bool = False


@dataclass(frozen=True)
class AlignmentResult:
    """Paths and summary produced by one alignment run."""

    demo_dir: Path
    status: str
    config_path: Path
    index_path: Path
    manifest_path: Path
    report_path: Path
    sample_count: int
    valid_count: int
    base: str
    base_kind: str
    zmq_clock_offsets: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_manifest_entry(self, started_ns: int, finished_ns: int) -> dict[str, Any]:
        """Return the manifest.alignment entry for this result."""
        return {
            'status': self.status,
            'config_path': relative_to(self.config_path, self.demo_dir),
            'index_path': relative_to(self.index_path, self.demo_dir),
            'manifest_path': relative_to(self.manifest_path, self.demo_dir),
            'report_path': relative_to(self.report_path, self.demo_dir),
            'started_ns': started_ns,
            'finished_ns': finished_ns,
            'sample_count': self.sample_count,
            'valid_count': self.valid_count,
            'base': self.base,
            'base_kind': self.base_kind,
            'zmq_clock_offsets': self.zmq_clock_offsets,
            'warnings': list(self.warnings),
        }


@dataclass
class ChildSource:
    name: str
    display_name: str
    time_ns: np.ndarray
    source_index: np.ndarray
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    row_valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.row_valid is None:
            self.row_valid = np.ones(len(self.time_ns), dtype=bool)
        else:
            self.row_valid = np.asarray(self.row_valid, dtype=bool)
        if len(self.row_valid) != len(self.time_ns):
            raise ValueError(f'{self.name} row_valid length does not match time_ns length')

    def subset(self, indices: np.ndarray) -> 'ChildSource':
        return ChildSource(
            self.name,
            self.display_name,
            self.time_ns[indices].astype(np.int64, copy=False),
            self.source_index[indices].astype(np.int64, copy=False),
            {key: value[indices] for key, value in self.columns.items()},
            dict(self.details),
            self.row_valid[indices],
        )


@dataclass
class Source:
    name: str
    display_name: str
    time_ns: np.ndarray
    source_index: np.ndarray
    tolerance_causal_ns: int
    tolerance_nearest_ns: int
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    children: dict[str, ChildSource] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    row_valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.row_valid is None:
            self.row_valid = np.ones(len(self.time_ns), dtype=bool)
        else:
            self.row_valid = np.asarray(self.row_valid, dtype=bool)
        if len(self.row_valid) != len(self.time_ns):
            raise ValueError(f'{self.name} row_valid length does not match time_ns length')

    def sorted_valid(self) -> 'Source':
        valid = self.time_ns > 0
        order = np.argsort(self.time_ns[valid], kind='stable')
        indices = np.nonzero(valid)[0][order]
        return Source(
            self.name,
            self.display_name,
            self.time_ns[indices].astype(np.int64, copy=False),
            self.source_index[indices].astype(np.int64, copy=False),
            self.tolerance_causal_ns,
            self.tolerance_nearest_ns,
            {key: value[indices] for key, value in self.columns.items()},
            {key: child.subset(indices) for key, child in self.children.items()},
            dict(self.details),
            self.row_valid[indices],
        )

    def trim(self, start_ns: int, end_ns: int) -> 'Source':
        keep = (self.time_ns >= start_ns) & (self.time_ns <= end_ns)
        return Source(
            self.name,
            self.display_name,
            self.time_ns[keep],
            self.source_index[keep],
            self.tolerance_causal_ns,
            self.tolerance_nearest_ns,
            {key: value[keep] for key, value in self.columns.items()},
            {key: child.subset(keep) for key, child in self.children.items()},
            trim_details(self.details, self.source_index[keep]),
            self.row_valid[keep],
        )


@dataclass
class RosbagImageStream:
    header_time_ns: np.ndarray
    recorded_time_ns: np.ndarray


@dataclass
class Match:
    index: np.ndarray
    position: np.ndarray
    time_ns: np.ndarray
    delta_ns: np.ndarray
    valid: np.ndarray


@dataclass
class AlignmentContext:
    sources: dict[str, Source]
    xense_pair: Source | None


def align_demo_timestamps(demo_dir: Path, options: AlignmentOptions | None = None) -> AlignmentResult:
    """Generate alignment config, index, manifest, and report for one demo."""
    options = options or AlignmentOptions()
    demo_dir = demo_dir.resolve()
    output_dir = (options.output_dir or demo_dir / 'aligned').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(demo_dir / 'manifest.json')
    if manifest.get('status') != 'done' and not options.allow_degraded:
        raise RuntimeError(f"alignment requires manifest.status == 'done', got {manifest.get('status')!r}")

    warnings: list[str] = []
    npz_paths = resolve_npz_paths(demo_dir, manifest)
    sources, zmq_clock_offsets = load_sources(demo_dir, manifest, npz_paths, warnings)
    sources = {name: source for name, source in sources.items() if len(source.time_ns) > 0}
    if not sources:
        raise RuntimeError('no timestamp streams found')

    xense_pair = build_xense_pair_source(npz_paths) if 'xense' in npz_paths else None
    context = AlignmentContext(sources, xense_pair)
    base = validate_base(options.base, sources)
    start_ns, end_ns = alignment_window(context, options)
    target = build_target_source(base, context, manifest, options, warnings, start_ns, end_ns)
    t_ns = target.time_ns
    if len(t_ns) == 0:
        raise RuntimeError('target timeline is empty after trims')

    arrays: dict[str, np.ndarray] = {
        't_ns': t_ns,
        'segment_id': np.zeros(len(t_ns), dtype=np.int64),
    }
    arrays.update(target.columns)
    stats: dict[str, dict[str, Any]] = {}
    valid_masks: list[np.ndarray] = []
    realsense_alignment_kind, realsense_details = align_realsense_group(
        base,
        sources,
        manifest,
        target,
        t_ns,
        options,
        warnings,
        start_ns,
        end_ns,
        arrays,
        stats,
        valid_masks,
    )
    xense_alignment_kind = align_xense_pair_group(xense_pair, t_ns, options, arrays, stats, valid_masks)
    scalar_alignment_kind = align_scalar_sources(sources, t_ns, options, arrays, stats, valid_masks)

    sample_valid = np.logical_and.reduce(valid_masks) if valid_masks else np.ones(len(t_ns), dtype=bool)
    arrays['sample_valid'] = sample_valid
    index_path = output_dir / 'aligned_index.npz'
    np.savez_compressed(index_path, **arrays)

    source_path_payload = source_paths(manifest)
    config = {
        'schema_version': 3,
        'demo_dir': '.',
        'output_dir': relative_to(output_dir, demo_dir),
        'base': base,
        'base_kind': target.details.get('kind', 'source'),
        'realsense_alignment_kind': realsense_alignment_kind,
        'xense_alignment_kind': xense_alignment_kind,
        'scalar_alignment_kind': scalar_alignment_kind,
        'mode': options.mode,
        'hz': options.hz,
        'start_trim_s': options.start_trim_s,
        'end_trim_s': options.end_trim_s,
        'trim_window_ns': {'start': int(start_ns), 'end': int(end_ns)},
        'sources': source_path_payload,
        'streams': {name: {'display_name': source.display_name, 'topic': source.details.get('topic')} for name, source in sources.items()},
        'base_details': target.details,
        'realsense_details': realsense_details,
    }
    config_path = output_dir / 'alignment_config.json'
    write_json(config_path, config)

    aligned_manifest = {
        'schema_version': 3,
        'status': 'done',
        'demo_dir': '.',
        'sample_count': int(len(t_ns)),
        'valid_count': int(sample_valid.sum()),
        'base': base,
        'base_kind': target.details.get('kind', 'source'),
        'realsense_alignment_kind': realsense_alignment_kind,
        'xense_alignment_kind': xense_alignment_kind,
        'scalar_alignment_kind': scalar_alignment_kind,
        'mode': options.mode,
        'hz': options.hz,
        'start_trim_s': options.start_trim_s,
        'end_trim_s': options.end_trim_s,
        'trim_window_ns': {'start': int(start_ns), 'end': int(end_ns)},
        'sources': source_path_payload,
        'streams': stats,
        'base_details': target.details,
        'realsense_details': realsense_details,
        'zmq_clock_offsets': zmq_clock_offsets,
        'clock_domain': clock_domain_summary(npz_paths.get('realsense'), warnings),
        'drop_monitors': manifest.get('drop_monitors', {}),
        'warnings': warnings,
    }
    aligned_manifest_path = output_dir / 'aligned_manifest.json'
    write_json(aligned_manifest_path, aligned_manifest)
    report_path = output_dir / 'alignment_report.md'
    report_path.write_text(render_report(aligned_manifest), encoding='utf-8')
    return AlignmentResult(
        demo_dir=demo_dir,
        status='done',
        config_path=config_path,
        index_path=index_path,
        manifest_path=aligned_manifest_path,
        report_path=report_path,
        sample_count=int(len(t_ns)),
        valid_count=int(sample_valid.sum()),
        base=base,
        base_kind=target.details.get('kind', 'source'),
        zmq_clock_offsets=zmq_clock_offsets,
        warnings=tuple(warnings),
    )


def update_manifest_alignment(manifest_path: Path, entry: dict[str, Any]) -> None:
    """Update one demo manifest with an independent alignment status entry."""
    manifest = read_json(manifest_path)
    manifest['alignment'] = entry
    write_json(manifest_path, manifest)


def failure_manifest_entry(started_ns: int, error: Exception) -> dict[str, Any]:
    """Return a manifest.alignment failure entry."""
    return {
        'status': 'failed',
        'started_ns': started_ns,
        'finished_ns': time.time_ns(),
        'error': str(error),
    }


def load_sources(
    demo_dir: Path,
    manifest: dict[str, Any],
    npz_paths: dict[str, Path],
    warnings: list[str],
) -> tuple[dict[str, Source], dict[str, dict[str, Any]]]:
    sources: dict[str, Source] = {}
    zmq_clock_offsets: dict[str, dict[str, Any]] = {}
    if 'ft300' in npz_paths:
        data = np.load(npz_paths['ft300'], allow_pickle=True)
        sources['ft300s'] = Source('ft300s', 'FT300S', int_array(data['timestamp_ns']), np.arange(len(data['timestamp_ns']), dtype=np.int64), 20_000_000, 10_000_000).sorted_valid()
    if 'xense' in npz_paths:
        data = np.load(npz_paths['xense'], allow_pickle=True)
        sources['xense_0'] = Source('xense_0', 'Xense sensor 0', int_array(data['timestamp_ns_0']), np.arange(len(data['timestamp_ns_0']), dtype=np.int64), 66_700_000, 33_400_000).sorted_valid()
        sources['xense_1'] = Source('xense_1', 'Xense sensor 1', int_array(data['timestamp_ns_1']), np.arange(len(data['timestamp_ns_1']), dtype=np.int64), 66_700_000, 33_400_000).sorted_valid()
    if 'zmq' in npz_paths:
        data = np.load(npz_paths['zmq'], allow_pickle=True)
        zmq_source_ids = int_array(data['source'])
        raw_stamp_ns = np.asarray([int(round(float(value) * NSEC_PER_SEC)) for value in data['stamp_s']], dtype=np.int64)
        recv_time_ns = int_array(data['recv_time_ns'])
        for source_id in sorted(set(int(value) for value in zmq_source_ids if value > 0)):
            mask = zmq_source_ids == source_id
            offset_ns = int(np.median(recv_time_ns[mask] - raw_stamp_ns[mask]))
            stream_name = f'zmq_source_{source_id}'
            offset_ms = float(offset_ns / 1_000_000.0)
            zmq_clock_offsets[stream_name] = {
                'source': source_id,
                'offset_ms': offset_ms,
                'offset_ns': offset_ns,
                'frame_count': int(mask.sum()),
            }
            if abs(offset_ms) > ZMQ_CLOCK_OFFSET_WARN_MS:
                warnings.append(zmq_clock_offset_warning(source_id, offset_ms))
            sources[stream_name] = Source(
                stream_name,
                f'ZMQ source {source_id}',
                raw_stamp_ns[mask] + offset_ns,
                np.nonzero(mask)[0].astype(np.int64),
                40_000_000,
                20_000_000,
            ).sorted_valid()
    sources.update(realsense_sources(demo_dir, manifest, npz_paths.get('realsense'), warnings))
    return sources, zmq_clock_offsets


def zmq_clock_offset_warning(source: int, offset_ms: float) -> str:
    return (
        f'ZMQ source {source} clock offset {offset_ms:.3f} ms exceeds '
        f'{ZMQ_CLOCK_OFFSET_WARN_MS:.3f} ms; {ZMQ_CLOCK_OFFSET_CHECK_HINT}'
    )


def realsense_sources(demo_dir: Path, manifest: dict[str, Any], npz_path: Path | None, warnings: list[str]) -> dict[str, Source]:
    if npz_path is None:
        return {}
    required_topics = required_image_topics(manifest)
    rosbag = read_rosbag_image_streams(resolve_rosbag_uri(demo_dir, manifest), required_topics, warnings)
    missing_required = [
        topic
        for topic in required_topics
        if topic not in rosbag or len(rosbag[topic].header_time_ns) == 0
    ]
    if missing_required:
        raise RuntimeError(f'RealSense required image topics are unreadable from rosbag image messages: {missing_required}')
    sources: dict[str, Source] = {}
    for image_topic in required_topics:
        name = realsense_stream_name(image_topic)
        image_stream = rosbag[image_topic]
        columns: dict[str, np.ndarray] = {
            f'{name}_recorded_time_ns': image_stream.recorded_time_ns,
        }
        sources[name] = Source(
            name,
            f'RealSense {image_topic}',
            image_stream.header_time_ns,
            np.arange(len(image_stream.header_time_ns), dtype=np.int64),
            66_700_000,
            33_400_000,
            columns,
            {},
            {'topic': image_topic, 'timestamp_source': 'rosbag_image'},
        ).sorted_valid()
    return sources


def read_rosbag_image_streams(rosbag_uri: Path | None, topics: list[str], warnings: list[str]) -> dict[str, RosbagImageStream]:
    if rosbag_uri is None or not rosbag_uri.exists() or not topics:
        return {}
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except Exception as exc:
        warnings.append(f'rosbag image header read skipped: {exc}')
        return {}
    try:
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(rosbag_uri), storage_id=detect_storage_id(rosbag_uri)), rosbag2_py.ConverterOptions('', ''))
        topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
        selected = [topic for topic in topics if topic in topic_types]
        reader.set_filter(rosbag2_py.StorageFilter(topics=selected))
        classes = {topic: get_message(topic_types[topic]) for topic in selected}
        header_times: dict[str, list[int]] = {topic: [] for topic in selected}
        recorded_times: dict[str, list[int]] = {topic: [] for topic in selected}
        while reader.has_next():
            topic, serialized, _recorded = reader.read_next()
            header_times[topic].append(stamp_to_ns(deserialize_message(serialized, classes[topic]).header.stamp))
            recorded_times[topic].append(int(_recorded))
        return {
            topic: RosbagImageStream(
                np.asarray(header_times[topic], dtype=np.int64),
                np.asarray(recorded_times[topic], dtype=np.int64),
            )
            for topic in selected
        }
    except Exception as exc:
        warnings.append(f'rosbag image header read failed: {exc}')
        return {}


def build_target_source(
    base: str,
    context: AlignmentContext,
    manifest: dict[str, Any],
    options: AlignmentOptions,
    warnings: list[str],
    start_ns: int,
    end_ns: int,
) -> Source:
    if base == 'xense:pair':
        if context.xense_pair is None:
            raise RuntimeError('xense:pair base requires xense timestamp npz')
        return context.xense_pair.trim(start_ns, end_ns)
    if base == 'realsense:bundle':
        return build_realsense_bundle_source(context.sources, manifest, warnings, start_ns, end_ns)
    if base == 'grid':
        step_ns = int(round(NSEC_PER_SEC / options.hz))
        time_ns = np.arange(start_ns, end_ns + 1, step_ns, dtype=np.int64)
        return synthetic_source(base, base, time_ns, {'kind': 'grid', 'hz': options.hz})
    if base == 'robot':
        source = context.sources.get('zmq_source_2')
        if source is None:
            raise RuntimeError('robot base requires zmq_source_2')
        return clone_target_source('robot', source, start_ns, end_ns, {'kind': 'robot'})
    if is_realsense_single_topic_base(base):
        topic = base.split(':', 1)[1]
        source = context.sources.get(realsense_stream_name(topic))
        if source is None:
            raise RuntimeError(f'base stream has no data: {base}')
        return clone_target_source(base, source, start_ns, end_ns, {'kind': 'realsense_single', 'topic': topic})
    raise RuntimeError(
        f'unsupported --base {base!r}; choose realsense:bundle, realsense:<topic>, xense:pair, grid, or robot'
    )


def clone_target_source(
    name: str,
    source: Source,
    start_ns: int,
    end_ns: int,
    details: dict[str, Any],
) -> Source:
    start_ns = max(int(source.time_ns[0]), start_ns)
    end_ns = min(int(source.time_ns[-1]), end_ns)
    if end_ns < start_ns:
        time_ns = np.asarray([], dtype=np.int64)
    else:
        time_ns = source.time_ns[(source.time_ns >= start_ns) & (source.time_ns <= end_ns)]
    return Source(
        name,
        name,
        time_ns,
        np.arange(len(time_ns), dtype=np.int64),
        source.tolerance_causal_ns,
        source.tolerance_nearest_ns,
        details=details,
    )


def synthetic_source(name: str, display_name: str, time_ns: np.ndarray, details: dict[str, Any]) -> Source:
    return Source(
        name,
        display_name,
        time_ns,
        np.arange(len(time_ns), dtype=np.int64),
        66_700_000,
        33_400_000,
        details=details,
    )


def build_xense_pair_source(npz_paths: dict[str, Path]) -> Source:
    if 'xense' not in npz_paths:
        raise RuntimeError('xense:pair base requires xense timestamp npz')
    data = np.load(npz_paths['xense'], allow_pickle=True)
    ts0 = int_array(data['timestamp_ns_0'])
    ts1 = int_array(data['timestamp_ns_1'])
    if len(ts0) != len(ts1):
        raise RuntimeError(f'xense:pair requires equal timestamp array lengths, got {len(ts0)} and {len(ts1)}')
    valid_pair = (ts0 > 0) & (ts1 > 0)
    if not bool(valid_pair.all()):
        raise RuntimeError(f'xense:pair requires both sensor timestamps on every row; invalid rows: {int((~valid_pair).sum())}')
    source_index = np.arange(len(ts0), dtype=np.int64)
    pair_time = np.maximum(ts0, ts1).astype(np.int64, copy=False)
    order = np.argsort(pair_time, kind='stable')
    pair_time = pair_time[order]
    source_index = source_index[order]
    ts0 = ts0[order]
    ts1 = ts1[order]
    columns = {
        'xense_pair_time_ns': pair_time,
        'xense_pair_source_index': source_index,
    }
    children = {
        'xense_0': ChildSource('xense_0', 'Xense sensor 0', ts0, source_index),
        'xense_1': ChildSource('xense_1', 'Xense sensor 1', ts1, source_index),
    }
    details = {
        'kind': 'xense_pair',
        'pair_count': int(len(pair_time)),
        'source_index_start': None if len(source_index) == 0 else int(source_index[0]),
        'source_index_end': None if len(source_index) == 0 else int(source_index[-1]),
    }
    return Source(
        'xense_pair',
        'Xense same-row pair',
        pair_time,
        source_index,
        66_700_000,
        33_400_000,
        columns,
        children,
        details,
    )


def trim_details(details: dict[str, Any], source_index: np.ndarray) -> dict[str, Any]:
    result = dict(details)
    if result.get('kind') == 'xense_pair':
        result['pair_count'] = int(len(source_index))
        result['source_index_start'] = None if len(source_index) == 0 else int(source_index[0])
        result['source_index_end'] = None if len(source_index) == 0 else int(source_index[-1])
    if result.get('kind') == 'realsense_bundle':
        result['bundle_count'] = int(len(source_index))
    return result


def empty_source(name: str, display_name: str, kind: str) -> Source:
    return Source(
        name,
        display_name,
        np.asarray([], dtype=np.int64),
        np.asarray([], dtype=np.int64),
        66_700_000,
        33_400_000,
        {},
        {},
        {'kind': kind},
    )


def alignment_window(context: AlignmentContext, options: AlignmentOptions) -> tuple[int, int]:
    bounds: list[tuple[int, int]] = []
    if context.xense_pair is not None and len(context.xense_pair.time_ns) > 0:
        bounds.append((int(context.xense_pair.time_ns[0]), int(context.xense_pair.time_ns[-1])))
    for name, source in context.sources.items():
        if name in {'xense_0', 'xense_1'}:
            continue
        bounds.append((int(source.time_ns[0]), int(source.time_ns[-1])))
    if not bounds:
        raise RuntimeError('no timestamp groups found')
    start_ns = max(start for start, _ in bounds) + int(round(options.start_trim_s * NSEC_PER_SEC))
    end_ns = min(end for _, end in bounds) - int(round(options.end_trim_s * NSEC_PER_SEC))
    if end_ns < start_ns:
        raise RuntimeError('target timeline is empty after global trims')
    return int(start_ns), int(end_ns)


def validate_base(base: str, sources: dict[str, Source]) -> str:
    if base == 'auto':
        raise RuntimeError(
            '--base auto is no longer supported; choose an explicit base: '
            'realsense:bundle, realsense:<topic>, xense:pair, grid, or robot'
        )
    if base in {'xense:0', 'xense:1'}:
        raise RuntimeError(f'--base {base} is no longer supported; use --base xense:pair')
    if base in {'grid', 'robot', 'xense:pair', 'realsense:bundle'}:
        return base
    if base.startswith('realsense:') and base.split(':', 1)[1]:
        return base
    raise RuntimeError(
        f'unsupported --base {base!r}; choose realsense:bundle, realsense:<topic>, xense:pair, grid, or robot'
    )


def align_realsense_group(
    base: str,
    sources: dict[str, Source],
    manifest: dict[str, Any],
    target: Source,
    t_ns: np.ndarray,
    options: AlignmentOptions,
    warnings: list[str],
    start_ns: int,
    end_ns: int,
    arrays: dict[str, np.ndarray],
    stats: dict[str, dict[str, Any]],
    valid_masks: list[np.ndarray],
) -> tuple[str, dict[str, Any]]:
    realsense_sources_for_group = {
        name: source for name, source in sources.items() if name.startswith('realsense_')
    }
    if not realsense_sources_for_group:
        return 'absent', {}
    if is_realsense_single_topic_base(base):
        for source in realsense_sources_for_group.values():
            emit_source(source, match_source(t_ns, source, options.mode), t_ns, arrays, stats, valid_masks)
        return 'single_topic', {
            'kind': 'single_topic',
            'topic': base.split(':', 1)[1],
        }

    if base == 'realsense:bundle':
        bundle_source = target
    else:
        bundle_source = build_realsense_bundle_source(sources, manifest, warnings, start_ns, end_ns)
    if len(bundle_source.time_ns) == 0:
        raise RuntimeError('RealSense bundle source timeline is empty')
    bundle_match = match_source(t_ns, bundle_source, options.mode)
    emit_source(
        bundle_source,
        bundle_match,
        t_ns,
        arrays,
        stats,
        valid_masks,
        valid_output_name='realsense_bundle_valid',
        project_invalid_rows=True,
    )
    details = dict(bundle_source.details)
    details['kind'] = 'bundle'
    details['source_kind'] = bundle_source.details.get('kind', 'realsense_bundle')
    details['projection'] = {
        'matched_count': int(bundle_match.valid.sum()),
        'invalid_count': int(len(bundle_match.valid) - bundle_match.valid.sum()),
        'mode': options.mode,
    }
    return 'bundle', details


def is_realsense_single_topic_base(base: str) -> bool:
    return base.startswith('realsense:') and base != 'realsense:bundle'


def emit_source(
    source: Source,
    source_match: Match,
    t_ns: np.ndarray,
    arrays: dict[str, np.ndarray],
    stats: dict[str, dict[str, Any]],
    valid_masks: list[np.ndarray],
    valid_output_name: str | None = None,
    project_invalid_rows: bool = False,
) -> None:
    positions = source_match.position
    valid = source_match.valid
    present = positions >= 0
    matched = present & valid
    project_present = present if project_invalid_rows else matched
    arrays[f'{source.name}_index'] = source_match.index
    arrays[f'{source.name}_time_ns'] = source_match.time_ns
    arrays[f'{source.name}_delta_ns'] = source_match.delta_ns
    arrays[f'{source.name}_valid'] = valid
    if valid_output_name:
        arrays[valid_output_name] = valid
    arrays.update(project_columns(source.columns, positions, project_present))
    for child in source.children.values():
        emit_child_source(child, positions, project_present, valid, t_ns, arrays, stats, valid_masks)
    valid_masks.append(valid)
    stats[source.name] = entity_stats(source.display_name, len(source.time_ns), source_match, source.details)


def project_columns(columns: dict[str, np.ndarray], positions: np.ndarray, present: np.ndarray) -> dict[str, np.ndarray]:
    return {
        key: project_table_column(values, positions, present)
        for key, values in columns.items()
    }


def project_table_column(values: np.ndarray, positions: np.ndarray, present: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {'U', 'S', 'O'}:
        result = np.asarray(['invalid'] * len(positions), dtype=values.dtype)
        result[present] = values[positions[present]]
        return result
    if values.dtype.kind == 'b':
        result = np.zeros(len(positions), dtype=bool)
        result[present] = values[positions[present]]
        return result
    if values.dtype.kind == 'u':
        result = np.full(len(positions), np.iinfo(values.dtype).max, dtype=values.dtype)
        result[present] = values[positions[present]]
        return result
    result = np.full(len(positions), -1, dtype=values.dtype)
    result[present] = values[positions[present]]
    return result


def emit_child_source(
    child: ChildSource,
    parent_positions: np.ndarray,
    matched: np.ndarray,
    parent_valid: np.ndarray,
    t_ns: np.ndarray,
    arrays: dict[str, np.ndarray],
    stats: dict[str, dict[str, Any]],
    valid_masks: list[np.ndarray],
) -> None:
    child_match = project_child_match(child, parent_positions, matched, parent_valid, t_ns)
    arrays[f'{child.name}_index'] = child_match.index
    arrays[f'{child.name}_time_ns'] = child_match.time_ns
    arrays[f'{child.name}_delta_ns'] = child_match.delta_ns
    arrays[f'{child.name}_valid'] = child_match.valid
    arrays.update(project_columns(child.columns, parent_positions, matched))
    valid_masks.append(child_match.valid)
    stats[child.name] = entity_stats(child.display_name, len(child.time_ns), child_match, child.details)


def project_child_match(
    child: ChildSource,
    positions: np.ndarray,
    matched: np.ndarray,
    parent_valid: np.ndarray,
    t_ns: np.ndarray,
) -> Match:
    index = np.full(len(t_ns), -1, dtype=np.int64)
    position = np.full(len(t_ns), -1, dtype=np.int64)
    time_ns = np.full(len(t_ns), -1, dtype=np.int64)
    index[matched] = child.source_index[positions[matched]]
    position[matched] = positions[matched]
    time_ns[matched] = child.time_ns[positions[matched]]
    source_valid = np.zeros(len(t_ns), dtype=bool)
    source_valid[matched] = (child.time_ns[positions[matched]] > 0) & child.row_valid[positions[matched]]
    return Match(index, position, time_ns, time_ns - t_ns, parent_valid & matched & source_valid)


def align_xense_pair_group(
    xense_pair: Source | None,
    t_ns: np.ndarray,
    options: AlignmentOptions,
    arrays: dict[str, np.ndarray],
    stats: dict[str, dict[str, Any]],
    valid_masks: list[np.ndarray],
) -> str:
    if xense_pair is None:
        return 'absent'
    if 'xense_0' not in xense_pair.children or 'xense_1' not in xense_pair.children:
        raise RuntimeError('Xense pair alignment requires xense_0 and xense_1 streams')
    pair_match = match_source(t_ns, xense_pair, options.mode)
    emit_source(xense_pair, pair_match, t_ns, arrays, stats, valid_masks)
    pair_delta_ms = np.full(len(t_ns), np.nan, dtype=np.float64)
    present = pair_match.position >= 0
    pair_delta_ms[present] = pair_match.delta_ns[present].astype(np.float64) / 1_000_000.0
    arrays['xense_pair_delta_ms'] = pair_delta_ms
    return 'same_row_pair'


def align_scalar_sources(
    sources: dict[str, Source],
    t_ns: np.ndarray,
    options: AlignmentOptions,
    arrays: dict[str, np.ndarray],
    stats: dict[str, dict[str, Any]],
    valid_masks: list[np.ndarray],
) -> str:
    scalar_count = 0
    for name, source in sources.items():
        if name.startswith('realsense_') or name in {'xense_0', 'xense_1'}:
            continue
        emit_source(source, match_source(t_ns, source, options.mode), t_ns, arrays, stats, valid_masks)
        scalar_count += 1
    return 'matched_streams' if scalar_count else 'absent'


def build_realsense_bundle_source(
    sources: dict[str, Source],
    manifest: dict[str, Any],
    warnings: list[str],
    start_ns: int,
    end_ns: int,
) -> Source:
    required_topics = required_image_topics(manifest)
    if not required_topics:
        raise RuntimeError('realsense:bundle requires manifest required image topics')
    missing_topics = []
    non_image_sources = []
    for topic in required_topics:
        source = sources.get(realsense_stream_name(topic))
        if source is None:
            missing_topics.append(topic)
        elif source.details.get('timestamp_source') != 'rosbag_image':
            non_image_sources.append(topic)
    if missing_topics:
        raise RuntimeError(f'realsense:bundle requires readable rosbag image topics; missing: {missing_topics}')
    if non_image_sources:
        raise RuntimeError(f'realsense:bundle requires image-topic header timestamps for every required topic; not rosbag image: {non_image_sources}')

    camera_topics = group_realsense_topics_by_camera(required_topics)
    representative_topics: dict[str, str] = {}
    representative_sources: dict[str, Source] = {}
    for camera, topics in camera_topics.items():
        color_topics = [topic for topic in topics if '/color/image_raw' in topic]
        if not color_topics:
            raise RuntimeError(f'realsense:bundle requires a color image topic for camera {camera!r}')
        topic = sorted(color_topics)[0]
        representative_topics[camera] = topic
        representative_sources[camera] = sources[realsense_stream_name(topic)]

    start_ns = max(max(source.time_ns[0] for source in representative_sources.values()), start_ns)
    end_ns = min(min(source.time_ns[-1] for source in representative_sources.values()), end_ns)
    if end_ns < start_ns:
        return empty_source('realsense_bundle', 'RealSense visual bundle', 'realsense_bundle')

    cameras = sorted(representative_sources)
    initial = choose_initial_bundle(cameras, representative_sources, start_ns)
    if initial is None:
        return empty_source('realsense_bundle', 'RealSense visual bundle', 'realsense_bundle')

    selected_indices: list[dict[str, int]] = []
    modes: list[int] = []
    mode_labels: list[str] = []
    resync: list[bool] = []
    current = initial
    current_mode = 0
    current_mode_label = 'initial_search'
    current_resync = False
    last_t = -1
    while True:
        bundle_time = max(int(representative_sources[camera].time_ns[current[camera]]) for camera in cameras)
        if bundle_time > end_ns:
            break
        if bundle_time > last_t:
            selected_indices.append(dict(current))
            modes.append(current_mode)
            mode_labels.append(current_mode_label)
            resync.append(current_resync)
            last_t = bundle_time
        expected = {camera: current[camera] + 1 for camera in cameras}
        if any(expected[camera] >= len(representative_sources[camera].time_ns) for camera in cameras):
            break
        locked_span = bundle_span_ns(cameras, representative_sources, expected)
        if locked_span <= REALSENSE_BUNDLE_SPAN_WARN_NS:
            current = expected
            current_mode = 1
            current_mode_label = 'locked_plus_one'
            current_resync = False
            continue
        fallback = choose_local_bundle(cameras, representative_sources, expected, last_t, end_ns)
        if fallback is None:
            current = expected
            current_mode = 3
            current_mode_label = 'degraded_best_effort'
            current_resync = False
            if max(int(representative_sources[camera].time_ns[current[camera]]) for camera in cameras) <= last_t:
                break
            continue
        current = fallback
        current_mode = 2
        current_mode_label = 'fallback_search'
        current_resync = True

    if not selected_indices:
        return empty_source('realsense_bundle', 'RealSense visual bundle', 'realsense_bundle')

    bundle_time_ns = np.asarray(
        [max(int(representative_sources[camera].time_ns[indices[camera]]) for camera in cameras) for indices in selected_indices],
        dtype=np.int64,
    )
    bundle_span = np.asarray(
        [bundle_span_ns(cameras, representative_sources, indices) for indices in selected_indices],
        dtype=np.int64,
    )
    degraded = bundle_span > REALSENSE_BUNDLE_SPAN_WARN_NS
    mode_code = np.asarray(modes, dtype=np.uint8)
    resync_array = np.asarray(resync, dtype=bool)

    arrays: dict[str, np.ndarray] = {
        'realsense_bundle_time_ns': bundle_time_ns,
        'realsense_bundle_span_ns': bundle_span,
        'realsense_bundle_mode_code': mode_code,
    }
    children: dict[str, ChildSource] = {}
    selected_recorded_times: list[np.ndarray] = []
    bundle_row_valid = np.ones(len(bundle_time_ns), dtype=bool)
    reused_row = np.zeros(len(bundle_time_ns), dtype=bool)
    for camera in cameras:
        rep_source = representative_sources[camera]
        rep_positions = np.asarray([indices[camera] for indices in selected_indices], dtype=np.int64)
        arrays[f'realsense_bundle_{safe_key(camera)}_index'] = rep_source.source_index[rep_positions]
        arrays[f'realsense_bundle_{safe_key(camera)}_time_ns'] = rep_source.time_ns[rep_positions]
        arrays[f'realsense_bundle_{safe_key(camera)}_recorded_time_ns'] = rep_source.columns[f'{rep_source.name}_recorded_time_ns'][rep_positions]
        rep_times = rep_source.time_ns[rep_positions]
        for topic in camera_topics[camera]:
            source = sources[realsense_stream_name(topic)]
            if topic == representative_topics[camera]:
                positions = rep_positions
                has_causal = np.ones(len(rep_times), dtype=bool)
            else:
                positions, has_causal = causal_positions_for_times(source.time_ns, rep_times)
            child_time_ns = np.full(len(rep_times), -1, dtype=np.int64)
            child_time_ns[has_causal] = source.time_ns[positions[has_causal]]
            child_source_index = np.full(len(rep_times), -1, dtype=np.int64)
            child_source_index[has_causal] = source.source_index[positions[has_causal]]
            exact_stamp = has_causal & (child_time_ns == rep_times)
            invalid_count = int(np.count_nonzero(~exact_stamp))
            if invalid_count:
                missing_causal_count = int(np.count_nonzero(~has_causal))
                stamp_mismatch_count = int(np.count_nonzero(has_causal & (child_time_ns != rep_times)))
                warnings.append(
                    f'RealSense bundle camera {camera} topic {topic} has {invalid_count} invalid frame(s): '
                    f'{stamp_mismatch_count} header stamp mismatch, {missing_causal_count} missing causal frame'
                )
            bundle_row_valid &= exact_stamp
            if len(child_source_index) > 1:
                topic_reused = np.zeros(len(child_source_index), dtype=bool)
                topic_reused[1:] = (child_source_index[1:] >= 0) & (child_source_index[1:] == child_source_index[:-1])
                reused_row |= topic_reused
            child_columns = {
                key: project_table_column(value, positions, has_causal)
                for key, value in source.columns.items()
            }
            children[source.name] = ChildSource(
                source.name,
                source.display_name,
                child_time_ns,
                child_source_index,
                child_columns,
                dict(source.details),
                exact_stamp,
            )
            selected_recorded_times.append(child_columns[f'{source.name}_recorded_time_ns'])

    recorded_stack = np.vstack(selected_recorded_times)
    recorded_time_ns = recorded_stack.max(axis=0).astype(np.int64, copy=False)
    recorded_span_ns = (recorded_stack.max(axis=0) - recorded_stack.min(axis=0)).astype(np.int64, copy=False)
    recorded_valid = np.all(recorded_stack >= 0, axis=0)
    recorded_time_ns[~recorded_valid] = -1
    recorded_span_ns[~recorded_valid] = -1
    arrays['realsense_bundle_recorded_time_ns'] = recorded_time_ns
    arrays['realsense_bundle_recorded_span_ns'] = recorded_span_ns
    invalid_timestamp_mismatch = ~bundle_row_valid
    quality = np.where(
        invalid_timestamp_mismatch,
        'invalid_timestamp_mismatch',
        np.where(degraded, 'degraded_span', 'ok'),
    ).astype('<U32')
    arrays['realsense_bundle_quality'] = quality

    details = {
        'kind': 'realsense_bundle',
        'required_topics': required_topics,
        'representative_topics': representative_topics,
        'bundle_count': int(len(bundle_time_ns)),
        'span_ns': numeric_summary(bundle_span),
        'recorded_span_ns': numeric_summary(recorded_span_ns),
        'mode_counts': value_counts(np.asarray(mode_labels, dtype='<U32')),
        'quality_counts': value_counts(quality),
        'resync_count': int(resync_array.sum()),
        'degraded_count': int(degraded.sum()),
        'invalid_timestamp_mismatch_count': int(invalid_timestamp_mismatch.sum()),
        'reused_count': int(reused_row.sum()),
        'span_warn_ns': REALSENSE_BUNDLE_SPAN_WARN_NS,
    }
    return Source(
        'realsense_bundle',
        'RealSense visual bundle',
        bundle_time_ns,
        np.arange(len(bundle_time_ns), dtype=np.int64),
        66_700_000,
        33_400_000,
        arrays,
        children,
        details,
        bundle_row_valid,
    )


def match_source(t_ns: np.ndarray, source: Source, mode: str) -> Match:
    return match_timestamps(
        t_ns,
        source.time_ns,
        source.source_index,
        source.row_valid,
        source.tolerance_causal_ns,
        source.tolerance_nearest_ns,
        mode,
    )


def match_timestamps(
    t_ns: np.ndarray,
    source_time_ns: np.ndarray,
    source_index: np.ndarray,
    source_row_valid: np.ndarray,
    tolerance_causal_ns: int,
    tolerance_nearest_ns: int,
    mode: str,
) -> Match:
    if len(source_time_ns) == 0:
        empty_time = np.full(len(t_ns), -1, dtype=np.int64)
        empty_valid = np.zeros(len(t_ns), dtype=bool)
        return Match(
            np.full(len(t_ns), -1, dtype=np.int64),
            np.full(len(t_ns), -1, dtype=np.int64),
            empty_time,
            empty_time - t_ns,
            empty_valid,
        )
    right = np.searchsorted(source_time_ns, t_ns, side='right')
    if mode == 'causal':
        chosen = right - 1
    elif mode == 'nearest':
        left = np.maximum(right - 1, 0)
        next_ = np.minimum(right, len(source_time_ns) - 1)
        chosen = np.where(np.abs(source_time_ns[next_] - t_ns) < np.abs(source_time_ns[left] - t_ns), next_, left)
    else:
        raise ValueError(f'unsupported mode: {mode}')
    valid_index = (chosen >= 0) & (chosen < len(source_time_ns))
    matched_time = np.full(len(t_ns), -1, dtype=np.int64)
    matched_time[valid_index] = source_time_ns[chosen[valid_index]]
    delta_ns = matched_time - t_ns
    tolerance = tolerance_causal_ns if mode == 'causal' else tolerance_nearest_ns
    valid_row = np.zeros(len(t_ns), dtype=bool)
    valid_row[valid_index] = source_row_valid[chosen[valid_index]]
    valid = valid_index & valid_row & (np.abs(delta_ns) <= tolerance)
    if mode == 'causal':
        valid &= delta_ns <= 0
    index = np.full(len(t_ns), -1, dtype=np.int64)
    index[valid_index] = source_index[chosen[valid_index]]
    position = np.full(len(t_ns), -1, dtype=np.int64)
    position[valid_index] = chosen[valid_index]
    return Match(index, position, matched_time, delta_ns, valid)


def group_realsense_topics_by_camera(topics: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for topic in topics:
        parts = [part for part in topic.split('/') if part]
        camera = parts[0] if parts else 'camera'
        result.setdefault(camera, []).append(topic)
    return result


def choose_initial_bundle(
    cameras: list[str],
    sources: dict[str, Source],
    start_ns: int,
) -> dict[str, int] | None:
    candidate_lists: list[list[int]] = []
    for camera in cameras:
        times = sources[camera].time_ns
        center = int(np.searchsorted(times, start_ns, side='left'))
        candidates = bounded_index_window(len(times), center, REALSENSE_BUNDLE_INITIAL_SEARCH_RADIUS)
        candidate_lists.append(candidates)
    return choose_best_bundle(cameras, sources, candidate_lists, min_time_ns=start_ns, max_time_ns=None)


def choose_local_bundle(
    cameras: list[str],
    sources: dict[str, Source],
    expected: dict[str, int],
    last_time_ns: int,
    end_ns: int,
) -> dict[str, int] | None:
    candidate_lists = [
        bounded_index_window(len(sources[camera].time_ns), expected[camera], REALSENSE_BUNDLE_FALLBACK_SEARCH_RADIUS)
        for camera in cameras
    ]
    return choose_best_bundle(cameras, sources, candidate_lists, min_time_ns=last_time_ns + 1, max_time_ns=end_ns)


def choose_best_bundle(
    cameras: list[str],
    sources: dict[str, Source],
    candidate_lists: list[list[int]],
    min_time_ns: int,
    max_time_ns: int | None,
) -> dict[str, int] | None:
    best: tuple[int, int, dict[str, int]] | None = None
    for combo in product(*candidate_lists):
        indices = dict(zip(cameras, combo))
        times = [int(sources[camera].time_ns[indices[camera]]) for camera in cameras]
        bundle_time = max(times)
        if bundle_time < min_time_ns:
            continue
        if max_time_ns is not None and bundle_time > max_time_ns:
            continue
        span = max(times) - min(times)
        score = (span, bundle_time)
        if best is None or score < (best[0], best[1]):
            best = (span, bundle_time, indices)
    return None if best is None else best[2]


def bounded_index_window(length: int, center: int, radius: int) -> list[int]:
    start = max(0, center - radius)
    stop = min(length - 1, center + radius)
    if stop < start:
        return []
    return list(range(start, stop + 1))


def bundle_span_ns(cameras: list[str], sources: dict[str, Source], indices: dict[str, int]) -> int:
    times = [int(sources[camera].time_ns[indices[camera]]) for camera in cameras]
    return max(times) - min(times)


def causal_positions_for_times(stream_time_ns: np.ndarray, target_time_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(stream_time_ns, target_time_ns, side='right') - 1
    valid = positions >= 0
    return np.maximum(positions, 0).astype(np.int64), valid


def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {'count': 0, 'min': None, 'median': None, 'p95': None, 'max': None}
    return {
        'count': int(len(values)),
        'min': int(np.min(values)),
        'median': float(np.median(values)),
        'p95': float(np.percentile(values, 95)),
        'max': int(np.max(values)),
    }


def value_counts(values: np.ndarray) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result


def entity_stats(display_name: str, frame_count: int, match: Match, details: dict[str, Any] | None = None) -> dict[str, Any]:
    valid = match.valid
    abs_delta = np.abs(match.delta_ns[valid])
    result = {
        'display_name': display_name,
        'frame_count': int(frame_count),
        'used_count': int(valid.sum()),
        'invalid_count': int(len(valid) - valid.sum()),
        'max_abs_delta_ns': None if len(abs_delta) == 0 else int(abs_delta.max()),
        'mean_abs_delta_ns': None if len(abs_delta) == 0 else float(abs_delta.mean()),
        'median_abs_delta_ns': None if len(abs_delta) == 0 else float(np.median(abs_delta)),
    }
    if details:
        for key in ('topic', 'timestamp_source'):
            if details.get(key) is not None:
                result[key] = details[key]
    return result


def clock_domain_summary(npz_path: Path | None, warnings: list[str]) -> dict[str, Any]:
    if npz_path is None:
        return {}
    data = np.load(npz_path, allow_pickle=True)
    domains = np.asarray(data['clock_domain']).astype(str)
    counts: dict[str, int] = {}
    missing = 0
    for value in domains:
        key = value if value and value != 'None' else '<missing>'
        counts[key] = counts.get(key, 0) + 1
        if key == '<missing>':
            missing += 1
    if missing:
        warnings.append(f'RealSense metadata clock_domain missing on {missing} frame(s)')
    return {'counts': counts, 'missing_count': missing}


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        '# Alignment Report',
        '',
        f"Schema version: {manifest.get('schema_version', 1)}",
        f"Status: {manifest['status']}",
        f"Base: {manifest['base']}",
        f"Base kind: {manifest.get('base_kind', 'stream')}",
        f"RealSense alignment kind: {manifest.get('realsense_alignment_kind', 'unknown')}",
        f"Xense alignment kind: {manifest.get('xense_alignment_kind', 'unknown')}",
        f"Samples: {manifest['valid_count']} / {manifest['sample_count']} valid",
        '',
        '## Streams',
    ]
    for name, stats in manifest['streams'].items():
        median = stats.get('median_abs_delta_ns')
        median_ms = 'n/a' if median is None else f'{median / 1e6:.3f}'
        lines.append(f"- {stats['display_name']} (`{name}`): used {stats['used_count']}/{stats['frame_count']}, median abs delta {median_ms} ms")
    if manifest.get('zmq_clock_offsets'):
        lines.extend(['', '## ZMQ Clock Offsets'])
        for name, offset in sorted(manifest['zmq_clock_offsets'].items()):
            lines.append(
                f"- ZMQ source {offset['source']} (`{name}`): "
                f"offset {offset['offset_ms']:.3f} ms over {offset['frame_count']} frame(s)"
            )
    if manifest.get('clock_domain'):
        lines.extend(['', '## RealSense Clock Domain', json.dumps(manifest['clock_domain'].get('counts', {}), ensure_ascii=True)])
    base_details = manifest.get('base_details') or {}
    realsense_details = manifest.get('realsense_details') or {}
    bundle_details = realsense_details if realsense_details.get('kind') == 'bundle' else base_details
    if bundle_details.get('kind') in {'bundle', 'realsense_bundle'}:
        span = bundle_details.get('span_ns') or {}
        recorded_span = bundle_details.get('recorded_span_ns') or {}
        projection = bundle_details.get('projection') or {}
        lines.extend(
            [
                '',
                '## RealSense Bundle',
                f"Bundles: {bundle_details.get('bundle_count', 0)}",
                (
                    'Header span ns: '
                    f"median={span.get('median')}, p95={span.get('p95')}, max={span.get('max')}"
                ),
                (
                    'Recorded span ns: '
                    f"median={recorded_span.get('median')}, p95={recorded_span.get('p95')}, max={recorded_span.get('max')}"
                ),
                f"Modes: {json.dumps(bundle_details.get('mode_counts', {}), ensure_ascii=True)}",
                f"Quality: {json.dumps(bundle_details.get('quality_counts', {}), ensure_ascii=True)}",
                (
                    f"Resync/degraded/reused: {bundle_details.get('resync_count', 0)} / "
                    f"{bundle_details.get('degraded_count', 0)} / {bundle_details.get('reused_count', 0)}"
                ),
            ]
        )
        if projection:
            lines.append(
                'Projection: '
                f"matched={projection.get('matched_count', 0)}, "
                f"invalid={projection.get('invalid_count', 0)}, "
                f"mode={projection.get('mode', 'n/a')}"
            )
    elif realsense_details.get('kind') == 'single_topic':
        lines.extend(['', '## RealSense Single Topic', f"Topic: {realsense_details.get('topic')}"])
    if base_details.get('kind') == 'xense_pair':
        lines.extend(['', '## Xense Pair Base', f"Pairs: {base_details.get('pair_count', 0)}"])
    if manifest.get('warnings'):
        lines.extend(['', '## Warnings'])
        lines.extend(f"- {warning}" for warning in manifest['warnings'])
    return '\n'.join(lines) + '\n'


def resolve_npz_paths(demo_dir: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key, value in (manifest.get('npz') or {}).items():
        path = Path(value)
        if not path.is_absolute():
            path = demo_dir / path
        if path.exists():
            result[key] = path
    return result


def source_paths(manifest: dict[str, Any]) -> dict[str, Any]:
    sensor_paths = manifest.get('sensor_paths') or {}
    return {
        'npz': dict(manifest.get('npz') or {}),
        'ft300s_saved_file': sensor_paths.get('ft300'),
        'xense_saved_file': sensor_paths.get('xense'),
        'rosbag_uri': manifest.get('rosbag_uri'),
    }


def resolve_rosbag_uri(demo_dir: Path, manifest: dict[str, Any]) -> Path | None:
    value = manifest.get('rosbag_uri')
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else demo_dir / path


def relative_to(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def required_image_topics(manifest: dict[str, Any]) -> list[str]:
    postcheck = manifest.get('realsense_rosbag_postcheck') or {}
    readiness = manifest.get('realsense_image_readiness') or {}
    return [str(topic) for topic in (postcheck.get('required_topics') or readiness.get('required_topics') or [])]


def realsense_stream_name(topic: str) -> str:
    parts = [part for part in topic.split('/') if part]
    camera = parts[0] if parts else 'camera'
    if 'color' in parts:
        role = 'color'
    elif 'aligned_depth_to_color' in parts:
        role = 'aligned_depth'
    elif 'depth' in parts:
        role = 'depth'
    else:
        role = 'stream'
    return f'realsense_{safe_key(camera)}_{role}'


def safe_key(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]+', '_', value).strip('_').lower()


def int_array(values: Any) -> np.ndarray:
    result = []
    for value in values:
        try:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                result.append(-1)
            else:
                result.append(int(value))
        except Exception:
            result.append(-1)
    return np.asarray(result, dtype=np.int64)


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * NSEC_PER_SEC + int(stamp.nanosec)


def detect_storage_id(bag_dir: Path) -> str:
    metadata_file = bag_dir / 'metadata.yaml'
    if metadata_file.exists():
        match = re.search(r'storage_identifier:\s*([A-Za-z0-9_\-]+)', metadata_file.read_text(encoding='utf-8', errors='ignore'))
        if match:
            return match.group(1)
    if list(bag_dir.glob('*.mcap')):
        return 'mcap'
    return 'sqlite3'


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')
