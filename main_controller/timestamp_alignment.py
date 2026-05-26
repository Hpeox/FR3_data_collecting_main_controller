"""Offline timestamp alignment for one completed MainController demo."""

from __future__ import annotations

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


@dataclass(frozen=True)
class AlignmentOptions:
    """Configuration for timestamp-index generation."""

    repo_root: Path | None = None
    output_dir: Path | None = None
    base: str = 'auto'
    alignment_base_source: str = 'realsense'
    mode: str = 'causal'
    hz: float = 30.0
    start_trim_s: float = 0.0
    stream_start_trim: dict[str, float] = field(default_factory=dict)
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
    warnings: tuple[str, ...] = ()

    def to_manifest_entry(self, started_ns: int, finished_ns: int) -> dict[str, Any]:
        """Return the manifest.alignment entry for this result."""
        return {
            'status': self.status,
            'config_path': _relative_to(self.config_path, self.demo_dir),
            'index_path': _relative_to(self.index_path, self.demo_dir),
            'manifest_path': _relative_to(self.manifest_path, self.demo_dir),
            'report_path': _relative_to(self.report_path, self.demo_dir),
            'started_ns': started_ns,
            'finished_ns': finished_ns,
            'sample_count': self.sample_count,
            'valid_count': self.valid_count,
            'base': self.base,
            'warnings': list(self.warnings),
        }


@dataclass
class StreamTable:
    """One normalized timestamp stream."""

    name: str
    display_name: str
    time_ns: np.ndarray
    source_index: np.ndarray
    tolerance_causal_ns: int
    tolerance_nearest_ns: int
    frame_number: np.ndarray | None = None
    topic: str | None = None

    def sorted_valid(self) -> 'StreamTable':
        """Return a copy with invalid timestamps removed and rows sorted."""
        valid = self.time_ns > 0
        order = np.argsort(self.time_ns[valid], kind='stable')
        indices = np.nonzero(valid)[0][order]
        frame_number = None if self.frame_number is None else self.frame_number[indices]
        return StreamTable(
            name=self.name,
            display_name=self.display_name,
            time_ns=self.time_ns[indices].astype(np.int64, copy=False),
            source_index=self.source_index[indices].astype(np.int64, copy=False),
            tolerance_causal_ns=self.tolerance_causal_ns,
            tolerance_nearest_ns=self.tolerance_nearest_ns,
            frame_number=frame_number,
            topic=self.topic,
        )


def align_demo_timestamps(demo_dir: Path, options: AlignmentOptions | None = None) -> AlignmentResult:
    """Generate alignment config, index, manifest, and report for one demo."""
    options = options or AlignmentOptions()
    demo_dir = demo_dir.resolve()
    output_dir = (options.output_dir or demo_dir / 'aligned').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = demo_dir / 'manifest.json'
    manifest = _read_json(manifest_path)
    if manifest.get('status') != 'done' and not options.allow_degraded:
        raise RuntimeError(f"alignment requires manifest.status == 'done', got {manifest.get('status')!r}")

    warnings: list[str] = []
    npz_paths = _resolve_npz_paths(demo_dir, manifest)
    streams = _load_streams(demo_dir, manifest, npz_paths, warnings)
    if not streams:
        raise RuntimeError('no timestamp streams found for alignment')

    resolved_base = _resolve_base(options, manifest, streams)
    base_stream = _base_stream(resolved_base, streams)
    if base_stream is None:
        raise RuntimeError(f'base stream has no data: {resolved_base}')

    t_ns = _target_times(base_stream, streams, options)
    if len(t_ns) == 0:
        raise RuntimeError('target timeline is empty after trims')

    index_arrays: dict[str, np.ndarray] = {
        't_ns': t_ns,
        'segment_id': np.zeros(len(t_ns), dtype=np.int64),
    }
    stream_stats: dict[str, dict[str, Any]] = {}
    valid_masks: list[np.ndarray] = []
    for stream in streams.values():
        match = _match_stream(t_ns, stream, options.mode)
        prefix = stream.name
        index_arrays[f'{prefix}_index'] = match['index']
        index_arrays[f'{prefix}_time_ns'] = match['time_ns']
        index_arrays[f'{prefix}_delta_ns'] = match['delta_ns']
        index_arrays[f'{prefix}_valid'] = match['valid']
        if stream.frame_number is not None:
            frame_number = np.full(len(t_ns), -1, dtype=np.int64)
            good = match['position'] >= 0
            frame_number[good] = stream.frame_number[match['position'][good]]
            index_arrays[f'{prefix}_frame_number'] = frame_number
        if stream.topic is not None:
            index_arrays[f'{prefix}_topic'] = np.asarray([stream.topic] * len(t_ns))
        valid_masks.append(match['valid'])
        stream_stats[prefix] = _stream_stats(stream, match)

    sample_valid = np.logical_and.reduce(valid_masks) if valid_masks else np.ones(len(t_ns), dtype=bool)
    index_arrays['sample_valid'] = sample_valid
    index_path = output_dir / 'aligned_index.npz'
    np.savez(index_path, **index_arrays)

    sources = _source_paths(manifest)
    alignment_config = {
        'demo_dir': '.',
        'output_dir': _relative_to(output_dir, demo_dir),
        'base': resolved_base,
        'requested_base': options.base,
        'alignment_base_source': options.alignment_base_source,
        'mode': options.mode,
        'hz': options.hz,
        'start_trim_s': options.start_trim_s,
        'stream_start_trim': options.stream_start_trim,
        'sources': sources,
        'streams': {name: {'display_name': stream.display_name, 'topic': stream.topic} for name, stream in streams.items()},
    }
    config_path = output_dir / 'alignment_config.json'
    _write_json(config_path, alignment_config)

    aligned_manifest = {
        'status': 'done',
        'demo_dir': '.',
        'sample_count': int(len(t_ns)),
        'valid_count': int(sample_valid.sum()),
        'base': resolved_base,
        'mode': options.mode,
        'hz': options.hz,
        'sources': sources,
        'streams': stream_stats,
        'clock_domain': _clock_domain_summary(npz_paths.get('realsense'), warnings),
        'drop_monitors': manifest.get('drop_monitors', {}),
        'warnings': warnings,
    }
    aligned_manifest_path = output_dir / 'aligned_manifest.json'
    _write_json(aligned_manifest_path, aligned_manifest)

    report_path = output_dir / 'alignment_report.md'
    report_path.write_text(_render_report(aligned_manifest), encoding='utf-8')

    return AlignmentResult(
        demo_dir=demo_dir,
        status='done',
        config_path=config_path,
        index_path=index_path,
        manifest_path=aligned_manifest_path,
        report_path=report_path,
        sample_count=int(len(t_ns)),
        valid_count=int(sample_valid.sum()),
        base=resolved_base,
        warnings=tuple(warnings),
    )


def update_manifest_alignment(manifest_path: Path, entry: dict[str, Any]) -> None:
    """Update one demo manifest with an independent alignment status entry."""
    manifest = _read_json(manifest_path)
    manifest['alignment'] = entry
    _write_json(manifest_path, manifest)


def failure_manifest_entry(started_ns: int, error: Exception) -> dict[str, Any]:
    """Return a manifest.alignment failure entry."""
    return {
        'status': 'failed',
        'started_ns': started_ns,
        'finished_ns': time.time_ns(),
        'error': str(error),
    }


def _load_streams(demo_dir: Path, manifest: dict[str, Any], npz_paths: dict[str, Path], warnings: list[str]) -> dict[str, StreamTable]:
    streams: dict[str, StreamTable] = {}
    if 'ft300' in npz_paths:
        data = np.load(npz_paths['ft300'], allow_pickle=True)
        streams['ft300s'] = StreamTable(
            name='ft300s',
            display_name='FT300S',
            time_ns=_int_array(data['timestamp_ns']),
            source_index=np.arange(len(data['timestamp_ns']), dtype=np.int64),
            tolerance_causal_ns=20_000_000,
            tolerance_nearest_ns=10_000_000,
        ).sorted_valid()

    if 'xense' in npz_paths:
        data = np.load(npz_paths['xense'], allow_pickle=True)
        streams['xense_0'] = StreamTable(
            name='xense_0',
            display_name='Xense sensor 0',
            time_ns=_int_array(data['timestamp_ns_0']),
            source_index=np.arange(len(data['timestamp_ns_0']), dtype=np.int64),
            tolerance_causal_ns=66_700_000,
            tolerance_nearest_ns=33_400_000,
        ).sorted_valid()
        streams['xense_1'] = StreamTable(
            name='xense_1',
            display_name='Xense sensor 1',
            time_ns=_int_array(data['timestamp_ns_1']),
            source_index=np.arange(len(data['timestamp_ns_1']), dtype=np.int64),
            tolerance_causal_ns=66_700_000,
            tolerance_nearest_ns=33_400_000,
        ).sorted_valid()

    if 'zmq' in npz_paths:
        data = np.load(npz_paths['zmq'], allow_pickle=True)
        sources = _int_array(data['source'])
        raw_stamp_ns = np.asarray([int(round(float(value) * NSEC_PER_SEC)) for value in data['stamp_s']], dtype=np.int64)
        recv_time_ns = _int_array(data['recv_time_ns'])
        for source in sorted(set(int(value) for value in sources if value > 0)):
            mask = sources == source
            if not np.any(mask):
                continue
            offset_ns = int(np.median(recv_time_ns[mask] - raw_stamp_ns[mask]))
            time_ns = raw_stamp_ns[mask] + offset_ns
            streams[f'zmq_source_{source}'] = StreamTable(
                name=f'zmq_source_{source}',
                display_name=f'ZMQ source {source}',
                time_ns=time_ns,
                source_index=np.nonzero(mask)[0].astype(np.int64),
                tolerance_causal_ns=40_000_000,
                tolerance_nearest_ns=20_000_000,
            ).sorted_valid()

    streams.update(_realsense_streams(demo_dir, manifest, npz_paths.get('realsense'), warnings))
    return {name: stream for name, stream in streams.items() if len(stream.time_ns) > 0}


def _realsense_streams(demo_dir: Path, manifest: dict[str, Any], npz_path: Path | None, warnings: list[str]) -> dict[str, StreamTable]:
    if npz_path is None:
        return {}
    metadata = np.load(npz_path, allow_pickle=True)
    metadata_by_topic: dict[str, dict[str, np.ndarray]] = {}
    topics = np.asarray(metadata['topic']).astype(str)
    for topic in sorted(set(topics)):
        mask = topics == topic
        metadata_by_topic[topic] = {
            'time_ns': _int_array(metadata['header_stamp_ns'][mask]),
            'source_index': np.nonzero(mask)[0].astype(np.int64),
            'frame_number': _int_array(metadata['frame_number'][mask]),
        }

    required_topics = _required_image_topics(manifest)
    rosbag_streams = _read_rosbag_image_streams(_resolve_rosbag_uri(demo_dir, manifest), required_topics, warnings)
    streams: dict[str, StreamTable] = {}
    for image_topic in required_topics:
        stream_name = _realsense_stream_name(image_topic)
        rosbag_times = rosbag_streams.get(image_topic)
        if rosbag_times is not None and len(rosbag_times) > 0:
            streams[stream_name] = StreamTable(
                name=stream_name,
                display_name=f'RealSense {image_topic}',
                time_ns=rosbag_times,
                source_index=np.arange(len(rosbag_times), dtype=np.int64),
                tolerance_causal_ns=66_700_000,
                tolerance_nearest_ns=33_400_000,
                topic=image_topic,
            ).sorted_valid()
            continue
        metadata_topic = _image_topic_to_metadata_topic(image_topic)
        meta = metadata_by_topic.get(metadata_topic)
        if meta is None:
            continue
        streams[stream_name] = StreamTable(
            name=stream_name,
            display_name=f'RealSense {image_topic}',
            time_ns=meta['time_ns'],
            source_index=meta['source_index'],
            tolerance_causal_ns=66_700_000,
            tolerance_nearest_ns=33_400_000,
            frame_number=meta['frame_number'],
            topic=image_topic,
        ).sorted_valid()

    if not streams:
        for topic, meta in metadata_by_topic.items():
            stream_name = _realsense_stream_name(topic)
            streams[stream_name] = StreamTable(
                name=stream_name,
                display_name=f'RealSense {topic}',
                time_ns=meta['time_ns'],
                source_index=meta['source_index'],
                tolerance_causal_ns=66_700_000,
                tolerance_nearest_ns=33_400_000,
                frame_number=meta['frame_number'],
                topic=topic,
            ).sorted_valid()
    return streams


def _read_rosbag_image_streams(rosbag_uri: Path | None, topics: list[str], warnings: list[str]) -> dict[str, np.ndarray]:
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
        storage_options = rosbag2_py.StorageOptions(uri=str(rosbag_uri), storage_id=_detect_storage_id(rosbag_uri))
        reader.open(storage_options, rosbag2_py.ConverterOptions('', ''))
        topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
        selected = [topic for topic in topics if topic in topic_types]
        if not selected:
            return {}
        reader.set_filter(rosbag2_py.StorageFilter(topics=selected))
        message_classes = {topic: get_message(topic_types[topic]) for topic in selected}
        result: dict[str, list[int]] = {topic: [] for topic in selected}
        while reader.has_next():
            topic, serialized, _recorded_time = reader.read_next()
            message = deserialize_message(serialized, message_classes[topic])
            result[topic].append(_stamp_to_ns(message.header.stamp))
        return {topic: np.asarray(times, dtype=np.int64) for topic, times in result.items()}
    except Exception as exc:
        warnings.append(f'rosbag image header read failed, using metadata fallback: {exc}')
        return {}


def _target_times(base_stream: StreamTable, streams: dict[str, StreamTable], options: AlignmentOptions) -> np.ndarray:
    start_trim_ns = int(round(options.start_trim_s * NSEC_PER_SEC))
    overlap_start = max(stream.time_ns[0] + int(round(options.stream_start_trim.get(stream.name, 0.0) * NSEC_PER_SEC)) for stream in streams.values())
    overlap_end = min(stream.time_ns[-1] for stream in streams.values())
    start_ns = max(base_stream.time_ns[0], overlap_start) + start_trim_ns
    end_ns = min(base_stream.time_ns[-1], overlap_end)
    if end_ns < start_ns:
        return np.asarray([], dtype=np.int64)
    if options.base == 'grid':
        step_ns = int(round(NSEC_PER_SEC / options.hz))
        return np.arange(start_ns, end_ns + 1, step_ns, dtype=np.int64)
    return base_stream.time_ns[(base_stream.time_ns >= start_ns) & (base_stream.time_ns <= end_ns)]


def _match_stream(t_ns: np.ndarray, stream: StreamTable, mode: str) -> dict[str, np.ndarray]:
    times = stream.time_ns
    if mode not in {'causal', 'nearest'}:
        raise ValueError(f'unsupported alignment mode: {mode}')
    right = np.searchsorted(times, t_ns, side='right')
    if mode == 'causal':
        chosen = right - 1
    else:
        left = np.maximum(right - 1, 0)
        next_ = np.minimum(right, len(times) - 1)
        choose_next = np.abs(times[next_] - t_ns) < np.abs(times[left] - t_ns)
        chosen = np.where(choose_next, next_, left)
        chosen = np.where(len(times) > 0, chosen, -1)
    valid_index = (chosen >= 0) & (chosen < len(times))
    matched_time = np.full(len(t_ns), -1, dtype=np.int64)
    matched_time[valid_index] = times[chosen[valid_index]]
    delta_ns = matched_time - t_ns
    tolerance_ns = stream.tolerance_causal_ns if mode == 'causal' else stream.tolerance_nearest_ns
    if mode == 'causal':
        valid = valid_index & (delta_ns <= 0) & (np.abs(delta_ns) <= tolerance_ns)
    else:
        valid = valid_index & (np.abs(delta_ns) <= tolerance_ns)
    index = np.full(len(t_ns), -1, dtype=np.int64)
    index[valid_index] = stream.source_index[chosen[valid_index]]
    position = np.full(len(t_ns), -1, dtype=np.int64)
    position[valid_index] = chosen[valid_index]
    return {'index': index, 'position': position, 'time_ns': matched_time, 'delta_ns': delta_ns, 'valid': valid}


def _resolve_base(options: AlignmentOptions, manifest: dict[str, Any], streams: dict[str, StreamTable]) -> str:
    if options.base != 'auto':
        return options.base
    if options.alignment_base_source == 'xense':
        return 'xense:0'
    required = _required_image_topics(manifest)
    for topic in required:
        if '/color/' in topic:
            stream_name = _realsense_stream_name(topic)
            if stream_name in streams:
                return f'realsense:{topic}'
    for name, stream in streams.items():
        if name.startswith('realsense_'):
            return f'realsense:{stream.topic or name}'
    return 'xense:0' if 'xense_0' in streams else next(iter(streams))


def _base_stream(base: str, streams: dict[str, StreamTable]) -> StreamTable | None:
    if base == 'grid':
        return next(iter(streams.values()))
    if base == 'robot':
        return streams.get('zmq_source_2')
    if base == 'xense:0':
        return streams.get('xense_0')
    if base.startswith('realsense:'):
        target = base.split(':', 1)[1]
        if target == 'auto':
            return next((stream for name, stream in streams.items() if name.startswith('realsense_')), None)
        name = _realsense_stream_name(target)
        return streams.get(name)
    return streams.get(base)


def _stream_stats(stream: StreamTable, match: dict[str, np.ndarray]) -> dict[str, Any]:
    valid = match['valid']
    abs_delta = np.abs(match['delta_ns'][valid])
    if len(abs_delta) == 0:
        delta_stats = {'max_abs_delta_ns': None, 'mean_abs_delta_ns': None, 'median_abs_delta_ns': None}
    else:
        delta_stats = {
            'max_abs_delta_ns': int(abs_delta.max()),
            'mean_abs_delta_ns': float(abs_delta.mean()),
            'median_abs_delta_ns': float(np.median(abs_delta)),
        }
    return {
        'display_name': stream.display_name,
        'frame_count': int(len(stream.time_ns)),
        'used_count': int(valid.sum()),
        'invalid_count': int(len(valid) - valid.sum()),
        **delta_stats,
    }


def _clock_domain_summary(npz_path: Path | None, warnings: list[str]) -> dict[str, Any]:
    if npz_path is None:
        return {}
    data = np.load(npz_path, allow_pickle=True)
    domains = np.asarray(data['clock_domain']).astype(str)
    missing = int(sum(1 for value in domains if not value or value == 'None'))
    if missing:
        warnings.append(f'RealSense metadata clock_domain missing on {missing} frame(s)')
    unique: dict[str, int] = {}
    for value in domains:
        key = value if value and value != 'None' else '<missing>'
        unique[key] = unique.get(key, 0) + 1
    return {'counts': unique, 'missing_count': missing}


def _render_report(aligned_manifest: dict[str, Any]) -> str:
    lines = [
        '# Alignment Report',
        '',
        f"Status: {aligned_manifest['status']}",
        f"Base: {aligned_manifest['base']}",
        f"Samples: {aligned_manifest['valid_count']} / {aligned_manifest['sample_count']} valid",
        '',
        '## Streams',
    ]
    for name, stats in aligned_manifest['streams'].items():
        median = stats.get('median_abs_delta_ns')
        median_ms = 'n/a' if median is None else f'{median / 1e6:.3f}'
        lines.append(f"- {stats['display_name']} (`{name}`): used {stats['used_count']}/{stats['frame_count']}, median abs delta {median_ms} ms")
    clock_domain = aligned_manifest.get('clock_domain') or {}
    if clock_domain:
        lines.extend(['', '## RealSense Clock Domain', json.dumps(clock_domain.get('counts', {}), ensure_ascii=True)])
    warnings = aligned_manifest.get('warnings') or []
    if warnings:
        lines.extend(['', '## Warnings'])
        lines.extend(f'- {warning}' for warning in warnings)
    return '\n'.join(lines) + '\n'


def _source_paths(manifest: dict[str, Any]) -> dict[str, Any]:
    sensor_paths = manifest.get('sensor_paths') or {}
    return {
        'npz': dict(manifest.get('npz') or {}),
        'ft300s_saved_file': sensor_paths.get('ft300'),
        'xense_saved_file': sensor_paths.get('xense'),
        'rosbag_uri': manifest.get('rosbag_uri'),
    }


def _resolve_npz_paths(demo_dir: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key, value in (manifest.get('npz') or {}).items():
        path = Path(value)
        if not path.is_absolute():
            path = demo_dir / path
        if path.exists():
            result[key] = path
    return result


def _resolve_rosbag_uri(demo_dir: Path, manifest: dict[str, Any]) -> Path | None:
    value = manifest.get('rosbag_uri')
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else demo_dir / path


def _relative_to(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _required_image_topics(manifest: dict[str, Any]) -> list[str]:
    postcheck = manifest.get('realsense_rosbag_postcheck') or {}
    readiness = manifest.get('realsense_image_readiness') or {}
    topics = postcheck.get('required_topics') or readiness.get('required_topics') or []
    return [str(topic) for topic in topics]


def _image_topic_to_metadata_topic(topic: str) -> str:
    if '/color/' in topic:
        return topic.replace('/color/image_raw', '/color/metadata')
    return re.sub(r'/aligned_depth_to_color/image_raw$', '/depth/metadata', topic)


def _realsense_stream_name(topic: str) -> str:
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
    return f'realsense_{_safe_key(camera)}_{role}'


def _safe_key(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]+', '_', value).strip('_').lower()


def _int_array(values: Any) -> np.ndarray:
    result: list[int] = []
    for value in values:
        try:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                result.append(-1)
            else:
                result.append(int(value))
        except Exception:
            result.append(-1)
    return np.asarray(result, dtype=np.int64)


def _stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * NSEC_PER_SEC + int(stamp.nanosec)


def _detect_storage_id(bag_dir: Path) -> str:
    metadata_file = bag_dir / 'metadata.yaml'
    if metadata_file.exists():
        content = metadata_file.read_text(encoding='utf-8', errors='ignore')
        match = re.search(r'storage_identifier:\s*([A-Za-z0-9_\-]+)', content)
        if match:
            return match.group(1)
    if list(bag_dir.glob('*.mcap')):
        return 'mcap'
    return 'sqlite3'


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')
