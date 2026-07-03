"""Render a fixed gripper preview from saved ZMQ telemetry."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SOURCE_GELLO = 1
SOURCE_GRIPPER = 3


@dataclass(frozen=True)
class GripperSeries:
    """Full-rate gripper command and feedback series."""

    command_index: np.ndarray
    mapped_command: np.ndarray
    feedback_index: np.ndarray
    gpo: np.ndarray
    gcu: np.ndarray


@dataclass(frozen=True)
class TactilePreviewSeries:
    """Lightweight tactile force-resultant preview loaded from Xense."""

    sensor_ids: tuple[str, str]
    frame_index: np.ndarray
    force_resultant: np.ndarray
    edge_warning: np.ndarray
    edge_max: np.ndarray


def load_gripper_series(npz_path: Path) -> GripperSeries:
    """Load gripper series using source-local row indices."""
    with np.load(npz_path, allow_pickle=False) as data:
        required = {'source', 'floats_58', 'gripper_gPO', 'gripper_gCU'}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"missing ZMQ fields: {', '.join(missing)}")

        source = np.asarray(data['source'])
        floats = np.asarray(data['floats_58'])
        gpo_all = np.asarray(data['gripper_gPO'])
        gcu_all = np.asarray(data['gripper_gCU'])

    row_count = len(source)
    if floats.ndim != 2 or floats.shape != (row_count, 58):
        raise ValueError(
            f'floats_58 must have shape ({row_count}, 58), got {floats.shape}'
        )
    if len(gpo_all) != row_count or len(gcu_all) != row_count:
        raise ValueError('ZMQ fields must have equal row counts')

    command = np.asarray(floats[source == SOURCE_GELLO, 7], dtype=np.float64)
    command = command[np.isfinite(command)]
    feedback_mask = source == SOURCE_GRIPPER
    gpo = np.asarray(gpo_all[feedback_mask], dtype=np.uint8)
    gcu = np.asarray(gcu_all[feedback_mask], dtype=np.uint8)

    if command.size == 0:
        raise ValueError('no finite GELLO gripper command samples')
    if gpo.size == 0:
        raise ValueError('no gripper feedback samples')

    return GripperSeries(
        command_index=np.arange(command.size, dtype=np.int64),
        mapped_command=(1.0 - command) * 255.0,
        feedback_index=np.arange(gpo.size, dtype=np.int64),
        gpo=gpo,
        gcu=gcu,
    )


def load_tactile_preview(preview_path: Path) -> TactilePreviewSeries:
    """Load a small tactile preview archive."""
    with np.load(preview_path, allow_pickle=False) as data:
        required = {'sensor_ids', 'frame_index', 'force_resultant', 'edge_warning', 'edge_max'}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"missing tactile preview fields: {', '.join(missing)}")
        sensor_ids_array = np.asarray(data['sensor_ids']).astype(str)
        frame_index = np.asarray(data['frame_index'], dtype=np.int64)
        force_resultant = np.asarray(data['force_resultant'], dtype=np.float64)
        edge_warning = np.asarray(data['edge_warning'], dtype=np.bool_)
        edge_max = np.asarray(data['edge_max'], dtype=np.float64)

    if sensor_ids_array.shape != (2,):
        raise ValueError(f'sensor_ids must have shape (2,), got {sensor_ids_array.shape}')
    if force_resultant.ndim != 3 or force_resultant.shape[0] != 2 or force_resultant.shape[2] != 6:
        raise ValueError(f'force_resultant must have shape (2, N, 6), got {force_resultant.shape}')
    if frame_index.shape != (force_resultant.shape[1],):
        raise ValueError(
            f'frame_index must have shape ({force_resultant.shape[1]},), got {frame_index.shape}'
        )
    if edge_warning.shape != (2,):
        raise ValueError(f'edge_warning must have shape (2,), got {edge_warning.shape}')
    if edge_max.shape != (2,):
        raise ValueError(f'edge_max must have shape (2,), got {edge_max.shape}')
    if not np.all(np.isfinite(force_resultant)):
        raise ValueError('force_resultant contains non-finite values')
    return TactilePreviewSeries(
        sensor_ids=(str(sensor_ids_array[0]), str(sensor_ids_array[1])),
        frame_index=frame_index,
        force_resultant=force_resultant,
        edge_warning=edge_warning,
        edge_max=edge_max,
    )


def _mark_warning_axis(axis, sensor_id: str) -> None:
    axis.set_title(f'{sensor_id} force_resultant (edge warning)', color='red')
    for spine in axis.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2.0)


def _plot_gripper_axes(axes, series: GripperSeries) -> None:
    axes[0].plot(
        series.feedback_index,
        series.gpo,
        label='gPO',
        linewidth=1.2,
    )
    axes[0].plot(
        series.command_index,
        series.mapped_command,
        label='mapped command = (1 - command) × 255',
        linewidth=1.0,
        alpha=0.85,
    )
    axes[0].set_ylabel('Gripper position (0–255)')
    axes[0].set_ylim(0, 255)
    axes[0].legend(loc='upper right')
    axes[1].step(
        series.feedback_index,
        series.gcu,
        where='post',
        linewidth=1.0,
    )
    axes[1].set_ylabel('gCU')
    axes[1].set_xlabel('Sample index')


def _plot_tactile_axis(axis, tactile: TactilePreviewSeries, sensor_index: int) -> None:
    sensor_id = tactile.sensor_ids[sensor_index]
    for channel in range(6):
        axis.plot(
            tactile.frame_index,
            tactile.force_resultant[sensor_index, :, channel],
            linewidth=0.9,
            label=f'ch{channel}',
        )
    axis.set_title(f'{sensor_id} force_resultant')
    axis.set_xlabel('Sample index')
    axis.set_ylabel('Force resultant')
    axis.legend(loc='upper right', ncol=3, fontsize='small')
    if bool(tactile.edge_warning[sensor_index]):
        _mark_warning_axis(axis, sensor_id)


def render_gripper_plot(
    npz_path: Path,
    output_path: Path,
    tactile_preview_path: Path | None = None,
) -> dict[str, object]:
    """Render one PNG and atomically replace the fixed preview."""
    series = load_gripper_series(npz_path)
    tactile = None
    tactile_preview_error = None
    tactile_preview_delete_error = None
    if tactile_preview_path is not None:
        try:
            tactile = load_tactile_preview(tactile_preview_path)
        except Exception as exc:
            tactile_preview_error = str(exc)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/main-controller-matplotlib')

    import matplotlib

    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f'.{output_path.stem}.{os.getpid()}.tmp{output_path.suffix}'
    )
    fig = None
    try:
        if tactile is None:
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(13, 7),
                sharex=False,
                gridspec_kw={'height_ratios': [3, 1]},
            )
            _plot_gripper_axes(axes, series)
            fig.suptitle('Gripper mapped command, position, and state')
        else:
            fig, axes_grid = plt.subplots(
                2,
                2,
                figsize=(16, 9),
                sharex=False,
            )
            _plot_gripper_axes([axes_grid[0, 0], axes_grid[1, 0]], series)
            _plot_tactile_axis(axes_grid[0, 1], tactile, 0)
            _plot_tactile_axis(axes_grid[1, 1], tactile, 1)
            axes = axes_grid.ravel()
            fig.suptitle('Gripper preview and tactile force-resultant check')
        for axis in np.ravel(axes):
            axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            temporary_path,
            format='png',
            dpi=120,
            bbox_inches='tight',
        )
        os.replace(temporary_path, output_path)
    finally:
        if fig is not None:
            plt.close(fig)
        temporary_path.unlink(missing_ok=True)
        if tactile_preview_path is not None:
            try:
                tactile_preview_path.unlink(missing_ok=True)
            except Exception as exc:
                tactile_preview_delete_error = str(exc)

    result = {
        'output_path': str(output_path),
        'command_samples': int(series.command_index.size),
        'feedback_samples': int(series.feedback_index.size),
        'tactile_preview_ok': tactile is not None,
    }
    if tactile is not None:
        result['tactile_samples'] = int(tactile.frame_index.size)
        result['tactile_sensor_ids'] = list(tactile.sensor_ids)
    if tactile_preview_error is not None:
        result['tactile_preview_error'] = tactile_preview_error
    if tactile_preview_delete_error is not None:
        result['tactile_preview_delete_error'] = tactile_preview_delete_error
    return result


def parse_args() -> argparse.Namespace:
    """Parse the standalone renderer arguments."""
    parser = argparse.ArgumentParser(
        description='Render a gripper preview from zmq_telemetry.npz.'
    )
    parser.add_argument('--npz', required=True, type=Path)
    parser.add_argument('--tactile-preview-npz', type=Path, default=None)
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the standalone renderer and print a machine-readable summary."""
    args = parse_args()
    result = render_gripper_plot(args.npz, args.output, args.tactile_preview_npz)
    print(json.dumps(result, separators=(',', ':')))


if __name__ == '__main__':
    main()
