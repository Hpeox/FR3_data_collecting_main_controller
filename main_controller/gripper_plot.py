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


def render_gripper_plot(npz_path: Path, output_path: Path) -> dict[str, object]:
    """Render one PNG and atomically replace the fixed preview."""
    series = load_gripper_series(npz_path)
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
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(13, 7),
            sharex=False,
            gridspec_kw={'height_ratios': [3, 1]},
        )
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
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.suptitle('Gripper mapped command, position, and state')
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

    return {
        'output_path': str(output_path),
        'command_samples': int(series.command_index.size),
        'feedback_samples': int(series.feedback_index.size),
    }


def parse_args() -> argparse.Namespace:
    """Parse the standalone renderer arguments."""
    parser = argparse.ArgumentParser(
        description='Render a gripper preview from zmq_telemetry.npz.'
    )
    parser.add_argument('--npz', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the standalone renderer and print a machine-readable summary."""
    args = parse_args()
    result = render_gripper_plot(args.npz, args.output)
    print(json.dumps(result, separators=(',', ':')))


if __name__ == '__main__':
    main()
