"""Per-frame processing statistics and their ROS diagnostics representation."""

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

COUNT_FIELDS = (
    ('input_points', 'Input points'),
    ('nonfinite_removed', 'Non-finite removed'),
    ('zero_removed', 'Zero (no-return) removed'),
    ('range_removed', 'Range removed'),
    ('crop_removed', 'Crop box removed'),
    ('outliers_removed', 'Outliers removed'),
    ('filtered_points', 'Filtered points'),
    ('voxel_points', 'Voxelized points'),
)


@dataclass
class FrameStats:
    """Point counts and stage timings for one processed frame."""

    input_points: int = 0
    nonfinite_removed: int = 0
    zero_removed: int = 0
    range_removed: int = 0
    crop_removed: int = 0
    outliers_removed: int = 0
    filtered_points: int = 0
    voxel_points: int = 0
    timings_ms: dict = field(default_factory=dict)
    total_ms: float = 0.0

    def as_dict(self):
        return asdict(self)

    def reduction_ratio(self):
        """Output points / input points of the final (voxelized if present) cloud."""
        if self.input_points == 0:
            return 0.0
        final = self.voxel_points if self.voxel_points else self.filtered_points
        return final / self.input_points

    def format_report(self):
        lines = ['Point Cloud Statistics', '─' * 40]
        for key, label in COUNT_FIELDS:
            lines.append(f'{label:<26}: {getattr(self, key):>10,}')
        lines.append('')
        for stage, ms in self.timings_ms.items():
            lines.append(f'{stage + " [ms]":<26}: {ms:>10.2f}')
        lines.append(f'{"total [ms]":<26}: {self.total_ms:>10.2f}')
        return '\n'.join(lines)


class RollingStats:
    """Aggregates the last ``window`` frames for periodic logging and diagnostics."""

    def __init__(self, window=100):
        self._frames = deque(maxlen=int(window))
        self.total_frames = 0

    def add(self, stats):
        self._frames.append(stats)
        self.total_frames += 1

    def __len__(self):
        return len(self._frames)

    def mean_count(self, name):
        if not self._frames:
            return 0.0
        return float(np.mean([getattr(f, name) for f in self._frames]))

    def total_ms(self):
        """(mean, p95, max) end-to-end processing time over the window."""
        if not self._frames:
            return 0.0, 0.0, 0.0
        values = np.array([f.total_ms for f in self._frames])
        return float(values.mean()), float(np.percentile(values, 95)), float(values.max())

    def mean_stage_ms(self):
        stages = {}
        for frame in self._frames:
            for stage, ms in frame.timings_ms.items():
                stages.setdefault(stage, []).append(ms)
        return {stage: float(np.mean(v)) for stage, v in stages.items()}

    def format_summary(self):
        mean_ms, p95_ms, max_ms = self.total_ms()
        lines = [f'Point-cloud processing: last {len(self)} frames '
                 f'({self.total_frames} total)']
        for key, label in COUNT_FIELDS:
            lines.append(f'  {label:<26}: {self.mean_count(key):>12,.0f}')
        stage_text = ', '.join(f'{k}={v:.1f}' for k, v in self.mean_stage_ms().items())
        lines.append(f'  stage means [ms]           : {stage_text}')
        lines.append(f'  total [ms] mean/p95/max    : {mean_ms:.1f} / {p95_ms:.1f} / {max_ms:.1f}')
        return '\n'.join(lines)


def build_diagnostic_array(stats, rolling, stamp, name, hardware_id='', budget_ms=100.0):
    """Build a diagnostic_msgs/msg/DiagnosticArray for one frame (lazy ROS import).

    Level is WARN when the rolling mean processing time exceeds ``budget_ms``
    (at a 10 Hz LiDAR the budget is 100 ms), and when a frame produced no points.
    """
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

    mean_ms, p95_ms, max_ms = rolling.total_ms()
    status = DiagnosticStatus()
    status.name = name
    status.hardware_id = hardware_id
    if stats.input_points > 0 and stats.filtered_points == 0:
        status.level = DiagnosticStatus.WARN
        status.message = 'All points were filtered out'
    elif budget_ms > 0 and mean_ms > budget_ms:
        status.level = DiagnosticStatus.WARN
        status.message = f'Mean processing time {mean_ms:.1f} ms exceeds budget {budget_ms:.1f} ms'
    else:
        status.level = DiagnosticStatus.OK
        status.message = 'OK'

    values = [KeyValue(key=key, value=str(getattr(stats, key))) for key, _ in COUNT_FIELDS]
    values += [KeyValue(key=f'time_{stage}_ms', value=f'{ms:.3f}')
               for stage, ms in stats.timings_ms.items()]
    values += [
        KeyValue(key='time_total_ms', value=f'{stats.total_ms:.3f}'),
        KeyValue(key='time_total_mean_ms', value=f'{mean_ms:.3f}'),
        KeyValue(key='time_total_p95_ms', value=f'{p95_ms:.3f}'),
        KeyValue(key='time_total_max_ms', value=f'{max_ms:.3f}'),
        KeyValue(key='reduction_ratio', value=f'{stats.reduction_ratio():.4f}'),
        KeyValue(key='frames_processed', value=str(rolling.total_frames)),
    ]
    status.values = values

    array_msg = DiagnosticArray()
    array_msg.header.stamp = stamp
    array_msg.status = [status]
    return array_msg
