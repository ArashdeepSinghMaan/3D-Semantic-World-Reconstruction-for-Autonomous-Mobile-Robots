"""Rolling timing statistics and diagnostic_msgs output for the geometry node."""

from collections import deque

import numpy as np


class RollingTimes:

    def __init__(self, window=100):
        self._totals = deque(maxlen=int(window))
        self._stages = deque(maxlen=int(window))
        self._stats = deque(maxlen=int(window))
        self.total_frames = 0

    def add(self, total_ms, stage_ms, stats):
        self._totals.append(total_ms)
        self._stages.append(dict(stage_ms))
        self._stats.append({k: v for k, v in stats.items() if isinstance(v, (int, float))})
        self.total_frames += 1

    def total(self):
        if not self._totals:
            return 0.0, 0.0, 0.0
        values = np.asarray(self._totals)
        return float(values.mean()), float(np.percentile(values, 95)), float(values.max())

    def stage_means(self):
        merged = {}
        for stages in self._stages:
            for name, ms in stages.items():
                merged.setdefault(name, []).append(ms)
        return {name: float(np.mean(v)) for name, v in merged.items()}

    def stat_means(self):
        merged = {}
        for stats in self._stats:
            for name, value in stats.items():
                merged.setdefault(name, []).append(value)
        return {name: float(np.mean(v)) for name, v in merged.items()}

    def format_summary(self):
        mean, p95, peak = self.total()
        lines = [f'Geometric analysis: last {len(self._totals)} frames '
                 f'({self.total_frames} total)']
        for name, value in self.stat_means().items():
            lines.append(f'  {name:<32}: {value:>12,.2f}')
        stages = ', '.join(f'{k}={v:.1f}' for k, v in self.stage_means().items())
        lines.append(f'  {"stage means [ms]":<32}: {stages}')
        lines.append(f'  {"total [ms] mean/p95/max":<32}: {mean:.1f} / {p95:.1f} / {peak:.1f}')
        return '\n'.join(lines)


def build_diagnostic_array(stats, stage_ms, total_ms, rolling, stamp, name, hardware_id='',
                           budget_ms=100.0):
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

    mean, p95, peak = rolling.total()
    status = DiagnosticStatus()
    status.name = name
    status.hardware_id = hardware_id
    ground_status = stats.get('ground_status', 'ok')
    if ground_status not in ('ok', None):
        status.level = DiagnosticStatus.WARN
        status.message = f'Ground not found: {ground_status}'
    elif budget_ms > 0 and mean > budget_ms:
        status.level = DiagnosticStatus.WARN
        status.message = f'Mean processing time {mean:.1f} ms exceeds budget {budget_ms:.1f} ms'
    else:
        status.level = DiagnosticStatus.OK
        status.message = 'OK'

    def fmt(value):
        return f'{value:.4f}' if isinstance(value, float) else str(value)

    status.values = (
        [KeyValue(key=k, value=fmt(v)) for k, v in stats.items()]
        + [KeyValue(key=f'time_{k}_ms', value=f'{v:.3f}') for k, v in stage_ms.items()]
        + [KeyValue(key='time_total_ms', value=f'{total_ms:.3f}'),
           KeyValue(key='time_total_mean_ms', value=f'{mean:.3f}'),
           KeyValue(key='time_total_p95_ms', value=f'{p95:.3f}'),
           KeyValue(key='time_total_max_ms', value=f'{peak:.3f}'),
           KeyValue(key='frames_processed', value=str(rolling.total_frames))])

    array = DiagnosticArray()
    array.header.stamp = stamp
    array.status = [status]
    return array
