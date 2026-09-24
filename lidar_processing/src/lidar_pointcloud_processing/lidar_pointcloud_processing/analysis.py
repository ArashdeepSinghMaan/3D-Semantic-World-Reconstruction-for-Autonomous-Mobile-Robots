"""Timestamp, synchronization and data-quality analysis (ROS-independent).

Used by the live inspector node and by the offline ``inspect_bag`` tool to
close the Phase 0 checklist items: point fields, frame, timestamp
distribution and sensor-to-sensor offsets.
"""

import numpy as np

from .filters import finite_mask, nonzero_mask


def stamp_to_sec(stamp):
    return int(stamp.sec) + int(stamp.nanosec) * 1e-9


def stamp_statistics(stamps_s):
    """Statistics of a sequence of timestamps (seconds), in arrival order."""
    stamps = np.asarray(stamps_s, dtype=np.float64)
    result = {'count': int(stamps.size)}
    if stamps.size < 2:
        return result
    dt = np.diff(stamps)
    positive = dt[dt > 0]
    median = float(np.median(positive)) if positive.size else float('nan')
    duration = float(stamps.max() - stamps.min())
    result.update({
        'duration_s': duration,
        'rate_hz': (stamps.size - 1) / duration if duration > 0 else float('nan'),
        'dt_mean_ms': float(dt.mean() * 1e3),
        'dt_median_ms': median * 1e3,
        'dt_std_ms': float(dt.std() * 1e3),
        'dt_min_ms': float(dt.min() * 1e3),
        'dt_max_ms': float(dt.max() * 1e3),
        'jitter_p95_ms': float(np.percentile(np.abs(dt - median), 95) * 1e3)
        if positive.size else float('nan'),
        'gaps_gt_1.5x_median': int(np.count_nonzero(dt > 1.5 * median))
        if positive.size else 0,
        'non_monotonic': int(np.count_nonzero(dt <= 0)),
    })
    return result


def nearest_offsets(reference_s, other_s):
    """For each reference stamp, signed offset (other - reference) to the nearest other stamp."""
    reference = np.asarray(reference_s, dtype=np.float64)
    other = np.sort(np.asarray(other_s, dtype=np.float64))
    if reference.size == 0 or other.size == 0:
        return np.zeros(0)
    if other.size == 1:
        return other[0] - reference
    idx = np.clip(np.searchsorted(other, reference), 1, other.size - 1)
    before = other[idx - 1]
    after = other[idx]
    use_after = np.abs(after - reference) < np.abs(reference - before)
    return np.where(use_after, after, before) - reference


def offset_statistics(offsets_s):
    offsets = np.asarray(offsets_s, dtype=np.float64)
    if offsets.size == 0:
        return {'count': 0}
    magnitude = np.abs(offsets)
    return {
        'count': int(offsets.size),
        'mean_ms': float(offsets.mean() * 1e3),
        'median_ms': float(np.median(offsets) * 1e3),
        'mean_abs_ms': float(magnitude.mean() * 1e3),
        'p95_abs_ms': float(np.percentile(magnitude, 95) * 1e3),
        'max_abs_ms': float(magnitude.max() * 1e3),
    }


def cloud_quality(cloud):
    """Per-frame quality metrics of a decoded structured cloud."""
    cloud = np.asarray(cloud).reshape(-1)
    names = cloud.dtype.names or ()
    result = {'points': int(cloud.shape[0])}
    if not all(n in names for n in ('x', 'y', 'z')) or cloud.shape[0] == 0:
        return result

    xyz = np.column_stack([cloud['x'], cloud['y'], cloud['z']]).astype(np.float64)
    finite = finite_mask(xyz)
    valid = finite & nonzero_mask(xyz)
    ranges = np.linalg.norm(xyz[valid], axis=1)
    result.update({
        'nonfinite': int(np.count_nonzero(~finite)),
        'zero': int(np.count_nonzero(finite & ~valid)),
        'valid': int(np.count_nonzero(valid)),
        'valid_fraction': float(valid.mean()),
    })
    if ranges.size:
        result.update({
            'range_min_m': float(ranges.min()),
            'range_median_m': float(np.median(ranges)),
            'range_p99_m': float(np.percentile(ranges, 99)),
            'range_max_m': float(ranges.max()),
        })
    if 't' in names and cloud['t'].ndim == 1:
        t = cloud['t'].astype(np.float64)
        result['t_min'] = float(t.min())
        result['t_max'] = float(t.max())
        # Ouster drivers publish t in nanoseconds since the start of the sweep.
        result['t_span_ms_if_ns'] = float((t.max() - t.min()) * 1e-6)
    if 'ring' in names and cloud['ring'].ndim == 1:
        result['rings'] = int(np.unique(cloud['ring']).size)
    return result


def aggregate_quality(qualities):
    """Mean / min / max of each numeric metric over several frames."""
    keys = sorted({k for q in qualities for k in q})
    result = {}
    for key in keys:
        values = np.array([q[key] for q in qualities if key in q], dtype=np.float64)
        if values.size:
            result[key] = (float(values.mean()), float(values.min()), float(values.max()))
    return result


def format_table(title, mapping):
    lines = [title, '─' * max(len(title), 40)]
    for key, value in mapping.items():
        if isinstance(value, tuple) and len(value) == 3:
            text = f'mean {value[0]:,.3f}   min {value[1]:,.3f}   max {value[2]:,.3f}'
        elif isinstance(value, float):
            text = f'{value:,.3f}'
        else:
            text = f'{value}'
        lines.append(f'  {key:<24}: {text}')
    return '\n'.join(lines)
