"""ROS-independent LiDAR preprocessing pipeline.

Stage order::

    decode ─► invalid (non-finite + zero) ─► range ─► crop box
           ─► [outlier, if apply_to == 'filtered'] ─► FILTERED OUTPUT
           ─► voxel ─► [outlier, if apply_to == 'voxelized'] ─► VOXELIZED OUTPUT

The filtered output keeps every input field (t, ring, reflectivity, ...)
at full resolution, optionally in the original organized H x W layout.
The voxelized output is x, y, z plus averaged fields and a point count.

Configuration is a tree of dataclasses. The same tree is filled from ROS
parameters by the node and from the YAML file by the offline tools, so an
experiment run offline uses exactly the settings the node would use.
"""

import copy
from dataclasses import dataclass, field, fields, is_dataclass
import time
from typing import Optional

import numpy as np

from .diagnostics import FrameStats
from .filters import (
    crop_box_mask,
    finite_mask,
    nonzero_mask,
    radius_outlier_mask,
    range_mask,
    statistical_outlier_mask,
)
from .pointcloud_utils import scalar_field_names, xyz_from_array
from .voxelization import voxel_downsample

OUTLIER_METHODS = ('none', 'statistical', 'radius')
OUTLIER_STAGES = ('filtered', 'voxelized')
OUTLIER_BACKENDS = ('scipy', 'open3d')
RESERVED_VOXEL_FIELDS = ('x', 'y', 'z', 'point_count')


@dataclass
class FilterConfig:
    remove_nonfinite: bool = True
    remove_zero: bool = True
    zero_epsilon: float = 1e-6


@dataclass
class RangeFilterConfig:
    enabled: bool = True
    min_range: float = 0.5
    max_range: float = 50.0


@dataclass
class CropBoxConfig:
    enabled: bool = False
    min_xyz: tuple = (-0.5, -0.5, -0.5)
    max_xyz: tuple = (0.5, 0.5, 0.5)
    negative: bool = True


@dataclass
class VoxelConfig:
    enabled: bool = True
    voxel_size: float = 0.05
    min_points_per_voxel: int = 1
    average_fields: tuple = ('intensity',)
    include_point_count: bool = True


@dataclass
class OutlierConfig:
    method: str = 'none'
    apply_to: str = 'voxelized'
    backend: str = 'scipy'
    nb_neighbors: int = 20
    std_ratio: float = 2.0
    radius: float = 0.2
    min_points: int = 5


@dataclass
class OutputConfig:
    keep_organized: bool = False


@dataclass
class PipelineConfig:
    filters: FilterConfig = field(default_factory=FilterConfig)
    range_filter: RangeFilterConfig = field(default_factory=RangeFilterConfig)
    crop_box: CropBoxConfig = field(default_factory=CropBoxConfig)
    voxel: VoxelConfig = field(default_factory=VoxelConfig)
    outlier: OutlierConfig = field(default_factory=OutlierConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def errors(self):
        """List of configuration errors (empty if the configuration is valid)."""
        errors = []
        if self.filters.zero_epsilon < 0:
            errors.append('filters.zero_epsilon must be >= 0')
        r = self.range_filter
        if r.min_range < 0:
            errors.append('range_filter.min_range must be >= 0')
        if r.max_range <= r.min_range:
            errors.append('range_filter.max_range must be greater than min_range')
        c = self.crop_box
        if len(c.min_xyz) != 3 or len(c.max_xyz) != 3:
            errors.append('crop_box.min_xyz and max_xyz must have 3 values')
        elif any(lo >= hi for lo, hi in zip(c.min_xyz, c.max_xyz)):
            errors.append('crop_box.min_xyz must be smaller than max_xyz on every axis')
        v = self.voxel
        if v.voxel_size <= 0:
            errors.append('voxel.voxel_size must be > 0')
        if v.min_points_per_voxel < 1:
            errors.append('voxel.min_points_per_voxel must be >= 1')
        reserved = [f for f in v.average_fields if f in RESERVED_VOXEL_FIELDS]
        if reserved:
            errors.append(f'voxel.average_fields cannot contain {reserved}')
        o = self.outlier
        if o.method not in OUTLIER_METHODS:
            errors.append(f'outlier.method must be one of {OUTLIER_METHODS}, got {o.method!r}')
        if o.apply_to not in OUTLIER_STAGES:
            errors.append(f'outlier.apply_to must be one of {OUTLIER_STAGES}, got {o.apply_to!r}')
        if o.backend not in OUTLIER_BACKENDS:
            errors.append(f'outlier.backend must be one of {OUTLIER_BACKENDS}, got {o.backend!r}')
        if o.nb_neighbors < 1:
            errors.append('outlier.nb_neighbors must be >= 1')
        if o.std_ratio <= 0:
            errors.append('outlier.std_ratio must be > 0')
        if o.radius <= 0:
            errors.append('outlier.radius must be > 0')
        if o.min_points < 1:
            errors.append('outlier.min_points must be >= 1')
        return errors

    def warnings(self):
        """Valid but probably unintended settings."""
        warnings = []
        o, v = self.outlier, self.voxel
        stage = o.apply_to if v.enabled else 'filtered'
        if o.method == 'radius' and stage == 'voxelized' and o.radius < 2.0 * v.voxel_size:
            warnings.append(
                f'outlier.radius ({o.radius} m) is less than 2 x voxel_size '
                f'({v.voxel_size} m). Voxel centroids are roughly one voxel apart, so '
                f'few neighbours fit in this radius and most points may be removed.')
        if o.method != 'none' and stage == 'filtered':
            warnings.append(
                'Outlier removal on the full-resolution cloud is expensive '
                '(hundreds of ms for a full Ouster scan). Consider apply_to: voxelized.')
        if o.method != 'none' and o.apply_to == 'voxelized' and not v.enabled:
            warnings.append('outlier.apply_to is voxelized but voxel is disabled; '
                            'outlier removal runs on the filtered cloud instead.')
        if not self.filters.remove_zero:
            warnings.append('filters.remove_zero is off: Ouster no-return points at (0, 0, 0) '
                            'are only removed if range_filter.min_range > 0.')
        return warnings

    def validate(self):
        errors = self.errors()
        if errors:
            raise ValueError('Invalid point-cloud pipeline configuration: ' + '; '.join(errors))
        return self


# ─── Configuration conversion ────────────────────────────────────────────────

def _coerce(value, reference, path):
    """Convert ``value`` to the type of ``reference`` (the current/default value)."""
    if isinstance(reference, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ('true', '1', 'yes', 'on'):
                return True
            if text in ('false', '0', 'no', 'off'):
                return False
        raise ValueError(f'{path}: expected a boolean, got {value!r}')
    if isinstance(reference, int):
        if isinstance(value, bool):
            raise ValueError(f'{path}: expected an integer, got {value!r}')
        number = float(value)
        if not number.is_integer():
            raise ValueError(f'{path}: expected an integer, got {value!r}')
        return int(number)
    if isinstance(reference, float):
        if isinstance(value, bool):
            raise ValueError(f'{path}: expected a number, got {value!r}')
        return float(value)
    if isinstance(reference, str):
        return str(value).strip().lower()
    if isinstance(reference, tuple):
        if isinstance(value, (str, bytes)) or not hasattr(value, '__iter__'):
            raise ValueError(f'{path}: expected a list, got {value!r}')
        items = list(value)
        if reference and isinstance(reference[0], float):
            if len(items) != len(reference):
                raise ValueError(f'{path}: expected {len(reference)} numbers, got {items!r}')
            return tuple(float(item) for item in items)
        # String lists: empty strings are dropped, so [""] means "empty list"
        # (ROS YAML parameter files cannot express an empty list).
        return tuple(str(item).strip() for item in items if str(item).strip())
    raise TypeError(f'{path}: unsupported configuration type {type(reference)}')


def config_from_dict(data, base=None):
    """Build a PipelineConfig from a nested dict like {'voxel': {'voxel_size': 0.1}}.

    Returns ``(config, unknown_keys)``. Values are type-coerced (e.g. an
    integer 30 is accepted for a float parameter). The config is not validated.
    """
    config = copy.deepcopy(base) if base is not None else PipelineConfig()
    unknown = []
    for section_name, section_values in (data or {}).items():
        section = getattr(config, section_name, None)
        if not is_dataclass(section):
            unknown.append(section_name)
            continue
        if not isinstance(section_values, dict):
            raise ValueError(f'{section_name}: expected a mapping, got {section_values!r}')
        for key, value in section_values.items():
            path = f'{section_name}.{key}'
            if not hasattr(section, key):
                unknown.append(path)
                continue
            setattr(section, key, _coerce(value, getattr(section, key), path))
    return config, unknown


def config_to_flat_dict(config):
    """Flatten to {'voxel.voxel_size': 0.05, ...} with ROS-friendly values (lists, not tuples)."""
    flat = {}
    for section_field in fields(config):
        section = getattr(config, section_field.name)
        for item in fields(section):
            value = getattr(section, item.name)
            flat[f'{section_field.name}.{item.name}'] = list(value) if isinstance(
                value, tuple) else value
    return flat


def flat_to_nested(flat):
    """Inverse of config_to_flat_dict's key scheme: 'a.b' -> {'a': {'b': ...}}."""
    nested = {}
    for key, value in flat.items():
        section, _, name = key.partition('.')
        if not name:
            continue
        nested.setdefault(section, {})[name] = value
    return nested


def load_config_yaml(path, node_name='pointcloud_processing'):
    """Load the pipeline configuration from a ROS 2 parameter YAML file.

    Accepts the standard ``<node>: ros__parameters: ...`` layout (also
    ``/**:``) or a plain mapping of sections. Top-level scalar node parameters
    (topics, QoS, logging) are ignored. Returns ``(config, unknown_keys)``.
    """
    import yaml

    with open(path, 'r', encoding='utf-8') as handle:
        document = yaml.safe_load(handle) or {}

    params = document
    for key in (node_name, f'/{node_name}', '/**'):
        if key in document:
            params = document[key]
            break
    params = params.get('ros__parameters', params)
    sections = {k: v for k, v in params.items() if isinstance(v, dict)}
    config, unknown = config_from_dict(sections)
    config.validate()
    return config, unknown


# ─── Pipeline ────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    filtered: np.ndarray              # structured array, all input fields
    filtered_height: int
    filtered_width: int
    filtered_is_dense: bool
    voxelized: Optional[np.ndarray]   # structured array or None if voxel disabled
    keep_mask: np.ndarray             # (N,) which input points reached the filtered output
    stats: FrameStats
    missing_fields: tuple = ()        # average_fields not present in the input


class _StageTimer:
    def __init__(self, stats):
        self._stats = stats
        self._name = None
        self._start = 0.0

    def __call__(self, name):
        self._name = name
        return self

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = (time.perf_counter() - self._start) * 1e3
        timings = self._stats.timings_ms
        timings[self._name] = timings.get(self._name, 0.0) + elapsed
        return False


def _apply(keep, mask):
    """Apply ``mask`` to ``keep`` in place; return how many kept points it removed."""
    removed = int(np.count_nonzero(keep & ~mask))
    keep &= mask
    return removed


class PointCloudPipeline:
    """Stateless per-frame processor. Create a new instance to change configuration."""

    def __init__(self, config=None):
        self.config = (config or PipelineConfig()).validate()

    def process(self, cloud, height=1, width=None):
        """Process one cloud.

        Args:
            cloud: structured array with at least x, y, z (any shape; flattened).
            height, width: organized layout of the input (defaults to 1 x N).

        """
        cfg = self.config
        start = time.perf_counter()
        stats = FrameStats()
        timer = _StageTimer(stats)

        cloud = np.asarray(cloud).reshape(-1)
        n_points = cloud.shape[0]
        if width is None:
            height, width = 1, n_points
        height, width = int(height), int(width)
        if height * width != n_points:
            raise ValueError(f'height * width = {height * width} != {n_points} points')
        stats.input_points = n_points

        with timer('extract_xyz'):
            xyz = xyz_from_array(cloud)
        keep = np.ones(n_points, dtype=bool)

        with timer('invalid'):
            if cfg.filters.remove_nonfinite:
                stats.nonfinite_removed = _apply(keep, finite_mask(xyz))
            if cfg.filters.remove_zero:
                stats.zero_removed = _apply(
                    keep, nonzero_mask(xyz, cfg.filters.zero_epsilon))

        if cfg.range_filter.enabled:
            with timer('range'):
                stats.range_removed = _apply(keep, range_mask(
                    xyz, cfg.range_filter.min_range, cfg.range_filter.max_range))

        if cfg.crop_box.enabled:
            with timer('crop_box'):
                stats.crop_removed = _apply(keep, crop_box_mask(
                    xyz, cfg.crop_box.min_xyz, cfg.crop_box.max_xyz, cfg.crop_box.negative))

        outlier_on = cfg.outlier.method != 'none'
        outlier_stage = cfg.outlier.apply_to if cfg.voxel.enabled else 'filtered'

        if outlier_on and outlier_stage == 'filtered':
            with timer('outlier'):
                indices = np.flatnonzero(keep)
                ok = self._outlier_mask(xyz[indices])
                keep[indices[~ok]] = False
                stats.outliers_removed = int(np.count_nonzero(~ok))

        stats.filtered_points = int(np.count_nonzero(keep))

        with timer('build_filtered'):
            if cfg.output.keep_organized:
                filtered = cloud.copy()
                drop = ~keep
                for axis in ('x', 'y', 'z'):
                    filtered[axis][drop] = np.nan
                f_height, f_width = height, width
                f_dense = bool(keep.all())
            else:
                filtered = cloud[keep]
                f_height, f_width = 1, filtered.shape[0]
                f_dense = (cfg.filters.remove_nonfinite
                           or bool(np.isfinite(xyz[keep]).all()))

        voxelized = None
        missing = ()
        if cfg.voxel.enabled:
            with timer('voxel'):
                selected = np.flatnonzero(keep)
                available = set(scalar_field_names(cloud))
                fields_to_average = [f for f in cfg.voxel.average_fields if f in available]
                missing = tuple(f for f in cfg.voxel.average_fields if f not in available)
                values = {name: cloud[name][selected] for name in fields_to_average}
                grid = voxel_downsample(
                    xyz[selected], cfg.voxel.voxel_size, values,
                    cfg.voxel.min_points_per_voxel)
            centroids, counts, averaged = grid.centroids, grid.counts, grid.values

            if outlier_on and outlier_stage == 'voxelized':
                with timer('outlier'):
                    ok = self._outlier_mask(centroids)
                    stats.outliers_removed = int(np.count_nonzero(~ok))
                    centroids, counts = centroids[ok], counts[ok]
                    averaged = {name: v[ok] for name, v in averaged.items()}

            with timer('build_voxelized'):
                voxelized = self._voxel_array(centroids, counts, averaged, fields_to_average)
            stats.voxel_points = int(voxelized.shape[0])

        stats.total_ms = (time.perf_counter() - start) * 1e3
        return PipelineResult(
            filtered=filtered,
            filtered_height=f_height,
            filtered_width=f_width,
            filtered_is_dense=f_dense,
            voxelized=voxelized,
            keep_mask=keep,
            stats=stats,
            missing_fields=missing,
        )

    def _outlier_mask(self, xyz):
        o = self.config.outlier
        if o.method == 'statistical':
            return statistical_outlier_mask(xyz, o.nb_neighbors, o.std_ratio, o.backend)
        if o.method == 'radius':
            return radius_outlier_mask(xyz, o.radius, o.min_points, o.backend)
        return np.ones(len(xyz), dtype=bool)

    def _voxel_array(self, centroids, counts, averaged, field_order):
        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        dtype += [(name, np.float32) for name in field_order]
        if self.config.voxel.include_point_count:
            dtype.append(('point_count', np.uint32))
        out = np.empty(centroids.shape[0], dtype=dtype)
        out['x'], out['y'], out['z'] = centroids[:, 0], centroids[:, 1], centroids[:, 2]
        for name in field_order:
            out[name] = averaged[name]
        if self.config.voxel.include_point_count:
            out['point_count'] = counts
        return out
