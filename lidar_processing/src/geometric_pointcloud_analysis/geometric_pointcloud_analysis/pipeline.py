"""ROS-independent geometric analysis pipeline.

    xyz ─► ground (ransac | grid) ─► ground mask, plane(s), height above ground
        ─► clustering (non-ground) ─► labels
        ─► per cluster: centroid, extent, bounding box, shape features, height
        ─► normals + curvature (all points)

Configuration is a two-level tree of dataclasses (section.parameter). It is
converted to and from ROS parameters with the Phase 1 helpers, so the same
YAML drives the node and the offline tool.
"""

import copy
from dataclasses import dataclass, field
import time
from typing import Optional

from lidar_pointcloud_processing.pipeline import (
    config_from_dict as _config_from_dict,
    config_to_flat_dict,
    flat_to_nested,
)
import numpy as np

from .bounding_boxes import BoundingBox, compute_box
from .clustering import euclidean_clusters
from .ground import GroundResult, segment_ground
from .normals import estimate_normals, NormalsResult
from .ransac import normalize
from .surface_analysis import (
    dominant_shape,
    eigen_features,
    ground_statistics,
    local_features,
    mean_local_features,
)

GROUND_METHODS = ('ransac', 'grid')
BOX_TYPES = ('upright', 'pca', 'aabb')
NORMAL_ORIENTATIONS = ('viewpoint', 'reference', 'none')
CLUSTER_INPUTS = ('non_ground', 'all')
NORMAL_TARGETS = ('all', 'non_ground')


@dataclass
class GroundConfig:
    enabled: bool = True
    method: str = 'ransac'
    reference_normal: tuple = (0.0, 0.0, 1.0)   # "up" in the cloud frame (or reference_frame)
    reference_frame: str = ''                   # node only: TF frame the reference is expressed in
    max_angle_deg: float = 20.0
    min_inliers: int = 100
    candidate_max_height_enabled: bool = False
    candidate_max_height: float = 0.5           # along "up", relative to the sensor origin


@dataclass
class RansacConfig:
    distance_threshold: float = 0.08
    max_iterations: int = 1000
    probability: float = 0.999
    refine: bool = True
    score_sample_size: int = 20000              # 0 = score hypotheses on all points
    seed: int = -1                              # -1 = random each frame


@dataclass
class GridConfig:
    cell_size: float = 5.0
    min_points: int = 50
    distance_threshold: float = 0.08
    max_iterations: int = 100
    max_tilt_deg: float = 25.0
    max_height_offset: float = 0.4


@dataclass
class ClusteringConfig:
    enabled: bool = True
    input: str = 'non_ground'
    tolerance: float = 0.25
    min_points: int = 20
    max_points: int = 50000                     # 0 = unlimited
    min_height_above_ground: float = -1.0e9     # drop clusters whose top is below this


@dataclass
class NormalsConfig:
    enabled: bool = True
    neighbors: int = 30
    radius: float = 0.0                         # > 0: hybrid kNN + radius search
    min_neighbors: int = 5
    orientation: str = 'viewpoint'
    viewpoint: tuple = (0.0, 0.0, 0.0)
    target: str = 'non_ground'                  # all | non_ground
    max_points: int = 0                         # > 0: random subset of queried points


@dataclass
class BoxesConfig:
    enabled: bool = True
    type: str = 'upright'


@dataclass
class FeaturesConfig:
    enabled: bool = True


@dataclass
class GeometryConfig:
    ground: GroundConfig = field(default_factory=GroundConfig)
    ransac: RansacConfig = field(default_factory=RansacConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    normals: NormalsConfig = field(default_factory=NormalsConfig)
    boxes: BoxesConfig = field(default_factory=BoxesConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)

    def errors(self):
        e = []
        g, r, gr, c, n = self.ground, self.ransac, self.grid, self.clustering, self.normals
        if g.method not in GROUND_METHODS:
            e.append(f'ground.method must be one of {GROUND_METHODS}')
        if len(g.reference_normal) != 3 or np.linalg.norm(g.reference_normal) < 1e-9:
            e.append('ground.reference_normal must be a non-zero 3-vector')
        if not 0 < g.max_angle_deg <= 90:
            e.append('ground.max_angle_deg must be in (0, 90]')
        if g.min_inliers < 3:
            e.append('ground.min_inliers must be >= 3')
        if r.distance_threshold <= 0 or gr.distance_threshold <= 0:
            e.append('ransac/grid distance_threshold must be > 0')
        if r.max_iterations < 1 or gr.max_iterations < 1:
            e.append('ransac/grid max_iterations must be >= 1')
        if not 0 < r.probability < 1:
            e.append('ransac.probability must be in (0, 1)')
        if r.score_sample_size < 0:
            e.append('ransac.score_sample_size must be >= 0')
        if gr.cell_size <= 0 or gr.min_points < 3:
            e.append('grid.cell_size must be > 0 and grid.min_points >= 3')
        if not 0 < gr.max_tilt_deg <= 90 or gr.max_height_offset < 0:
            e.append('grid.max_tilt_deg must be in (0, 90] and max_height_offset >= 0')
        if c.input not in CLUSTER_INPUTS:
            e.append(f'clustering.input must be one of {CLUSTER_INPUTS}')
        if c.tolerance <= 0 or c.min_points < 1 or c.max_points < 0:
            e.append('clustering: tolerance > 0, min_points >= 1, max_points >= 0')
        if c.max_points and c.max_points < c.min_points:
            e.append('clustering.max_points must be 0 or >= min_points')
        if n.neighbors < 3 or n.min_neighbors < 3 or n.radius < 0:
            e.append('normals: neighbors >= 3, min_neighbors >= 3, radius >= 0')
        if n.orientation not in NORMAL_ORIENTATIONS:
            e.append(f'normals.orientation must be one of {NORMAL_ORIENTATIONS}')
        if n.target not in NORMAL_TARGETS:
            e.append(f'normals.target must be one of {NORMAL_TARGETS}')
        if n.max_points < 0:
            e.append('normals.max_points must be >= 0')
        if len(n.viewpoint) != 3:
            e.append('normals.viewpoint must have 3 values')
        if self.boxes.type not in BOX_TYPES:
            e.append(f'boxes.type must be one of {BOX_TYPES}')
        return e

    def warnings(self, input_voxel_size=None):
        w = []
        c = self.clustering
        if c.input == 'all' and self.ground.enabled:
            w.append('clustering.input is "all": the ground will connect most objects into '
                     'one cluster.')
        if not self.ground.enabled and c.input == 'non_ground':
            w.append('ground is disabled, so clustering runs on all points.')
        if input_voxel_size and c.tolerance < 1.5 * input_voxel_size:
            w.append(f'clustering.tolerance ({c.tolerance} m) is below 1.5 x the input voxel '
                     f'size ({input_voxel_size} m): neighbouring points of one surface may '
                     f'not connect.')
        if self.ransac.distance_threshold > 0.3:
            w.append('ransac.distance_threshold > 0.3 m absorbs low obstacles into the ground.')
        return w

    def validate(self):
        errors = self.errors()
        if errors:
            raise ValueError('Invalid geometry configuration: ' + '; '.join(errors))
        return self


def config_from_dict(data, base=None):
    """Nested dict -> GeometryConfig (type-coerced, not validated); returns (config, unknown)."""
    config, unknown = _config_from_dict(
        data, base=copy.deepcopy(base) if base else GeometryConfig())
    # String values are lower-cased by the shared coercion (right for enum
    # values), but TF frame ids are case-sensitive: keep them verbatim.
    frame = (data or {}).get('ground', {}).get('reference_frame')
    if frame is not None:
        config.ground.reference_frame = str(frame).strip()
    return config, unknown


def load_config_yaml(path, node_name='geometric_analysis'):
    import yaml
    with open(path, 'r', encoding='utf-8') as handle:
        document = yaml.safe_load(handle) or {}
    params = document
    for key in (node_name, f'/{node_name}', '/**'):
        if key in document:
            params = document[key]
            break
    params = params.get('ros__parameters', params)
    config, unknown = config_from_dict({k: v for k, v in params.items() if isinstance(v, dict)})
    return config.validate(), unknown


__all__ = ['GeometryConfig', 'GeometryPipeline', 'config_from_dict', 'config_to_flat_dict',
           'flat_to_nested', 'load_config_yaml']


# ─── Results ────────────────────────────────────────────────────────────────

@dataclass
class ClusterInfo:
    id: int
    point_count: int
    centroid: np.ndarray
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    box: Optional[BoundingBox]
    features: dict
    shape: str
    height_min: float          # height above local ground (NaN if unknown)
    height_max: float
    mean_curvature: float

    @property
    def dimensions(self):
        return self.max_xyz - self.min_xyz

    def summary(self):
        dims = self.box.extent if self.box is not None else self.dimensions
        return (f'cluster {self.id}: {self.point_count} pts, centroid '
                f'({self.centroid[0]:.2f}, {self.centroid[1]:.2f}, {self.centroid[2]:.2f}), '
                f'size {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} m, {self.shape}, '
                f'height {self.height_max:.2f} m')


@dataclass
class GeometryResult:
    ground: Optional[GroundResult]
    labels: np.ndarray                 # (N,) cluster id per point, -1 = none
    clusters: list
    normals: Optional[NormalsResult]
    up: np.ndarray
    stats: dict                        # counts and scalar metrics
    timings_ms: dict
    total_ms: float = 0.0

    @property
    def ground_mask(self):
        if self.ground is None:
            return np.zeros(len(self.labels), dtype=bool)
        return self.ground.mask


class _Timer:
    def __init__(self, timings):
        self.timings = timings

    def __call__(self, name):
        self.name = name
        return self

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *exc):
        self.timings[self.name] = self.timings.get(self.name, 0.0) + (
            time.perf_counter() - self.start) * 1e3
        return False


class GeometryPipeline:

    def __init__(self, config=None):
        self.config = (config or GeometryConfig()).validate()

    def process(self, xyz, reference_normal=None):
        """Analyse an (N, 3) cloud of finite points (in the sensor frame).

        ``reference_normal`` overrides ``ground.reference_normal`` (the node
        passes the TF-derived "up" here).
        """
        cfg = self.config
        start = time.perf_counter()
        timings = {}
        timer = _Timer(timings)
        xyz = np.asarray(xyz, dtype=np.float64)
        n_points = xyz.shape[0]
        up = normalize(reference_normal if reference_normal is not None
                       else cfg.ground.reference_normal)
        rng = None if cfg.ransac.seed < 0 else cfg.ransac.seed
        stats = {'input_points': n_points}

        ground = None
        if cfg.ground.enabled:
            with timer('ground'):
                ground = segment_ground(xyz, cfg, up, rng=rng)
            if ground.plane is not None:
                up = ground.plane.normal
            stats['ground_points'] = ground.ground_count
            stats['ground_status'] = ground.status
            reference_up = (reference_normal if reference_normal is not None
                            else cfg.ground.reference_normal)
            stats.update(ground_statistics(ground, reference_up=reference_up))
        ground_mask = ground.mask if ground is not None else np.zeros(n_points, dtype=bool)
        stats['non_ground_points'] = int(n_points - np.count_nonzero(ground_mask))

        normals = None
        if cfg.normals.enabled:
            with timer('normals'):
                n = cfg.normals
                if n.target == 'non_ground' and ground is not None:
                    query = np.flatnonzero(~ground_mask)
                else:
                    query = np.arange(n_points)
                if n.max_points and len(query) > n.max_points:
                    query = np.sort(np.random.default_rng(0).choice(
                        query, size=n.max_points, replace=False))
                normals = estimate_normals(
                    xyz, n.neighbors, n.radius, n.min_neighbors, n.orientation,
                    n.viewpoint, up, query_index=query)

        labels = np.full(n_points, -1, dtype=np.int64)
        clusters = []
        if cfg.clustering.enabled:
            with timer('clustering'):
                use_all = cfg.clustering.input == 'all' or ground is None
                subset = np.arange(n_points) if use_all else np.flatnonzero(~ground_mask)
                c = cfg.clustering
                result = euclidean_clusters(xyz[subset], c.tolerance, c.min_points, c.max_points)
                labels[subset] = result.labels
            stats.update({'clusters_raw': len(result.sizes),
                          'cluster_pairs': result.pair_count,
                          'cluster_rejected_small_points': result.rejected_small,
                          'cluster_rejected_large_points': result.rejected_large})
            with timer('cluster_analysis'):
                clusters, labels = self._describe_clusters(xyz, labels, ground, normals, up)
            stats['clusters'] = len(clusters)
            stats['clustered_points'] = int(np.count_nonzero(labels >= 0))

        if normals is not None:
            stats['normals_valid'] = int(np.count_nonzero(normals.valid))
            if normals.valid.any():
                stats['mean_curvature'] = float(np.nanmean(normals.curvature))

        return GeometryResult(ground=ground, labels=labels, clusters=clusters, normals=normals,
                              up=up, stats=stats, timings_ms=timings,
                              total_ms=(time.perf_counter() - start) * 1e3)

    def _describe_clusters(self, xyz, labels, ground, normals, up):
        cfg = self.config
        if labels.max(initial=-1) < 0:
            return [], labels
        per_point = None
        if cfg.features.enabled and normals is not None:
            per_point = local_features(normals.eigenvalues, normals.normals, up)
        members_sorted = np.argsort(labels, kind='stable')
        sorted_labels = labels[members_sorted]
        first = np.searchsorted(sorted_labels, 0)
        members_sorted, sorted_labels = members_sorted[first:], sorted_labels[first:]
        starts = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
        ends = np.r_[starts[1:], len(sorted_labels)]

        clusters = []
        new_labels = np.full_like(labels, -1)
        for start, end in zip(starts, ends):
            idx = members_sorted[start:end]
            points = xyz[idx]
            if ground is not None and ground.plane is not None:
                heights = ground.height[idx]
                h_min, h_max = float(np.nanmin(heights)), float(np.nanmax(heights))
            else:
                h_min = h_max = float('nan')
            if np.isfinite(h_max) and h_max < cfg.clustering.min_height_above_ground:
                continue
            if not cfg.features.enabled:
                features = {}
            elif per_point is not None:
                features = mean_local_features(per_point, idx)
            else:
                features = eigen_features(points, up)
            curvature = (float(np.nanmean(normals.curvature[idx]))
                         if normals is not None and np.isfinite(normals.curvature[idx]).any()
                         else float('nan'))
            cluster_id = len(clusters)
            new_labels[idx] = cluster_id
            clusters.append(ClusterInfo(
                id=cluster_id,
                point_count=int(len(idx)),
                centroid=points.mean(axis=0),
                min_xyz=points.min(axis=0),
                max_xyz=points.max(axis=0),
                box=compute_box(points, cfg.boxes.type, up) if cfg.boxes.enabled else None,
                features=features,
                shape=dominant_shape(features) if features else 'unknown',
                height_min=h_min,
                height_max=h_max,
                mean_curvature=curvature,
            ))
        return clusters, new_labels
