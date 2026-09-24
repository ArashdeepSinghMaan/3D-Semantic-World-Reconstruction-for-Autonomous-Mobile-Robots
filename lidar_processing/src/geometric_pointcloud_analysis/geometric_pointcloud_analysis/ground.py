"""Ground segmentation.

Two methods:

``ransac``
    One global plane (constrained RANSAC). Fast and robust on flat ground,
    but a single plane cannot follow slopes, bumps or undulating grass: far
    ground rises above or drops below the plane and is misclassified.

``grid``
    The global plane defines "up" and a 2D grid on the ground. Each grid cell
    gets its own constrained RANSAC plane. A cell's plane is accepted only
    if its tilt relative to the global plane is below ``max_tilt_deg`` and
    it stays within ``max_height_offset`` of the global plane at the cell
    centre, which rejects cells where the "plane" is a car roof or a bush top.
    Rejected or sparse cells fall back to the global plane. This is a simple
    piecewise-planar model in the spirit of region-wise methods such as
    Patchwork.

Every point also gets a signed height above its local ground model, which
later phases use for obstacle height and terrain reasoning.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .ransac import fit_plane_ransac, normalize, PlaneModel


@dataclass
class GroundCell:
    index: tuple            # (i, j) grid index
    center: np.ndarray      # (3,) cell centre projected onto its plane
    normal: np.ndarray
    d: float
    point_count: int
    inlier_count: int
    accepted: bool
    reason: str = ''


@dataclass
class GroundResult:
    mask: np.ndarray                  # (N,) True = ground
    height: np.ndarray                # (N,) signed height above local ground model (NaN = unknown)
    plane: Optional[PlaneModel]       # global plane (None if no valid ground plane)
    method: str
    up: np.ndarray                    # "up" direction used (cloud frame)
    cells: list = field(default_factory=list)
    cell_size: float = 0.0
    status: str = 'ok'

    @property
    def ground_count(self):
        return int(np.count_nonzero(self.mask))


def plane_basis(normal):
    """Two unit vectors spanning the plane orthogonal to ``normal``."""
    normal = normalize(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def segment_ground(xyz, config, reference_normal, rng=None):
    """Segment ground with ``config`` (a GeometryConfig). Returns a GroundResult."""
    g, r = config.ground, config.ransac
    xyz = np.asarray(xyz, dtype=np.float64)
    n_points = xyz.shape[0]
    up = normalize(reference_normal)
    rng = np.random.default_rng(rng)

    candidates = np.ones(n_points, dtype=bool)
    if g.candidate_max_height_enabled:
        # Height along "up" relative to the sensor origin (0 in the cloud frame).
        candidates = xyz @ up <= g.candidate_max_height

    empty = GroundResult(mask=np.zeros(n_points, dtype=bool),
                         height=np.full(n_points, np.nan), plane=None,
                         method=g.method, up=up)
    if np.count_nonzero(candidates) < max(3, g.min_inliers):
        empty.status = 'too_few_candidates'
        return empty

    candidate_index = np.flatnonzero(candidates)
    plane = fit_plane_ransac(
        xyz[candidate_index], r.distance_threshold, r.max_iterations, r.probability,
        reference_normal=up, max_angle_deg=g.max_angle_deg, refine=r.refine,
        score_sample_size=r.score_sample_size, min_inliers=g.min_inliers, rng=rng)
    if plane is None:
        empty.status = 'no_plane_within_angle'
        return empty

    # Re-express the plane over all points (not only candidates).
    height = xyz @ plane.normal + plane.d
    full_inliers = np.zeros(n_points, dtype=bool)
    full_inliers[candidate_index[plane.inliers]] = True
    plane = PlaneModel(normal=plane.normal, d=plane.d, inliers=full_inliers,
                       iterations=plane.iterations, rms=plane.rms)
    mask = np.abs(height) <= r.distance_threshold

    result = GroundResult(mask=mask, height=height, plane=plane, method=g.method,
                          up=plane.normal)
    if g.method == 'grid':
        _refine_with_grid(xyz, candidates, result, config, rng)
    return result


def _refine_with_grid(xyz, candidates, result, config, rng):
    grid = config.grid
    plane = result.plane
    u, v = plane_basis(plane.normal)
    cells_ij = np.floor(np.column_stack([xyz @ u, xyz @ v]) / grid.cell_size).astype(np.int64)
    result.cell_size = grid.cell_size

    keys = cells_ij[:, 0] * 1_000_003 + cells_ij[:, 1]
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    ends = np.r_[starts[1:], len(order)]
    cos_tilt = np.cos(np.radians(grid.max_tilt_deg))

    for start, end in zip(starts, ends):
        members = order[start:end]
        i, j = (int(c) for c in cells_ij[members[0]])
        centre_2d = (np.array([i, j]) + 0.5) * grid.cell_size
        ground_candidates = members[candidates[members]]
        cell = GroundCell(index=(i, j), center=np.zeros(3), normal=plane.normal, d=plane.d,
                          point_count=int(len(members)), inlier_count=0, accepted=False)

        if len(ground_candidates) < grid.min_points:
            cell.reason = 'too_few_points'
        else:
            local = fit_plane_ransac(
                xyz[ground_candidates], grid.distance_threshold, grid.max_iterations,
                config.ransac.probability, reference_normal=plane.normal,
                max_angle_deg=grid.max_tilt_deg, refine=True,
                min_inliers=grid.min_points // 2, rng=rng)
            if local is None:
                cell.reason = 'no_plane_within_tilt'
            else:
                # Height of the local plane above the global plane at the cell centre.
                on_global = centre_2d[0] * u + centre_2d[1] * v - plane.d * plane.normal
                # Distance along global "up" from the global plane to the local plane.
                offset = -(local.normal @ on_global + local.d) / (local.normal @ plane.normal)
                if abs(offset) > grid.max_height_offset:
                    cell.reason = f'height_offset_{offset:+.2f}m'
                elif local.normal @ plane.normal < cos_tilt:
                    cell.reason = 'tilt'
                else:
                    cell.accepted = True
                    cell.normal, cell.d = local.normal, local.d
                    cell.inlier_count = local.inlier_count
                    local_height = xyz[members] @ local.normal + local.d
                    result.height[members] = local_height
                    result.mask[members] = np.abs(local_height) <= grid.distance_threshold
                cell.center = on_global + offset * plane.normal
        if not cell.accepted:
            cell.center = centre_2d[0] * u + centre_2d[1] * v - plane.d * plane.normal
        result.cells.append(cell)
