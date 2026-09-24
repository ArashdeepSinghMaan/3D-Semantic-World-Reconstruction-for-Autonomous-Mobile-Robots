"""Point filters that return boolean keep-masks.

Every filter takes an (N, 3) float array and returns an (N,) boolean mask
where True means "keep". Returning masks instead of filtered arrays lets the
pipeline apply the same selection to every other field (intensity, t, ring,
...) and count exactly how many points each stage removed.
"""

import numpy as np
# Imported at module level: the first import costs ~0.2 s, which must not land
# on the first processed frame.
from scipy.spatial import cKDTree


def finite_mask(xyz):
    """Keep points whose x, y and z are all finite (no NaN / Inf)."""
    return np.isfinite(xyz).all(axis=1)


def nonzero_mask(xyz, epsilon=1e-6):
    """Keep points that are not at the sensor origin.

    Ouster drivers usually encode "no return" as (0, 0, 0) rather than NaN,
    so these points pass a finiteness check and must be removed separately.
    Only meaningful on finite points; the pipeline runs finite_mask first.
    """
    with np.errstate(invalid='ignore'):
        return (np.abs(xyz) > epsilon).any(axis=1)


def range_mask(xyz, min_range, max_range):
    """Keep points with min_range <= ||p|| <= max_range (sensor-frame Euclidean range)."""
    r2 = np.einsum('ij,ij->i', xyz, xyz)
    with np.errstate(invalid='ignore'):
        return (r2 >= min_range * min_range) & (r2 <= max_range * max_range)


def crop_box_mask(xyz, box_min, box_max, negative=True):
    """Axis-aligned box filter in the cloud's own frame.

    With ``negative=True`` points inside the box are removed (self-filtering:
    removing the robot body and legs). With ``negative=False`` only points
    inside the box are kept (region-of-interest cropping).
    """
    box_min = np.asarray(box_min, dtype=np.float64).reshape(1, 3)
    box_max = np.asarray(box_max, dtype=np.float64).reshape(1, 3)
    with np.errstate(invalid='ignore'):
        inside = ((xyz >= box_min) & (xyz <= box_max)).all(axis=1)
    return ~inside if negative else inside


def statistical_outlier_mask(xyz, nb_neighbors=20, std_ratio=2.0, backend='scipy'):
    """Statistical outlier removal.

    For each point, compute the mean distance to its ``nb_neighbors`` nearest
    neighbours. Points whose mean distance exceeds
    ``global_mean + std_ratio * global_std`` are removed.

    Assumes the cloud has roughly uniform local density. On LiDAR data this
    is not true (density falls with range), so distant sparse points are
    removed preferentially. Applying it after voxelization reduces that bias.
    """
    n_points = len(xyz)
    if n_points <= nb_neighbors:
        return np.ones(n_points, dtype=bool)

    if backend == 'open3d':
        from .pointcloud_utils import to_open3d
        _, indices = to_open3d(xyz).remove_statistical_outlier(
            nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
        return _indices_to_mask(indices, n_points)

    distances, _ = cKDTree(xyz).query(xyz, k=int(nb_neighbors) + 1, workers=-1)
    mean_distance = distances[:, 1:].mean(axis=1)  # column 0 is the point itself
    threshold = mean_distance.mean() + std_ratio * mean_distance.std()
    return mean_distance <= threshold


def radius_outlier_mask(xyz, radius=0.2, min_points=5, backend='scipy'):
    """Radius outlier removal.

    Keep a point if at least ``min_points`` points (including itself) lie
    within ``radius``. Removes points without enough local support.
    ``radius`` must be chosen relative to point spacing: after voxelization
    neighbouring points are at least about one voxel apart.
    """
    n_points = len(xyz)
    if n_points == 0:
        return np.ones(0, dtype=bool)

    if backend == 'open3d':
        from .pointcloud_utils import to_open3d
        _, indices = to_open3d(xyz).remove_radius_outlier(
            nb_points=int(min_points), radius=float(radius))
        return _indices_to_mask(indices, n_points)

    counts = cKDTree(xyz).query_ball_point(
        xyz, r=float(radius), workers=-1, return_length=True)
    return np.asarray(counts) >= int(min_points)


def _indices_to_mask(indices, n_points):
    mask = np.zeros(n_points, dtype=bool)
    mask[np.asarray(indices, dtype=np.int64)] = True
    return mask
