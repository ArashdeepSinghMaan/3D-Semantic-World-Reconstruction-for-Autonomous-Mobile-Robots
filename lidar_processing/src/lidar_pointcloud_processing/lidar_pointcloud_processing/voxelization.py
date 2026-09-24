"""Voxel-grid downsampling.

Space is divided into cubes of edge ``voxel_size``. All points falling into
the same cube are replaced by their centroid. Other per-point values (e.g.
intensity) can be averaged per voxel as well.

What is lost: per-point timing (``t``), beam index (``ring``), the organized
range-image structure, and any detail smaller than the voxel. That is why the
node publishes a full-resolution filtered cloud alongside the voxelized one.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VoxelGridResult:
    """Output of :func:`voxel_downsample`."""

    centroids: np.ndarray            # (M, 3) float64 centroid per voxel
    counts: np.ndarray               # (M,) number of input points per voxel
    voxel_indices: np.ndarray        # (M, 3) int64 integer voxel coordinates
    point_to_voxel: np.ndarray       # (N,) row in centroids per input point, -1 if dropped
    values: dict = field(default_factory=dict)  # name -> (M,) per-voxel mean


def voxel_downsample(xyz, voxel_size, values=None, min_points_per_voxel=1):
    """Downsample an (N, 3) cloud to one centroid per occupied voxel.

    Args:
        xyz: (N, 3) point coordinates.
        voxel_size: voxel edge length in the same unit as xyz (metres).
        values: optional mapping name -> (N,) array averaged per voxel.
        min_points_per_voxel: voxels with fewer points are discarded; values
            above 1 act as a simple density-based noise filter.

    """
    if voxel_size <= 0:
        raise ValueError(f'voxel_size must be positive, got {voxel_size}')
    xyz = np.asarray(xyz, dtype=np.float64)
    values = values or {}
    n_points = xyz.shape[0]

    if n_points == 0:
        return VoxelGridResult(
            centroids=np.zeros((0, 3)),
            counts=np.zeros(0, dtype=np.int64),
            voxel_indices=np.zeros((0, 3), dtype=np.int64),
            point_to_voxel=np.zeros(0, dtype=np.int64),
            values={name: np.zeros(0) for name in values},
        )

    # Work on contiguous 1-D columns: (N, 3) reductions along axis 0 are strided
    # and several times slower.
    columns = [np.floor(xyz[:, axis] / voxel_size).astype(np.int64) for axis in range(3)]
    lows = [int(col.min()) for col in columns]
    dims = [int(col.max()) - low + 1 for col, low in zip(columns, lows)]

    # Encode each voxel as one int64 key, then group equal keys with a single
    # argsort. This is several times faster than np.unique with
    # return_index/return_inverse/return_counts (which uses a stable sort).
    if dims[0] * dims[1] * dims[2] < 2 ** 62:
        keys = columns[0] - lows[0]
        keys *= dims[1]
        keys += columns[1] - lows[1]
        keys *= dims[2]
        keys += columns[2] - lows[2]
    else:  # extremely large extent relative to voxel size: dense re-labelling
        keys = np.unique(np.column_stack(columns), axis=0, return_inverse=True)[1].reshape(-1)

    order = np.argsort(keys)
    sorted_keys = keys[order]
    is_start = np.empty(n_points, dtype=bool)
    is_start[0] = True
    np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=is_start[1:])
    starts = np.flatnonzero(is_start)
    counts = np.diff(np.append(starts, n_points))
    n_voxels = starts.size

    inverse = np.empty(n_points, dtype=np.int64)
    inverse[order] = np.cumsum(is_start) - 1
    first_index = order[starts]

    centroids = np.empty((n_voxels, 3), dtype=np.float64)
    for axis in range(3):
        centroids[:, axis] = np.add.reduceat(xyz[:, axis][order], starts) / counts
    averaged = {
        name: np.add.reduceat(np.asarray(v, dtype=np.float64)[order], starts) / counts
        for name, v in values.items()
    }
    ijk = np.column_stack([col[first_index] for col in columns])
    voxel_indices = ijk
    point_to_voxel = inverse

    if min_points_per_voxel > 1:
        keep = counts >= min_points_per_voxel
        remap = np.full(n_voxels, -1, dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        point_to_voxel = remap[inverse]
        centroids = centroids[keep]
        counts = counts[keep]
        voxel_indices = voxel_indices[keep]
        averaged = {name: v[keep] for name, v in averaged.items()}

    return VoxelGridResult(
        centroids=centroids,
        counts=counts.astype(np.int64),
        voxel_indices=voxel_indices,
        point_to_voxel=point_to_voxel.astype(np.int64),
        values=averaged,
    )
