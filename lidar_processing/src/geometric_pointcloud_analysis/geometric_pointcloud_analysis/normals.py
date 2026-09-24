"""Surface normals and curvature by local PCA.

For each point: gather its neighbours (k nearest, optionally limited to a
radius), compute their covariance, and take the eigenvector of the smallest
eigenvalue as the normal. With eigenvalues l0 <= l1 <= l2:

* curvature (surface variation) = l0 / (l0 + l1 + l2): 0 on a perfect
  plane, up to 1/3 for isotropic scatter (vegetation, noise)
* the eigenvalues also give the linearity / planarity / scattering features
  used in ``surface_analysis``

A normal's sign is ambiguous (+n and -n fit equally well). ``orientation``
resolves it: ``viewpoint`` flips normals to face the sensor (the origin of
the cloud frame), and ``reference`` flips them to agree with the "up" vector.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def symmetric_eig3(cov):
    """Closed-form eigen-decomposition of (M, 3, 3) symmetric PSD matrices.

    Returns (eigenvalues ascending (M, 3), smallest-eigenvalue unit eigenvectors (M, 3)).
    Eigenvalues use the trigonometric solution of the characteristic cubic.
    The eigenvector is the largest cross product of two rows of (A - l_min I).
    Rows where that is ill-conditioned (repeated smallest eigenvalue, e.g. a
    perfect line, or isotropic scatter) fall back to numpy's eigh. This is
    several times faster than batched np.linalg.eigh, which is dominated by
    per-matrix LAPACK overhead for 3 x 3 inputs.
    """
    a00, a11, a22 = cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2]
    a01, a02, a12 = cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 2]
    q = (a00 + a11 + a22) / 3.0
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
    p = np.sqrt(p2 / 6.0)
    safe_p = np.where(p > 0, p, 1.0)
    b00, b11, b22 = (a00 - q) / safe_p, (a11 - q) / safe_p, (a22 - q) / safe_p
    b01, b02, b12 = a01 / safe_p, a02 / safe_p, a12 / safe_p
    det_b = (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
             + b02 * (b01 * b12 - b11 * b02))
    phi = np.arccos(np.clip(det_b / 2.0, -1.0, 1.0)) / 3.0
    largest = q + 2.0 * p * np.cos(phi)
    smallest = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    middle = 3.0 * q - largest - smallest
    values = np.column_stack([smallest, middle, largest])
    values[p <= 0] = q[p <= 0, None]

    shifted = cov - smallest[:, None, None] * np.eye(3)
    r0, r1, r2 = shifted[:, 0], shifted[:, 1], shifted[:, 2]
    candidates = np.stack([np.cross(r0, r1), np.cross(r0, r2), np.cross(r1, r2)], axis=1)
    norms = np.einsum('mci,mci->mc', candidates, candidates)
    best = np.argmax(norms, axis=1)
    vectors = candidates[np.arange(len(cov)), best]
    best_norm = norms[np.arange(len(cov)), best]
    # |cross| ~ (l_mid - l_min)(l_max - l_min); require a relative eigen-gap of ~1e-4.
    spread = values[:, 2] - values[:, 0]
    good = (spread > 0) & (best_norm > 1e-8 * spread ** 4)
    vectors[good] /= np.sqrt(best_norm[good])[:, None]
    bad = ~good
    if bad.any():
        bad_values, bad_vectors = np.linalg.eigh(cov[bad])
        values[bad] = bad_values
        vectors[bad] = bad_vectors[:, :, 0]
    return values, vectors


@dataclass
class NormalsResult:
    normals: np.ndarray          # (N, 3); NaN where too few neighbours
    curvature: np.ndarray        # (N,)
    eigenvalues: np.ndarray      # (N, 3) ascending
    neighbor_count: np.ndarray   # (N,)

    @property
    def valid(self):
        return np.isfinite(self.normals[:, 0])


def estimate_normals(xyz, neighbors=30, radius=0.0, min_neighbors=5, orientation='viewpoint',
                     viewpoint=(0.0, 0.0, 0.0), reference=(0.0, 0.0, 1.0), chunk_size=20000,
                     tree=None, query_index=None):
    """Estimate normals.

    Args:
        neighbors: k nearest neighbours (including the point itself).
        radius: if > 0, ignore neighbours farther than this (hybrid search).
        min_neighbors: fewer valid neighbours -> NaN normal.
        orientation: 'viewpoint', 'reference' or 'none'.
        tree: optional prebuilt cKDTree on ``xyz``.
        query_index: estimate normals only for these points (others stay NaN).
            Neighbours are still searched among *all* points, so accuracy is
            unchanged; the cost scales with the number of queried points.

    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n_points = xyz.shape[0]
    normals = np.full((n_points, 3), np.nan)
    curvature = np.full(n_points, np.nan)
    eigenvalues = np.full((n_points, 3), np.nan)
    counts = np.zeros(n_points, dtype=np.int64)
    if n_points == 0:
        return NormalsResult(normals, curvature, eigenvalues, counts)

    k = int(min(max(neighbors, 3), n_points))
    tree = tree if tree is not None else cKDTree(xyz)
    bound = float(radius) if radius and radius > 0 else np.inf
    queries = (np.arange(n_points) if query_index is None
               else np.asarray(query_index, dtype=np.int64))
    if queries.size == 0:
        return NormalsResult(normals, curvature, eigenvalues, counts)
    _, index = tree.query(xyz[queries], k=k, distance_upper_bound=bound, workers=-1)
    index = np.asarray(index).reshape(len(queries), k)
    limited = np.isfinite(bound)

    for start in range(0, len(queries), chunk_size):
        idx = index[start:start + chunk_size]
        target = queries[start:start + chunk_size]
        if limited:
            valid = idx < n_points                  # missing neighbours are reported as n
            weight = valid.astype(np.float64)[..., None]
            neighbours = xyz[np.minimum(idx, n_points - 1)]
            count = valid.sum(axis=1)
            mean = (neighbours * weight).sum(axis=1) / count[:, None]
            centered = (neighbours - mean[:, None, :]) * weight
        else:                                       # every query has exactly k neighbours
            neighbours = xyz[idx]
            count = np.full(len(idx), k)
            centered = neighbours - neighbours.mean(axis=1, keepdims=True)
        # Batched matmul is ~5x faster than the equivalent einsum here.
        cov = np.matmul(centered.transpose(0, 2, 1), centered) / count[:, None, None]
        values, smallest_vectors = symmetric_eig3(cov)
        ok = count >= min_neighbors
        rows = target[ok]
        normals[rows] = smallest_vectors[ok]
        total = values[ok].sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            curvature[rows] = np.where(total > 0, values[ok, 0] / total, 0.0)
        eigenvalues[rows] = values[ok]
        counts[target] = count

    with np.errstate(invalid='ignore'):
        if orientation == 'viewpoint':
            towards = np.asarray(viewpoint, dtype=np.float64) - xyz
            flip = np.einsum('ij,ij->i', normals, towards) < 0
            normals[flip] *= -1.0
        elif orientation == 'reference':
            flip = normals @ np.asarray(reference, dtype=np.float64) < 0
            normals[flip] *= -1.0

    return NormalsResult(normals, curvature, eigenvalues, counts)
