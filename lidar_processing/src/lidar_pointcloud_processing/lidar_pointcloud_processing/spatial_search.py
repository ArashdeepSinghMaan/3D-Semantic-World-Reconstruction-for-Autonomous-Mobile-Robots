"""Spatial indexing with a KD-tree, plus brute-force baselines.

A KD-tree does not change the cloud; it indexes it so that nearest-neighbour
and radius queries cost roughly O(log N) each instead of O(N). Later phases
(normal estimation, ICP, semantic association) are built on these queries.

The live node deliberately does not build a KD-tree per frame: nothing
consumes it yet, so it would only add latency. Use this module from later
phases and from the offline benchmark (``offline_experiments kdtree-bench``).
"""

import time

import numpy as np
# Imported at module level: the first import costs ~0.2 s, which must not land
# on the first processed frame.
from scipy.spatial import cKDTree


class KDTreeIndex:
    """Thin wrapper around scipy's cKDTree with a stable, explicit API."""

    def __init__(self, points, leafsize=16):
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f'points must have shape (N, 3), got {self.points.shape}')
        self._tree = cKDTree(self.points, leafsize=leafsize)

    @property
    def size(self):
        return self.points.shape[0]

    def query_knn(self, queries, k):
        """Return (distances, indices), each of shape (Q, k), sorted by distance."""
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float64))
        k = int(min(k, self.size))
        distances, indices = self._tree.query(queries, k=k, workers=-1)
        return distances.reshape(-1, k), indices.reshape(-1, k)

    def query_radius(self, queries, radius, sort=False):
        """Return a list (one per query) of index arrays within ``radius``."""
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float64))
        result = self._tree.query_ball_point(queries, r=float(radius), workers=-1)
        out = []
        for indices in result:
            indices = np.asarray(indices, dtype=np.int64)
            if sort and indices.size:
                query_index = len(out)
                d = np.linalg.norm(self.points[indices] - queries[query_index], axis=1)
                indices = indices[np.argsort(d, kind='stable')]
            out.append(indices)
        return out

    def count_radius(self, queries, radius):
        """Number of points within ``radius`` of each query (including coincident points)."""
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float64))
        return np.asarray(self._tree.query_ball_point(
            queries, r=float(radius), workers=-1, return_length=True))


def brute_force_knn(points, queries, k, chunk_size=256):
    """Exact k-NN by computing every distance. O(Q * N) reference implementation."""
    points = np.asarray(points, dtype=np.float64)
    queries = np.atleast_2d(np.asarray(queries, dtype=np.float64))
    k = int(min(k, points.shape[0]))
    all_d = np.empty((queries.shape[0], k))
    all_i = np.empty((queries.shape[0], k), dtype=np.int64)
    points_sq = np.einsum('ij,ij->i', points, points)
    for start in range(0, queries.shape[0], chunk_size):
        q = queries[start:start + chunk_size]
        d2 = (np.einsum('ij,ij->i', q, q)[:, None] + points_sq[None, :]
              - 2.0 * q @ points.T)
        np.maximum(d2, 0.0, out=d2)
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        part_d2 = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(part_d2, axis=1)
        all_i[start:start + len(q)] = np.take_along_axis(part, order, axis=1)
        all_d[start:start + len(q)] = np.sqrt(np.take_along_axis(part_d2, order, axis=1))
    return all_d, all_i


def brute_force_radius_count(points, queries, radius, chunk_size=256):
    """Exact radius-neighbour counts by computing every distance."""
    points = np.asarray(points, dtype=np.float64)
    queries = np.atleast_2d(np.asarray(queries, dtype=np.float64))
    counts = np.empty(queries.shape[0], dtype=np.int64)
    points_sq = np.einsum('ij,ij->i', points, points)
    r2 = float(radius) ** 2
    for start in range(0, queries.shape[0], chunk_size):
        q = queries[start:start + chunk_size]
        d2 = (np.einsum('ij,ij->i', q, q)[:, None] + points_sq[None, :]
              - 2.0 * q @ points.T)
        counts[start:start + len(q)] = (d2 <= r2 + 1e-12).sum(axis=1)
    return counts


def benchmark_knn(points, n_queries=1000, k=10, radius=0.5, seed=0):
    """Compare KD-tree and brute-force queries on the same data.

    Returns a dict with build time, query times, speedups and whether both
    methods agree (k-NN distances equal within tolerance).
    """
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=np.float64)
    queries = points[rng.choice(points.shape[0], size=min(n_queries, points.shape[0]),
                                replace=False)]

    t0 = time.perf_counter()
    index = KDTreeIndex(points)
    t1 = time.perf_counter()
    tree_d, _ = index.query_knn(queries, k)
    t2 = time.perf_counter()
    tree_counts = index.count_radius(queries, radius)
    t3 = time.perf_counter()
    brute_d, _ = brute_force_knn(points, queries, k)
    t4 = time.perf_counter()
    brute_counts = brute_force_radius_count(points, queries, radius)
    t5 = time.perf_counter()

    return {
        'n_points': int(points.shape[0]),
        'n_queries': int(queries.shape[0]),
        'k': int(k),
        'radius': float(radius),
        'build_ms': (t1 - t0) * 1e3,
        'kdtree_knn_ms': (t2 - t1) * 1e3,
        'kdtree_radius_ms': (t3 - t2) * 1e3,
        'brute_knn_ms': (t4 - t3) * 1e3,
        'brute_radius_ms': (t5 - t4) * 1e3,
        'knn_speedup_incl_build': (t4 - t3) / max((t2 - t0), 1e-12),
        'knn_agree': bool(np.allclose(tree_d, brute_d, atol=1e-6)),
        # Counts can differ by float round-off exactly at the radius boundary.
        'radius_count_max_diff': int(np.abs(tree_counts - brute_counts).max()),
    }
