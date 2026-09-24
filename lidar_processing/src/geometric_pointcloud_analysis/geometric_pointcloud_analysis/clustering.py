"""Euclidean clustering.

Two points belong to the same cluster if a chain of points connects them in
which consecutive points are at most ``tolerance`` apart. This is PCL's
EuclideanClusterExtraction, implemented as the connected components of the
radius-neighbour graph: a KD-tree finds all pairs within the tolerance
(C++ in scipy) and a sparse-graph routine labels the components.

Run it on non-ground points. The ground is one large connected surface
touching everything that stands on it, so clustering the whole cloud merges
the ground and every object into a single cluster.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass
class ClusteringResult:
    labels: np.ndarray        # (N,) cluster id, sorted by size (0 = largest); -1 = rejected
    sizes: np.ndarray         # (K,) points per accepted cluster
    components: int           # connected components before size filtering
    rejected_small: int       # points in components smaller than min_points
    rejected_large: int       # points in components larger than max_points
    pair_count: int           # neighbour pairs found (memory / cost indicator)


def euclidean_clusters(xyz, tolerance, min_points=1, max_points=0):
    """Label connected components of the ``tolerance`` radius graph.

    ``max_points`` = 0 means no upper limit.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n_points = xyz.shape[0]
    if n_points == 0:
        return ClusteringResult(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                                0, 0, 0, 0)

    pairs = cKDTree(xyz).query_pairs(r=float(tolerance), output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                       shape=(n_points, n_points))
    n_components, component = connected_components(graph, directed=False)
    sizes = np.bincount(component, minlength=n_components)

    too_small = sizes < min_points
    too_large = (sizes > max_points) if max_points and max_points > 0 else np.zeros_like(too_small)
    accepted = ~too_small & ~too_large

    # Relabel accepted components 0..K-1 in descending size order.
    accepted_ids = np.flatnonzero(accepted)
    accepted_ids = accepted_ids[np.argsort(-sizes[accepted_ids], kind='stable')]
    relabel = np.full(n_components, -1, dtype=np.int64)
    relabel[accepted_ids] = np.arange(len(accepted_ids))

    return ClusteringResult(
        labels=relabel[component],
        sizes=sizes[accepted_ids].astype(np.int64),
        components=int(n_components),
        rejected_small=int(sizes[too_small].sum()),
        rejected_large=int(sizes[too_large].sum()),
        pair_count=int(len(pairs)),
    )
