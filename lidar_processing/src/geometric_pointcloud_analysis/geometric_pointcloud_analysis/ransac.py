"""RANSAC plane fitting with an optional orientation constraint.

A plane is ``n . p + d = 0`` with unit normal ``n``. RANSAC repeatedly fits
a plane through three random points, counts points within
``distance_threshold`` of it, and keeps the best-supported plane.

RANSAC finds *a* plane, not *the ground*. Passing ``reference_normal`` and
``max_angle_deg`` rejects hypotheses whose normal is too far from the
expected "up" direction *during* sampling. Checking only the winning plane
afterwards is weaker: if a wall has more support than the ground, the wall
wins and the ground is never considered.

Hypotheses are generated and scored in vectorized batches, and the number
of iterations adapts to the observed inlier ratio::

    iterations = log(1 - probability) / log(1 - w^3)
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class PlaneModel:
    normal: np.ndarray      # (3,) unit normal; oriented along the reference if one was given
    d: float                # plane offset: normal . p + d = 0
    inliers: np.ndarray     # (N,) bool mask over the input points
    iterations: int         # hypotheses evaluated
    rms: float              # RMS point-to-plane distance of the inliers

    @property
    def inlier_count(self):
        return int(np.count_nonzero(self.inliers))

    def signed_distance(self, xyz):
        return np.asarray(xyz, dtype=np.float64) @ self.normal + self.d

    def angle_to_deg(self, reference):
        ref = normalize(reference)
        return float(np.degrees(np.arccos(np.clip(abs(self.normal @ ref), -1.0, 1.0))))

    def coefficients(self):
        """(a, b, c, d) of ax + by + cz + d = 0."""
        return (*map(float, self.normal), float(self.d))


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    length = np.linalg.norm(vector)
    if length < 1e-12:
        raise ValueError('reference vector must be non-zero')
    return vector / length


def fit_plane_least_squares(points):
    """Total-least-squares plane through ``points``: normal = smallest-eigenvalue direction."""
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, vectors = np.linalg.eigh(centered.T @ centered)
    normal = vectors[:, 0]
    return normal, float(-normal @ centroid)


def required_iterations(inlier_ratio, probability, sample_size=3):
    """Iterations needed to draw one all-inlier sample with the given probability."""
    if inlier_ratio <= 0.0:
        return math.inf
    if inlier_ratio >= 1.0:
        return 1
    denominator = math.log(1.0 - inlier_ratio ** sample_size)
    if denominator == 0.0:
        return math.inf
    return int(math.ceil(math.log(1.0 - probability) / denominator))


def fit_plane_ransac(xyz, distance_threshold=0.08, max_iterations=1000, probability=0.999,
                     reference_normal=None, max_angle_deg=None, refine=True,
                     score_sample_size=0, batch_size=64, min_inliers=3, rng=None):
    """Fit the best-supported plane. Returns a PlaneModel, or None if none qualifies.

    Args:
        xyz: (N, 3) points.
        distance_threshold: inlier distance to the plane (m).
        max_iterations: hard upper bound on hypotheses.
        probability: target probability of drawing one all-inlier sample
            (drives early termination).
        reference_normal: expected plane normal (e.g. gravity "up" in the
            cloud frame). The result is oriented to agree with it.
        max_angle_deg: reject hypotheses whose normal deviates more than
            this from reference_normal.
        refine: least-squares re-fit on the inliers of the best hypothesis.
        score_sample_size: score hypotheses on a random subset of this size
            (0 = all points). Final inliers always use all points.
        min_inliers: minimum inliers for a valid result.
        rng: numpy Generator or seed.

    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n_points = xyz.shape[0]
    if n_points < 3:
        return None
    rng = np.random.default_rng(rng)
    reference = normalize(reference_normal) if reference_normal is not None else None
    cos_limit = (math.cos(math.radians(max_angle_deg))
                 if reference is not None and max_angle_deg is not None else None)

    if score_sample_size and n_points > score_sample_size:
        score_points = xyz[rng.choice(n_points, size=score_sample_size, replace=False)]
    else:
        score_points = xyz

    best_count, best_normal, best_d = 0, None, 0.0
    needed, iterations = max_iterations, 0
    while iterations < min(needed, max_iterations):
        batch = min(batch_size, max_iterations - iterations)
        iterations += batch
        sample = rng.integers(0, n_points, size=(batch, 3))
        p0, p1, p2 = xyz[sample[:, 0]], xyz[sample[:, 1]], xyz[sample[:, 2]]
        normals = np.cross(p1 - p0, p2 - p0)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-9  # skip degenerate (collinear / repeated) samples
        if not valid.any():
            continue
        normals = normals[valid] / lengths[valid, None]
        p0 = p0[valid]
        if reference is not None:
            dots = normals @ reference
            normals[dots < 0] *= -1.0
            if cos_limit is not None:
                keep = np.abs(dots) >= cos_limit
                if not keep.any():
                    continue
                normals, p0 = normals[keep], p0[keep]
        offsets = -np.einsum('ij,ij->i', normals, p0)
        counts = (np.abs(score_points @ normals.T + offsets) <= distance_threshold).sum(axis=0)
        best = int(np.argmax(counts))
        if counts[best] > best_count:
            best_count, best_normal, best_d = int(counts[best]), normals[best], offsets[best]
            needed = required_iterations(best_count / score_points.shape[0], probability)

    if best_normal is None:
        return None

    normal, d = best_normal, float(best_d)
    inliers = np.abs(xyz @ normal + d) <= distance_threshold
    if refine and np.count_nonzero(inliers) >= 3:
        refined_normal, refined_d = fit_plane_least_squares(xyz[inliers])
        if reference is not None and refined_normal @ reference < 0:
            refined_normal, refined_d = -refined_normal, -refined_d
        acceptable = (cos_limit is None
                      or abs(refined_normal @ reference) >= cos_limit)
        refined_inliers = np.abs(xyz @ refined_normal + refined_d) <= distance_threshold
        if acceptable and refined_inliers.sum() >= inliers.sum():
            normal, d, inliers = refined_normal, refined_d, refined_inliers

    if np.count_nonzero(inliers) < max(3, min_inliers):
        return None
    residuals = xyz[inliers] @ normal + d
    return PlaneModel(normal=np.asarray(normal, dtype=np.float64), d=float(d), inliers=inliers,
                      iterations=int(iterations), rms=float(np.sqrt(np.mean(residuals ** 2))))
