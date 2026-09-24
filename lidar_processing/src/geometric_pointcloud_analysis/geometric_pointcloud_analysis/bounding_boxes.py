"""3D bounding boxes for point clusters.

``aabb``
    Axis-aligned box in the cloud frame. Fastest; loose for rotated objects.
``pca``
    Oriented box along the principal axes of the points. Tight for elongated
    objects, but the axes follow the point distribution and may tilt
    arbitrarily (e.g. a box around a bush may lean).
``upright``
    Gravity-aligned oriented box: vertical axis = "up", yaw chosen as the
    minimum-area rectangle of the points projected on the ground plane
    (rotating calipers on the 2D convex hull). This is usually the right choice
    for a ground robot: objects stand on the ground and only their yaw varies.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError


@dataclass
class BoundingBox:
    center: np.ndarray     # (3,)
    extent: np.ndarray     # (3,) full side lengths along the box axes
    rotation: np.ndarray   # (3, 3) columns = box axes in the cloud frame (right-handed)
    kind: str

    @property
    def volume(self):
        return float(np.prod(self.extent))

    def corners(self):
        """(8, 3) corners; the order matches ``EDGES``."""
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
                         dtype=np.float64)
        return self.center + (signs * (self.extent / 2.0)) @ self.rotation.T

    def quaternion(self):
        """(x, y, z, w) of ``rotation``."""
        return rotation_to_quaternion(self.rotation)


# Corner-index pairs forming the 12 box edges (see BoundingBox.corners).
EDGES = ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
         (0, 4), (1, 5), (2, 6), (3, 7))


def _right_handed(rotation):
    if np.linalg.det(rotation) < 0:
        rotation = rotation.copy()
        rotation[:, 2] *= -1.0
    return rotation


def _box_from_axes(points, rotation, kind):
    local = points @ rotation
    low, high = local.min(axis=0), local.max(axis=0)
    return BoundingBox(center=rotation @ ((low + high) / 2.0), extent=high - low,
                       rotation=rotation, kind=kind)


def aabb(points):
    return _box_from_axes(np.asarray(points, dtype=np.float64), np.eye(3), 'aabb')


def obb_pca(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return aabb(points)
    centered = points - points.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    rotation = _right_handed(vectors[:, ::-1])  # largest variance first
    return _box_from_axes(points, rotation, 'pca')


def obb_upright(points, up=(0.0, 0.0, 1.0)):
    points = np.asarray(points, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    flat = np.column_stack([points @ e1, points @ e2])

    angle = 0.0
    try:
        hull = flat[ConvexHull(flat).vertices]
        edges = np.roll(hull, -1, axis=0) - hull
        angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2))
        best_area = np.inf
        for candidate in angles:
            c, s = np.cos(candidate), np.sin(candidate)
            rotated = hull @ np.array([[c, -s], [s, c]])
            area = np.prod(rotated.max(axis=0) - rotated.min(axis=0))
            if area < best_area:
                best_area, angle = area, candidate
    except (QhullError, ValueError):
        # Fewer than 3 points or collinear footprint: align with the 2D principal axis.
        if len(flat) >= 2 and np.ptp(flat, axis=0).max() > 0:
            centered = flat - flat.mean(axis=0)
            _, vectors = np.linalg.eigh(centered.T @ centered)
            angle = float(np.arctan2(vectors[1, 1], vectors[0, 1]))

    c, s = np.cos(angle), np.sin(angle)
    a1 = c * e1 + s * e2
    a2 = np.cross(up, a1)
    rotation = np.column_stack([a1, a2, up])
    return _box_from_axes(points, rotation, 'upright')


def compute_box(points, kind='upright', up=(0.0, 0.0, 1.0)):
    if kind == 'aabb':
        return aabb(points)
    if kind == 'pca':
        return obb_pca(points)
    if kind == 'upright':
        return obb_upright(points, up)
    raise ValueError(f'Unknown box type {kind!r}')


def rotation_to_quaternion(rotation):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quaternion_rotate(quaternion, vector):
    """Rotate ``vector`` by quaternion (x, y, z, w)."""
    x, y, z, w = quaternion
    q = np.array([x, y, z])
    v = np.asarray(vector, dtype=np.float64)
    return v + 2.0 * np.cross(q, np.cross(q, v) + w * v)
