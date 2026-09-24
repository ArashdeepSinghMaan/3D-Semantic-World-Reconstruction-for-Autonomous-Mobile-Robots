import numpy as np

from geometric_pointcloud_analysis.bounding_boxes import (
    aabb,
    compute_box,
    obb_pca,
    obb_upright,
    quaternion_rotate,
    rotation_to_quaternion,
)
from geometric_pointcloud_analysis.clustering import euclidean_clusters
from geometric_pointcloud_analysis.normals import estimate_normals
from geometric_pointcloud_analysis.surface_analysis import (
    dominant_shape,
    eigen_features,
    local_features,
)


def test_euclidean_clusters_separates_and_filters():
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0, 0], 0.1, (300, 3))
    b = rng.normal([5, 0, 0], 0.1, (100, 3))
    lonely = np.array([[20.0, 0, 0], [20.05, 0, 0]])
    result = euclidean_clusters(np.vstack([a, b, lonely]), 0.3, min_points=5)
    assert list(result.sizes) == [300, 100]            # sorted by size
    assert (result.labels[:300] == 0).all() and (result.labels[300:400] == 1).all()
    assert (result.labels[400:] == -1).all() and result.rejected_small == 2
    capped = euclidean_clusters(np.vstack([a, b]), 0.3, min_points=5, max_points=200)
    assert list(capped.sizes) == [100] and capped.rejected_large == 300


def test_clustering_chains_points():
    line = np.column_stack([np.arange(0, 10, 0.2), np.zeros(50), np.zeros(50)])
    assert len(euclidean_clusters(line, 0.25).sizes) == 1
    assert len(euclidean_clusters(line, 0.15).sizes) == 50


def test_clustering_empty():
    assert euclidean_clusters(np.zeros((0, 3)), 0.2).labels.size == 0


def test_normals_plane_sphere_and_orientation():
    rng = np.random.default_rng(1)
    plane = np.column_stack([rng.uniform(-2, 2, (2000, 2)), np.full(2000, -1.0)])
    result = estimate_normals(plane, 20, orientation='viewpoint')
    assert np.all(result.normals[:, 2] > 0.999)       # faces the sensor above the plane
    assert np.nanmax(result.curvature) < 1e-6
    down = estimate_normals(plane, 20, orientation='reference', reference=(0, 0, -1))
    assert np.all(down.normals[:, 2] < -0.999)

    v = rng.normal(size=(3000, 3))
    sphere = 2.0 * v / np.linalg.norm(v, axis=1, keepdims=True)
    sphere_normals = estimate_normals(sphere, 15, orientation='none').normals
    radial = sphere / 2.0
    assert np.mean(np.abs(np.einsum('ij,ij->i', sphere_normals, radial))) > 0.99


def test_normals_radius_limit_marks_isolated_points_invalid():
    xyz = np.vstack([np.random.default_rng(2).uniform(0, 1, (200, 3)), [[50.0, 50, 50]]])
    result = estimate_normals(xyz, 10, radius=0.5, min_neighbors=5)
    assert not result.valid[-1] and result.valid[:-1].mean() > 0.95


def _rotated_box(yaw_deg=30, size=(4.0, 1.0, 1.0), n=3000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-np.array(size) / 2, np.array(size) / 2, (n, 3))
    a = np.radians(yaw_deg)
    r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return p @ r.T + [5.0, 1.0, 0.0]


def _contains(box, points):
    local = (points - box.center) @ box.rotation
    return np.all(np.abs(local) <= box.extent / 2 + 1e-9)


def test_box_types():
    points = _rotated_box()
    boxes = {kind: compute_box(points, kind) for kind in ('aabb', 'pca', 'upright')}
    for box in boxes.values():
        assert _contains(box, points)
        assert np.isclose(np.linalg.det(box.rotation), 1.0)
    assert np.allclose(sorted(boxes['upright'].extent), [1, 1, 4], atol=0.05)
    assert boxes['upright'].volume < boxes['pca'].volume < boxes['aabb'].volume
    assert np.allclose(boxes['upright'].rotation[:, 2], [0, 0, 1])


def test_box_degenerate_inputs():
    assert aabb(np.array([[1.0, 2, 3]])).volume == 0
    line = np.column_stack([np.linspace(0, 1, 10), np.zeros(10), np.zeros(10)])
    assert np.isclose(obb_upright(line).extent.max(), 1.0)
    assert obb_pca(line[:2]).kind == 'aabb'


def test_box_corners_and_quaternion():
    box = obb_upright(_rotated_box(yaw_deg=65))
    shrunk = box.corners() * 0.999 + box.center * 0.001
    assert box.corners().shape == (8, 3) and _contains(box, shrunk)
    q = rotation_to_quaternion(box.rotation)
    for axis in range(3):
        assert np.allclose(quaternion_rotate(q, np.eye(3)[axis]), box.rotation[:, axis])


def test_shape_features():
    rng = np.random.default_rng(3)
    line = np.column_stack([rng.uniform(0, 5, 500), rng.normal(0, 0.01, (500, 2))])
    wall = np.column_stack([np.zeros(500), rng.uniform(0, 3, 500), rng.uniform(0, 3, 500)])
    blob = rng.normal(0, 1, (500, 3))
    assert dominant_shape(eigen_features(line)) == 'linear'
    assert dominant_shape(eigen_features(wall)) == 'planar'
    assert eigen_features(wall)['verticality'] > 0.99
    assert dominant_shape(eigen_features(blob)) == 'scattered'
    normals = estimate_normals(wall, 20)
    local = local_features(normals.eigenvalues, normals.normals)
    assert np.nanmean(local['planarity']) > 0.5 and np.nanmean(local['verticality']) > 0.99
