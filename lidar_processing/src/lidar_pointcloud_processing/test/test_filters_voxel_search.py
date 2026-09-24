import numpy as np

from lidar_pointcloud_processing.filters import (
    crop_box_mask,
    finite_mask,
    nonzero_mask,
    radius_outlier_mask,
    range_mask,
    statistical_outlier_mask,
)
from lidar_pointcloud_processing.spatial_search import (
    benchmark_knn,
    brute_force_knn,
    brute_force_radius_count,
    KDTreeIndex,
)
from lidar_pointcloud_processing.voxelization import voxel_downsample


def test_basic_masks():
    xyz = np.array([[0, 0, 0], [np.nan, 1, 1], [1, 0, 0], [10, 0, 0], [0.1, 0.1, 0.1]], float)
    np.testing.assert_array_equal(finite_mask(xyz), [1, 0, 1, 1, 1])
    np.testing.assert_array_equal(nonzero_mask(xyz[[0, 2, 3, 4]]), [0, 1, 1, 1])
    np.testing.assert_array_equal(range_mask(xyz, 0.5, 5.0), [0, 0, 1, 0, 0])
    inside = crop_box_mask(xyz, [-0.2] * 3, [0.2] * 3, negative=False)
    np.testing.assert_array_equal(inside, [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(
        crop_box_mask(xyz, [-0.2] * 3, [0.2] * 3, negative=True), [0, 1, 1, 1, 0])


def test_outlier_filters_remove_isolated_point():
    rng = np.random.default_rng(1)
    cluster = rng.normal(0, 0.05, (500, 3))
    xyz = np.vstack([cluster, [[5.0, 5.0, 5.0]]])
    assert not statistical_outlier_mask(xyz, 20, 2.0)[-1]
    radius_keep = radius_outlier_mask(xyz, 0.2, 5)
    assert not radius_keep[-1] and radius_keep[:-1].mean() > 0.95


def test_radius_count_includes_self():
    xyz = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    np.testing.assert_array_equal(radius_outlier_mask(xyz, 0.5, 1), [True, True])
    np.testing.assert_array_equal(radius_outlier_mask(xyz, 0.5, 2), [False, False])


def test_voxel_centroids_counts_and_values():
    xyz = np.array([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [0.51, 0.0, 0.0],
                    [-0.01, 0.0, 0.0]])
    grid = voxel_downsample(xyz, 0.1, values={'intensity': np.array([1.0, 3.0, 10.0, 7.0])})
    assert len(grid.centroids) == 3
    first = grid.point_to_voxel[0]
    assert grid.point_to_voxel[1] == first
    np.testing.assert_allclose(grid.centroids[first], [0.02, 0.02, 0.02])
    assert grid.counts[first] == 2
    assert grid.values['intensity'][first] == 2.0
    assert grid.point_to_voxel[3] != first  # negative coordinates land in another voxel


def test_voxel_min_points():
    xyz = np.array([[0.01, 0, 0], [0.02, 0, 0], [5.0, 0, 0]])
    grid = voxel_downsample(xyz, 0.1, min_points_per_voxel=2)
    assert len(grid.centroids) == 1
    np.testing.assert_array_equal(grid.point_to_voxel, [0, 0, -1])


def test_voxel_empty():
    assert voxel_downsample(np.zeros((0, 3)), 0.1).centroids.shape == (0, 3)


def test_kdtree_matches_brute_force():
    rng = np.random.default_rng(2)
    points = rng.uniform(-10, 10, (3000, 3))
    queries = points[:50]
    index = KDTreeIndex(points)
    tree_d, _ = index.query_knn(queries, 8)
    brute_d, _ = brute_force_knn(points, queries, 8)
    np.testing.assert_allclose(tree_d, brute_d, atol=1e-6)  # expansion round-off
    np.testing.assert_array_equal(index.count_radius(queries, 1.5),
                                  brute_force_radius_count(points, queries, 1.5))
    assert index.query_knn(points[0], 1)[0].shape == (1, 1)
    near = index.query_radius(queries[:1], 2.0, sort=True)[0]
    d = np.linalg.norm(points[near] - queries[0], axis=1)
    assert np.all(np.diff(d) >= 0)


def test_benchmark_runs():
    points = np.random.default_rng(3).uniform(-5, 5, (2000, 3))
    result = benchmark_knn(points, n_queries=100, k=5, radius=0.5)
    assert result['knn_agree'] and result['radius_count_max_diff'] <= 1
