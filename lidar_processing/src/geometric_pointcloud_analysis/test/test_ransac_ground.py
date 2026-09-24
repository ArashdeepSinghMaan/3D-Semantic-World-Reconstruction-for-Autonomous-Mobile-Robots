import numpy as np

from conftest import make_scene
from geometric_pointcloud_analysis.ground import plane_basis, segment_ground
from geometric_pointcloud_analysis.pipeline import config_from_dict
from geometric_pointcloud_analysis.ransac import (
    fit_plane_least_squares,
    fit_plane_ransac,
    required_iterations,
)


def _ground_scores(mask, truth):
    gt = truth == 0
    precision = (mask & gt).sum() / max(mask.sum(), 1)
    recall = (mask & gt).sum() / gt.sum()
    return precision, recall


def test_least_squares_recovers_plane():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-5, 5, (500, 2))
    z = 0.1 * xy[:, 0] - 0.2 * xy[:, 1] + 1.0
    normal, d = fit_plane_least_squares(np.column_stack([xy, z]))
    expected = np.array([-0.1, 0.2, 1.0]) / np.linalg.norm([-0.1, 0.2, 1.0])
    assert abs(abs(normal @ expected) - 1) < 1e-9


def test_required_iterations():
    assert required_iterations(1.0, 0.99) == 1
    assert required_iterations(0.0, 0.99) == float('inf')
    assert 30 < required_iterations(0.5, 0.99) < 40  # log(0.01) / log(1 - 0.125)


def test_ransac_ignores_outliers_and_orients_normal():
    xyz, truth = make_scene()
    plane = fit_plane_ransac(xyz, 0.08, reference_normal=(0, 0, 1), max_angle_deg=20, rng=0)
    assert plane.normal[2] > 0.999 and abs(plane.d - 0.5) < 0.02
    assert plane.rms < 0.03


def test_angle_constraint_rejects_dominant_wall():
    rng = np.random.default_rng(1)
    wall = np.column_stack([np.zeros(5000), rng.uniform(-5, 5, 5000), rng.uniform(0, 5, 5000)])
    floor = np.column_stack([rng.uniform(1, 5, (1000, 2)), np.zeros(1000)])
    xyz = np.vstack([wall, floor])
    unconstrained = fit_plane_ransac(xyz, 0.05, rng=0)
    assert abs(unconstrained.normal[0]) > 0.99          # the wall wins without a constraint
    constrained = fit_plane_ransac(xyz, 0.05, reference_normal=(0, 0, 1), max_angle_deg=20,
                                   rng=0)
    assert constrained.normal[2] > 0.99                 # the floor wins with it
    assert fit_plane_ransac(wall, 0.05, reference_normal=(0, 0, 1), max_angle_deg=20,
                            rng=0) is None


def test_ransac_degenerate_input():
    assert fit_plane_ransac(np.zeros((2, 3))) is None
    assert fit_plane_ransac(np.zeros((100, 3)), rng=0) is None  # all samples collinear


def test_flat_and_sloped_ground():
    for slope in (0.0, 8.0):
        xyz, truth = make_scene(slope_deg=slope)
        config, _ = config_from_dict({'ransac': {'seed': 0}})
        result = segment_ground(xyz, config, (0, 0, 1), rng=0)
        precision, recall = _ground_scores(result.mask, truth)
        assert precision > 0.97 and recall > 0.98
        tilt = np.degrees(np.arccos(result.plane.normal[2]))
        assert abs(tilt - slope) < 0.5


def test_grid_beats_single_plane_on_rolling_terrain():
    xyz, truth = make_scene(undulation=0.6)
    recalls = {}
    for method in ('ransac', 'grid'):
        config, _ = config_from_dict({'ground': {'method': method}, 'ransac': {'seed': 0}})
        result = segment_ground(xyz, config, (0, 0, 1), rng=0)
        precision, recall = _ground_scores(result.mask, truth)
        assert precision > 0.9
        recalls[method] = recall
    assert recalls['grid'] > recalls['ransac'] + 0.3
    cells = segment_ground(xyz, config, (0, 0, 1), rng=0).cells
    assert cells and any(c.accepted for c in cells)


def test_no_ground_status():
    rng = np.random.default_rng(2)
    wall = np.column_stack([np.zeros(2000), rng.uniform(-5, 5, 2000), rng.uniform(0, 5, 2000)])
    config, _ = config_from_dict({})
    result = segment_ground(wall, config, (0, 0, 1), rng=0)
    assert result.status == 'no_plane_within_angle' and not result.mask.any()


def test_candidate_height_filter():
    xyz, _ = make_scene()
    config, _ = config_from_dict({'ground': {'candidate_max_height_enabled': True,
                                             'candidate_max_height': -10.0}})
    assert segment_ground(xyz, config, (0, 0, 1)).status == 'too_few_candidates'


def test_plane_basis_orthonormal():
    for n in ([0, 0, 1], [1, 0, 0], [0.3, -0.4, 0.8]):
        n = np.asarray(n, float) / np.linalg.norm(n)
        u, v = plane_basis(n)
        m = np.column_stack([u, v, n])
        assert np.allclose(m.T @ m, np.eye(3)) and np.linalg.det(m) > 0
