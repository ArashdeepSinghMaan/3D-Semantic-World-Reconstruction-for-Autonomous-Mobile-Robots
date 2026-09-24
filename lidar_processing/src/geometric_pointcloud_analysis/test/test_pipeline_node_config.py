from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from conftest import make_scene
from geometric_pointcloud_analysis.diagnostics import build_diagnostic_array, RollingTimes
from geometric_pointcloud_analysis.markers import (
    build_marker_array,
    cluster_color,
    label_colors,
    pack_rgb,
)
from geometric_pointcloud_analysis.pipeline import (
    config_from_dict,
    config_to_flat_dict,
    GeometryConfig,
    GeometryPipeline,
    load_config_yaml,
)

CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'geometric_analysis.yaml'


def test_pipeline_finds_objects_with_sensible_shapes():
    xyz, truth = make_scene()
    result = GeometryPipeline(config_from_dict({'ransac': {'seed': 0}})[0]).process(xyz)
    assert len(result.clusters) == 4
    assert result.stats['ground_status'] == 'ok'
    by_size = {c.point_count: c for c in result.clusters}
    wall = max(result.clusters, key=lambda c: c.point_count)
    assert wall.shape == 'planar' and wall.features['verticality'] > 0.9
    assert 1.8 < wall.height_max < 2.1
    bush = min(result.clusters, key=lambda c: abs(c.centroid[1] + 7))
    assert bush.shape == 'scattered'
    pole = min(result.clusters, key=lambda c: abs(c.centroid[0] + 4) + abs(c.centroid[1] - 5))
    assert pole.box.extent[2] > 2.2 and max(pole.box.extent[:2]) < 0.3
    # Every clustered point is non-ground, and each cluster is mostly one true object.
    assert not (result.labels >= 0)[result.ground_mask].any()
    for c in result.clusters:
        ids = truth[result.labels == c.id]
        assert np.bincount(ids).max() / len(ids) > 0.95
    assert len(by_size) == 4


def test_clustering_all_points_merges_ground():
    # Dense ground (~35 points / m^2, spacing below the tolerance), like a voxelized scan.
    xyz, truth = make_scene(ground_points=100000)
    config, _ = config_from_dict({'clustering': {'input': 'all', 'max_points': 0},
                                  'ransac': {'seed': 0}, 'normals': {'enabled': False}})
    merged = GeometryPipeline(config).process(xyz).clusters[0]
    assert merged.point_count > 50000                    # ground + objects in one cluster
    assert any('ground will connect' in w for w in config.warnings())


def test_min_height_filter_and_disabled_stages():
    xyz, _ = make_scene()
    config, _ = config_from_dict({'clustering': {'min_height_above_ground': 2.2},
                                  'ransac': {'seed': 0}, 'normals': {'enabled': False},
                                  'boxes': {'enabled': False}})
    result = GeometryPipeline(config).process(xyz)
    assert len(result.clusters) == 1 and result.clusters[0].box is None  # only the pole
    assert result.normals is None
    off, _ = config_from_dict({'ground': {'enabled': False}, 'clustering': {'enabled': False}})
    result = GeometryPipeline(off).process(xyz)
    assert result.ground is None and not result.clusters


def test_reference_normal_override():
    xyz, _ = make_scene()
    xyz = xyz[:, [2, 1, 0]] * [-1, 1, 1]   # "up" is now -x in this frame
    config, _ = config_from_dict({'ransac': {'seed': 0}})
    # With the wrong "up" (+z) the best horizontal plane is the wall, not the ground.
    wrong = GeometryPipeline(config).process(xyz)
    assert wrong.stats['ground_points'] < 5000
    result = GeometryPipeline(config).process(xyz, reference_normal=(-1, 0, 0))
    assert result.stats['ground_points'] > 19000 and result.up[0] < -0.99


def test_config_validation_and_coercion():
    config, unknown = config_from_dict({'ransac': {'max_iterations': 500.0},
                                        'ground': {'reference_frame': 'Base_Link'},
                                        'bogus': {}})
    assert config.ransac.max_iterations == 500 and unknown == ['bogus']
    assert config.ground.reference_frame == 'Base_Link'   # case preserved
    for bad in ({'ground': {'method': 'magic'}}, {'ransac': {'probability': 1.0}},
                {'clustering': {'max_points': 5, 'min_points': 10}},
                {'boxes': {'type': 'sphere'}}, {'ground': {'reference_normal': [0, 0, 0]}}):
        with pytest.raises(ValueError):
            config_from_dict(bad)[0].validate()
    tight, _ = config_from_dict({'clustering': {'tolerance': 0.05}})
    assert tight.warnings(input_voxel_size=0.05)


def test_yaml_loads_and_matches_declared_parameters():
    config, unknown = load_config_yaml(CONFIG)
    assert config == GeometryConfig()           # the YAML documents the defaults
    assert set(unknown) == {'markers'}           # node-only marker options
    params = yaml.safe_load(CONFIG.read_text())['geometric_analysis']['ros__parameters']
    yaml_keys = {f'{s}.{k}' for s, v in params.items() if isinstance(v, dict) and s != 'markers'
                 for k in v}
    assert yaml_keys == set(config_to_flat_dict(GeometryConfig()))


def test_colors_and_packing():
    assert cluster_color(0) != cluster_color(1)
    packed = pack_rgb(np.array([[1.0, 0.5, 0.0]])).view(np.uint32)[0]
    assert packed == (255 << 16 | 128 << 8 | 0)
    colors = label_colors(np.array([-1, 0, 0, 1]))
    assert np.allclose(colors[0], 0.5) and np.allclose(colors[1], colors[2])


def test_markers_and_diagnostics():
    xyz, _ = make_scene(undulation=0.6)
    for method in ('ransac', 'grid'):
        config, _ = config_from_dict({'ground': {'method': method}, 'ransac': {'seed': 0}})
        result = GeometryPipeline(config).process(xyz)
        header = NS(frame_id='os_sensor', stamp=None)
        array = build_marker_array(result, header, xyz=xyz, normals_max=50)
        namespaces = [m.ns for m in array.markers]
        assert array.markers[0].action == 3                          # DELETEALL first
        assert namespaces.count('boxes') == len(result.clusters)
        assert 'ground' in namespaces and 'normals' in namespaces
        box = next(m for m in array.markers if m.ns == 'boxes')
        assert len(box.points) == 24
        ground = next(m for m in array.markers if m.ns == 'ground')
        if method == 'grid':
            assert len(ground.points) == 6 * len(result.ground.cells) == len(ground.colors)
        rolling = RollingTimes()
        rolling.add(result.total_ms, result.timings_ms, result.stats)
        diag = build_diagnostic_array(result.stats, result.timings_ms, result.total_ms,
                                      rolling, None, 'test')
        assert {'ground_points', 'clusters'} <= {kv.key for kv in diag.status[0].values}
        assert 'clusters' in rolling.format_summary()
