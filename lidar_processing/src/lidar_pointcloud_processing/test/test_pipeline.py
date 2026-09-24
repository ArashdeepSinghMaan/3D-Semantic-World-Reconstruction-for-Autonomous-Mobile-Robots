from pathlib import Path

import numpy as np
import pytest

from conftest import make_ouster_cloud
from lidar_pointcloud_processing.diagnostics import build_diagnostic_array, RollingStats
from lidar_pointcloud_processing.pipeline import (
    config_from_dict,
    config_to_flat_dict,
    flat_to_nested,
    load_config_yaml,
    PipelineConfig,
    PointCloudPipeline,
)

CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'pointcloud_processing.yaml'


def test_counts_are_consistent():
    cloud = make_ouster_cloud()
    result = PointCloudPipeline().process(cloud, 16, 64)
    s = result.stats
    removed = s.nonfinite_removed + s.zero_removed + s.range_removed + s.crop_removed
    assert s.input_points == cloud.size
    assert s.filtered_points == s.input_points - removed
    assert s.zero_removed > 0  # zeros are not caught by the finiteness check
    assert result.filtered.shape[0] == s.filtered_points
    assert result.filtered.dtype == cloud.dtype  # all fields preserved
    assert 0 < s.voxel_points <= s.filtered_points
    assert set(result.voxelized.dtype.names) == {'x', 'y', 'z', 'intensity', 'point_count'}
    assert result.voxelized['point_count'].sum() == s.filtered_points


def test_keep_organized():
    cloud = make_ouster_cloud()
    config, _ = config_from_dict({'output': {'keep_organized': True}})
    result = PointCloudPipeline(config).process(cloud, 16, 64)
    assert (result.filtered_height, result.filtered_width) == (16, 64)
    assert result.filtered.shape[0] == cloud.size
    assert np.isnan(result.filtered['x']).sum() == cloud.size - result.stats.filtered_points
    assert not result.filtered_is_dense


def test_outlier_stages():
    cloud = make_ouster_cloud(height=32, width=256)
    for stage in ('filtered', 'voxelized'):
        config, _ = config_from_dict({'outlier': {'method': 'statistical', 'apply_to': stage}})
        result = PointCloudPipeline(config).process(cloud, 32, 256)
        assert result.stats.outliers_removed > 0
        assert 'outlier' in result.stats.timings_ms


def test_missing_average_field_is_reported():
    cloud = make_ouster_cloud()
    config, _ = config_from_dict({'voxel': {'average_fields': ['intensity', 'nope']}})
    result = PointCloudPipeline(config).process(cloud, 16, 64)
    assert result.missing_fields == ('nope',)


def test_config_coercion_and_validation():
    config, unknown = config_from_dict({
        'range_filter': {'max_range': 30, 'enabled': 'true'},
        'voxel': {'average_fields': [''], 'min_points_per_voxel': 2.0},
        'bogus': {'a': 1},
        'outlier': {'unknown_key': 1},
    })
    assert config.range_filter.max_range == 30.0
    assert config.voxel.average_fields == ()
    assert config.voxel.min_points_per_voxel == 2
    assert set(unknown) == {'bogus', 'outlier.unknown_key'}
    with pytest.raises(ValueError):
        config_from_dict({'voxel': {'min_points_per_voxel': 1.5}})
    bad, _ = config_from_dict({'range_filter': {'min_range': 5.0, 'max_range': 1.0}})
    with pytest.raises(ValueError):
        bad.validate()


def test_radius_warning():
    config, _ = config_from_dict({'outlier': {'method': 'radius', 'radius': 0.05}})
    assert any('2 x voxel_size' in w for w in config.warnings())


def test_flat_roundtrip_and_yaml():
    flat = config_to_flat_dict(PipelineConfig())
    assert isinstance(flat['crop_box.min_xyz'], list)
    config, unknown = config_from_dict(flat_to_nested(flat))
    assert config == PipelineConfig() and not unknown
    loaded, unknown = load_config_yaml(CONFIG)
    assert not unknown
    assert loaded.voxel.voxel_size == 0.05 and loaded.outlier.method == 'none'


def test_yaml_matches_node_parameter_names():
    """Every pipeline key in the YAML must be a declared node parameter and vice versa."""
    import yaml
    params = yaml.safe_load(CONFIG.read_text())['pointcloud_processing']['ros__parameters']
    yaml_keys = {f'{s}.{k}' for s, v in params.items() if isinstance(v, dict) for k in v}
    assert yaml_keys == set(config_to_flat_dict(PipelineConfig()))


def test_diagnostics():
    result = PointCloudPipeline().process(make_ouster_cloud(), 16, 64)
    rolling = RollingStats(10)
    rolling.add(result.stats)
    msg = build_diagnostic_array(result.stats, rolling, stamp=None, name='test')
    keys = {kv.key for kv in msg.status[0].values}
    assert {'input_points', 'voxel_points', 'time_total_ms'} <= keys
    assert 'Voxelized points' in rolling.format_summary()
    assert 'Input points' in result.stats.format_report()
