import numpy as np

from conftest import make_ouster_cloud
from lidar_pointcloud_processing.analysis import (
    cloud_quality,
    nearest_offsets,
    offset_statistics,
    stamp_statistics,
)
from lidar_pointcloud_processing.tools.bag_io import cdr_header_stamp


def test_stamp_statistics_detects_gap():
    stamps = np.arange(0, 10, 0.1).tolist()
    del stamps[50]
    stats = stamp_statistics(stamps)
    assert abs(stats['dt_median_ms'] - 100) < 1e-6
    assert stats['gaps_gt_1.5x_median'] == 1 and stats['non_monotonic'] == 0


def test_nearest_offsets():
    lidar = np.array([0.0, 0.1, 0.2])
    camera = np.array([0.01, 0.06, 0.11, 0.16, 0.21])
    offsets = nearest_offsets(lidar, camera)
    np.testing.assert_allclose(offsets, [0.01, 0.01, 0.01])
    assert abs(offset_statistics(offsets)['mean_ms'] - 10) < 1e-9
    np.testing.assert_allclose(nearest_offsets([1.0], [0.5]), [-0.5])


def test_cloud_quality():
    quality = cloud_quality(make_ouster_cloud())
    assert quality['zero'] > 0 and quality['rings'] == 16
    assert 90 < quality['t_span_ms_if_ns'] < 100


def test_cdr_header_stamp():
    import struct
    raw = b'\x00\x01\x00\x00' + struct.pack('<iI', 1690962427, 4361528) + b'rest'
    assert abs(cdr_header_stamp(raw) - 1690962427.004361528) < 1e-6
