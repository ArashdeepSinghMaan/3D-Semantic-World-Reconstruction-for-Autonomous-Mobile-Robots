import numpy as np
import pytest

from conftest import make_message, make_ouster_cloud
from lidar_pointcloud_processing.pointcloud_utils import (
    array_to_pointcloud2,
    describe_layout,
    fields_to_dtype,
    pointcloud2_to_array,
    scalar_field_names,
    xyz_from_array,
)


def _assert_same(a, b):
    assert a.dtype.names == b.dtype.names
    for name in a.dtype.names:
        np.testing.assert_array_equal(a[name], b[name])


def test_decode_preserves_every_field():
    cloud = make_ouster_cloud()
    decoded = pointcloud2_to_array(make_message(cloud, 16, 64))
    assert decoded.dtype.itemsize == 48
    _assert_same(decoded, cloud)


def test_decode_big_endian_returns_native():
    cloud = make_ouster_cloud()
    decoded = pointcloud2_to_array(make_message(cloud, 16, 64, is_bigendian=True))
    assert decoded.dtype.isnative
    _assert_same(decoded, cloud)


def test_decode_row_step_padding():
    cloud = make_ouster_cloud()
    decoded = pointcloud2_to_array(make_message(cloud, 16, 64, row_padding=8))
    _assert_same(decoded, cloud)


def test_decode_rejects_short_buffer():
    msg = make_message(make_ouster_cloud(), 16, 64)
    msg.data = msg.data[:-1]
    with pytest.raises(ValueError):
        pointcloud2_to_array(msg)


def test_field_outside_point_step_rejected():
    from types import SimpleNamespace as NS
    with pytest.raises(ValueError):
        fields_to_dtype([NS(name='x', offset=12, datatype=7, count=1)], point_step=12)


def test_roundtrip_through_pointcloud2():
    cloud = make_ouster_cloud()
    msg = make_message(cloud, 16, 64)
    encoded = array_to_pointcloud2(pointcloud2_to_array(msg), msg.header, 16, 64)
    assert encoded.point_step == 48 and encoded.row_step == 48 * 64
    assert [f.name for f in encoded.fields] == list(cloud.dtype.names)
    encoded.data = bytes(encoded.data)
    _assert_same(pointcloud2_to_array(encoded), cloud)


def test_encode_is_dense_detection():
    cloud = make_ouster_cloud()
    cloud['x'][0] = np.nan
    msg = array_to_pointcloud2(cloud, header=None)
    assert msg.is_dense is False and msg.height == 1


def test_helpers():
    cloud = make_ouster_cloud()
    assert xyz_from_array(cloud).shape == (cloud.size, 3)
    assert 't' in scalar_field_names(cloud)
    text = describe_layout(make_message(cloud, 16, 64))
    assert 'os_sensor' in text and 'reflectivity' in text
