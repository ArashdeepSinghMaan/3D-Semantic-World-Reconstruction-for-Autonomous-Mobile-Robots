"""Test helpers: synthetic Ouster-like clouds and a stand-in for sensor_msgs.

The stand-in is only installed when the real ROS message packages are not
importable, so the same tests run inside and outside a ROS environment.
"""

import sys
import types

import numpy as np
import pytest


def _install_fake_ros_messages():
    try:
        import sensor_msgs.msg  # noqa: F401
        import diagnostic_msgs.msg  # noqa: F401
        return
    except ImportError:
        pass

    class _Msg:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class PointField(_Msg):
        INT8, UINT8, INT16, UINT16, INT32, UINT32, FLOAT32, FLOAT64 = range(1, 9)

    class PointCloud2(_Msg):
        pass

    class KeyValue(_Msg):
        pass

    class DiagnosticStatus(_Msg):
        OK, WARN, ERROR, STALE = b'\x00', b'\x01', b'\x02', b'\x03'

    class DiagnosticArray(_Msg):
        def __init__(self, **kwargs):
            self.header = types.SimpleNamespace(stamp=None, frame_id='')
            self.status = []
            super().__init__(**kwargs)

    for package, classes in (
            ('sensor_msgs', {'PointField': PointField, 'PointCloud2': PointCloud2}),
            ('diagnostic_msgs', {'DiagnosticArray': DiagnosticArray,
                                 'DiagnosticStatus': DiagnosticStatus,
                                 'KeyValue': KeyValue})):
        pkg = types.ModuleType(package)
        msg = types.ModuleType(f'{package}.msg')
        for name, cls in classes.items():
            setattr(msg, name, cls)
        pkg.msg = msg
        sys.modules[package] = pkg
        sys.modules[f'{package}.msg'] = msg


_install_fake_ros_messages()


OUSTER_DTYPE = np.dtype({
    # Layout used by the Ouster ROS driver's default point type (48 bytes).
    'names': ['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'ambient', 'range'],
    'formats': ['<f4', '<f4', '<f4', '<f4', '<u4', '<u2', '<u2', '<u2', '<u4'],
    'offsets': [0, 4, 8, 16, 20, 24, 26, 28, 32],
    'itemsize': 48,
})


def make_ouster_cloud(height=16, width=64, zero_fraction=0.2, seed=0):
    """Organized synthetic scan: rings of points on a cylinder-ish scene, some zeros."""
    rng = np.random.default_rng(seed)
    cloud = np.zeros(height * width, dtype=OUSTER_DTYPE)
    azimuth = np.tile(np.linspace(0, 2 * np.pi, width, endpoint=False), height)
    elevation = np.repeat(np.linspace(-0.3, 0.3, height), width)
    distance = rng.uniform(1.0, 40.0, height * width)
    cloud['x'] = distance * np.cos(elevation) * np.cos(azimuth)
    cloud['y'] = distance * np.cos(elevation) * np.sin(azimuth)
    cloud['z'] = distance * np.sin(elevation)
    cloud['intensity'] = rng.uniform(0, 1000, height * width)
    cloud['t'] = np.tile(np.linspace(0, 99_000_000, width).astype(np.uint32), height)
    cloud['ring'] = np.repeat(np.arange(height, dtype=np.uint16), width)
    cloud['range'] = (distance * 1000).astype(np.uint32)
    zeros = rng.random(height * width) < zero_fraction
    for axis in ('x', 'y', 'z'):
        cloud[axis][zeros] = 0.0
    return cloud


def make_message(cloud, height, width, is_bigendian=False, row_padding=0):
    """Build a duck-typed PointCloud2 message from a structured array."""
    from types import SimpleNamespace as NS

    dtype = cloud.dtype
    type_codes = {('f', 4): 7, ('f', 8): 8, ('u', 4): 6, ('u', 2): 4, ('u', 1): 2,
                  ('i', 4): 5, ('i', 2): 3, ('i', 1): 1}
    fields = [NS(name=n, offset=dtype.fields[n][1],
                 datatype=type_codes[(dtype.fields[n][0].kind, dtype.fields[n][0].itemsize)],
                 count=1) for n in dtype.names]
    data_dtype = dtype.newbyteorder('>') if is_bigendian else dtype
    rows = cloud.astype(data_dtype).reshape(height, width)
    payload = b''.join(row.tobytes() + b'\xab' * row_padding for row in rows)
    return NS(
        header=NS(frame_id='os_sensor', stamp=NS(sec=12, nanosec=500)),
        height=height, width=width, fields=fields, is_bigendian=is_bigendian,
        point_step=dtype.itemsize, row_step=width * dtype.itemsize + row_padding,
        is_dense=False, data=payload)


@pytest.fixture
def ouster_cloud():
    return make_ouster_cloud()
