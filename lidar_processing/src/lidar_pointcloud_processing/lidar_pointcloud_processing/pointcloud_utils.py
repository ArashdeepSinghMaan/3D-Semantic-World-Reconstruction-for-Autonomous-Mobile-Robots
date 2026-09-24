"""Conversions between PointCloud2 messages and NumPy structured arrays.

The decoding functions are duck-typed on the message: they work with rclpy
messages and with messages deserialized by the ``rosbags`` library, so the
live node and the offline tools share a single implementation.

Decoding builds a NumPy structured dtype directly from ``msg.fields`` and
``msg.point_step`` and views the raw byte buffer through it. This is
zero-copy for the common case, keeps every field the driver published (for
an Ouster: x, y, z, intensity, t, reflectivity, ring, ambient, range, ...)
and is orders of magnitude faster than iterating with ``read_points``.

ROS imports happen lazily inside ``array_to_pointcloud2`` only, so this module
can be imported without a ROS installation.
"""

import array
import sys

import numpy as np

# sensor_msgs/msg/PointField datatype constants.
INT8 = 1
UINT8 = 2
INT16 = 3
UINT16 = 4
INT32 = 5
UINT32 = 6
FLOAT32 = 7
FLOAT64 = 8

DATATYPE_TO_NUMPY = {
    INT8: 'i1', UINT8: 'u1', INT16: 'i2', UINT16: 'u2',
    INT32: 'i4', UINT32: 'u4', FLOAT32: 'f4', FLOAT64: 'f8',
}

DATATYPE_NAMES = {
    INT8: 'INT8', UINT8: 'UINT8', INT16: 'INT16', UINT16: 'UINT16',
    INT32: 'INT32', UINT32: 'UINT32', FLOAT32: 'FLOAT32', FLOAT64: 'FLOAT64',
}

_NUMPY_TO_DATATYPE = {
    ('i', 1): INT8, ('u', 1): UINT8, ('i', 2): INT16, ('u', 2): UINT16,
    ('i', 4): INT32, ('u', 4): UINT32, ('f', 4): FLOAT32, ('f', 8): FLOAT64,
}


def fields_to_dtype(fields, point_step, is_bigendian=False):
    """Build a structured dtype matching a PointCloud2 memory layout.

    Padding between fields and at the end of each point is preserved through
    explicit offsets and ``itemsize=point_step``. Fields with an empty name
    (used by some drivers for padding) are skipped.
    """
    point_step = int(point_step)
    if point_step <= 0:
        raise ValueError(f'point_step must be positive, got {point_step}')

    order = '>' if is_bigendian else '<'
    names, formats, offsets = [], [], []
    for field in sorted(fields, key=lambda f: int(f.offset)):
        if not field.name:
            continue
        if field.datatype not in DATATYPE_TO_NUMPY:
            raise ValueError(
                f"Field '{field.name}' has unsupported datatype {field.datatype}")
        if field.name in names:
            raise ValueError(f"Duplicate field name '{field.name}'")
        base = np.dtype(order + DATATYPE_TO_NUMPY[field.datatype])
        count = int(field.count) if int(field.count) > 0 else 1
        offset = int(field.offset)
        if offset + base.itemsize * count > point_step:
            raise ValueError(
                f"Field '{field.name}' (offset {offset}, {count} x {base.itemsize} bytes) "
                f'does not fit in point_step {point_step}')
        names.append(field.name)
        formats.append(base if count == 1 else (base, (count,)))
        offsets.append(offset)

    return np.dtype({
        'names': names,
        'formats': formats,
        'offsets': offsets,
        'itemsize': point_step,
    })


def pointcloud2_to_array(msg):
    """Decode a PointCloud2 into a flat (N,) structured array in native byte order.

    N = height * width. For organized clouds, reshape with
    ``arr.reshape(msg.height, msg.width)`` to recover the image layout.
    Rows with ``row_step`` padding are handled.

    The returned array may be a read-only view of the message buffer; copy it
    before modifying it in place.
    """
    height, width = int(msg.height), int(msg.width)
    point_step, row_step = int(msg.point_step), int(msg.row_step)
    dtype = fields_to_dtype(msg.fields, point_step, bool(msg.is_bigendian))

    n_points = height * width
    if n_points == 0:
        return np.zeros(0, dtype=_native(dtype))

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    row_bytes = width * point_step

    if row_step == row_bytes or height == 1:
        needed = n_points * point_step
        if raw.size < needed:
            raise ValueError(
                f'PointCloud2 data has {raw.size} bytes, expected at least {needed}')
        cloud = raw[:needed].view(dtype)
    else:
        if row_step < row_bytes:
            raise ValueError(
                f'row_step {row_step} is smaller than width * point_step = {row_bytes}')
        needed = (height - 1) * row_step + row_bytes
        if raw.size < needed:
            raise ValueError(
                f'PointCloud2 data has {raw.size} bytes, expected at least {needed}')
        rows = np.lib.stride_tricks.as_strided(
            raw, shape=(height, row_bytes), strides=(row_step, 1))
        cloud = np.ascontiguousarray(rows).reshape(-1).view(dtype)

    if not dtype.isnative:
        cloud = cloud.astype(_native(dtype))
    return cloud


def _native(dtype):
    return dtype if dtype.isnative else dtype.newbyteorder('=')


def xyz_from_array(cloud, dtype=np.float64):
    """Return an (N, 3) array of x, y, z from a structured cloud."""
    cloud = np.asarray(cloud).reshape(-1)
    for name in ('x', 'y', 'z'):
        if cloud.dtype.names is None or name not in cloud.dtype.names:
            raise ValueError(f"Point cloud has no '{name}' field")
    xyz = np.empty((cloud.shape[0], 3), dtype=dtype)
    xyz[:, 0] = cloud['x']
    xyz[:, 1] = cloud['y']
    xyz[:, 2] = cloud['z']
    return xyz


def scalar_field_names(cloud_or_dtype):
    """Names of fields that hold a single number per point (no sub-arrays)."""
    dtype = getattr(cloud_or_dtype, 'dtype', cloud_or_dtype)
    if dtype.names is None:
        return ()
    return tuple(
        name for name in dtype.names
        if dtype.fields[name][0].subdtype is None
        and dtype.fields[name][0].kind in 'iuf')


def describe_layout(msg):
    """Human-readable description of a PointCloud2 header and field layout."""
    stamp = msg.header.stamp
    organized = int(msg.height) > 1
    lines = [
        'PointCloud2 layout',
        '─' * 60,
        f'frame_id      : {msg.header.frame_id}',
        f'stamp         : {stamp.sec}.{stamp.nanosec:09d}',
        f'height x width: {msg.height} x {msg.width} '
        f'({"organized" if organized else "unorganized"}, '
        f'{int(msg.height) * int(msg.width)} points)',
        f'point_step    : {msg.point_step} bytes',
        f'row_step      : {msg.row_step} bytes',
        f'is_dense      : {bool(msg.is_dense)}',
        f'is_bigendian  : {bool(msg.is_bigendian)}',
        f'data size     : {len(msg.data)} bytes',
        'fields:',
        f'  {"name":<14}{"offset":>7}  {"datatype":<9}{"count":>6}',
    ]
    for field in sorted(msg.fields, key=lambda f: int(f.offset)):
        type_name = DATATYPE_NAMES.get(field.datatype, f'?{field.datatype}')
        lines.append(f'  {field.name:<14}{field.offset:>7}  {type_name:<9}{field.count:>6}')
    return '\n'.join(lines)


def array_to_pointcloud2(cloud, header, height=None, width=None, is_dense=None):
    """Encode a structured array as a sensor_msgs/msg/PointCloud2.

    ``cloud`` is flattened; pass ``height`` and ``width`` to publish an
    organized cloud (height * width must equal the number of points).
    ``is_dense`` defaults to "all x, y, z are finite".
    """
    from sensor_msgs.msg import PointCloud2, PointField  # lazy ROS import

    flat = np.asarray(cloud).reshape(-1)
    if not flat.dtype.isnative:
        flat = flat.astype(_native(flat.dtype))
    if height is None or width is None:
        height, width = 1, flat.shape[0]
    if int(height) * int(width) != flat.shape[0]:
        raise ValueError(
            f'height * width = {int(height) * int(width)} does not match '
            f'{flat.shape[0]} points')

    fields = []
    for name in flat.dtype.names:
        sub_dtype, offset = flat.dtype.fields[name][:2]
        if sub_dtype.subdtype is not None:
            base, shape = sub_dtype.subdtype
            count = int(np.prod(shape))
        else:
            base, count = sub_dtype, 1
        key = (base.kind, base.itemsize)
        if key not in _NUMPY_TO_DATATYPE:
            raise ValueError(f"Field '{name}' has dtype {base} with no PointField equivalent")
        fields.append(PointField(
            name=name, offset=int(offset), datatype=_NUMPY_TO_DATATYPE[key], count=count))

    if is_dense is None:
        names = flat.dtype.names
        if all(n in names for n in ('x', 'y', 'z')):
            is_dense = bool(np.isfinite(xyz_from_array(flat, np.float32)).all())
        else:
            is_dense = True

    msg = PointCloud2()
    msg.header = header
    msg.height = int(height)
    msg.width = int(width)
    msg.fields = fields
    msg.is_bigendian = sys.byteorder == 'big'
    msg.point_step = flat.dtype.itemsize
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = bool(is_dense)
    # Assigning an array.array('B') takes rclpy's fast path; assigning bytes or
    # a list makes rclpy validate every byte in Python, which is very slow for
    # clouds of several megabytes.
    data = array.array('B')
    data.frombytes(flat.tobytes())
    msg.data = data
    return msg


def to_open3d(xyz, colors=None):
    """Create an open3d.geometry.PointCloud from an (N, 3) array (lazy import)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pcd
