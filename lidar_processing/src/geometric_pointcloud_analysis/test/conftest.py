"""Synthetic outdoor scenes with ground truth, and ROS message stand-ins."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running the tests from a source checkout next to the Phase 1 package.
_PHASE1 = Path(__file__).resolve().parents[2] / 'lidar_pointcloud_processing'
if _PHASE1.exists() and str(_PHASE1) not in sys.path:
    sys.path.insert(0, str(_PHASE1))
_PHASE1_CONFTEST = _PHASE1 / 'test' / 'conftest.py'
if _PHASE1_CONFTEST.exists():
    # Installs sensor_msgs / diagnostic_msgs stand-ins when ROS is not available.
    import importlib.util
    _spec = importlib.util.spec_from_file_location('phase1_conftest', _PHASE1_CONFTEST)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)

SENSOR_HEIGHT = 0.5


def ground_height(x, y, undulation=0.0, slope_deg=0.0):
    """Terrain height (sensor frame) at (x, y)."""
    return (-SENSOR_HEIGHT + np.tan(np.radians(slope_deg)) * x
            + undulation * np.sin(x / 6.0) * np.cos(y / 7.0))


def make_scene(undulation=0.0, slope_deg=0.0, seed=0, ground_points=20000, noise=0.01):
    """Return (xyz, truth) where truth is an int array: 0 ground, 1.. object ids."""
    rng = np.random.default_rng(seed)
    parts, labels = [], []

    r = np.sqrt(rng.uniform(1.0, 30.0 ** 2, ground_points))
    a = rng.uniform(0, 2 * np.pi, ground_points)
    gx, gy = r * np.cos(a), r * np.sin(a)
    gz = ground_height(gx, gy, undulation, slope_deg) + rng.normal(0, noise, ground_points)
    parts.append(np.column_stack([gx, gy, gz]))
    labels.append(np.zeros(ground_points, dtype=int))

    def on_ground(x, y):
        return ground_height(x, y, undulation, slope_deg)

    # 1: crate 1.2 x 0.8 x 0.8 m, rotated 30 degrees, at (6, 2)
    n = 1500
    local = rng.uniform([-0.6, -0.4, 0.05], [0.6, 0.4, 0.8], (n, 3))
    face = rng.integers(0, 3, n)
    for axis, (lo, hi) in enumerate([(-0.6, 0.6), (-0.4, 0.4), (0.05, 0.8)]):
        pick = face == axis
        local[pick, axis] = rng.choice([lo, hi], pick.sum())
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    crate = local @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + [6.0, 2.0, 0.0]
    crate[:, 2] += on_ground(6.0, 2.0)
    parts.append(crate)
    labels.append(np.full(n, 1))

    # 2: pole, radius 0.1 m, height 2.5 m, at (-4, 5)
    n = 600
    t = rng.uniform(0, 2 * np.pi, n)
    pole = np.column_stack([-4 + 0.1 * np.cos(t), 5 + 0.1 * np.sin(t), rng.uniform(0.1, 2.5, n)])
    pole[:, 2] += on_ground(-4.0, 5.0)
    parts.append(pole)
    labels.append(np.full(n, 2))

    # 3: bush, scattered ball of radius 0.6 m at (3, -7)
    n = 900
    bush = rng.normal(0, 0.3, (n, 3)) + [3.0, -7.0, 0.8]
    bush[:, 2] += on_ground(3.0, -7.0)
    parts.append(bush)
    labels.append(np.full(n, 3))

    # 4: wall, 6 m long, 2 m high, at x = -10
    n = 3000
    wall = np.column_stack([np.full(n, -10.0) + rng.normal(0, noise, n),
                            rng.uniform(-3, 3, n), rng.uniform(0.1, 2.0, n)])
    wall[:, 2] += on_ground(-10.0, 0.0)
    parts.append(wall)
    labels.append(np.full(n, 4))

    return np.vstack(parts), np.concatenate(labels)


@pytest.fixture
def flat_scene():
    return make_scene()


@pytest.fixture
def rolling_scene():
    return make_scene(undulation=0.6)


def _install_fake_marker_messages():
    """Minimal visualization_msgs / geometry_msgs / std_msgs stand-ins when ROS is absent."""
    try:
        import visualization_msgs.msg  # noqa: F401
        import geometry_msgs.msg  # noqa: F401
        import std_msgs.msg  # noqa: F401
        return
    except ImportError:
        pass
    import types
    from types import SimpleNamespace as NS

    class _Msg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Point(_Msg):
        pass

    class ColorRGBA(_Msg):
        pass

    class Marker:
        ADD, DELETEALL = 0, 3
        CUBE, LINE_LIST, TEXT_VIEW_FACING, TRIANGLE_LIST = 1, 5, 9, 11

        def __init__(self):
            self.header = None
            self.ns, self.id, self.type, self.action, self.text = '', 0, 0, 0, ''
            self.pose = NS(position=NS(x=0.0, y=0.0, z=0.0),
                           orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))
            self.scale = NS(x=0.0, y=0.0, z=0.0)
            self.color = None
            self.points, self.colors = [], []

    class MarkerArray:
        def __init__(self):
            self.markers = []

    stand_ins = (
        ('visualization_msgs', {'Marker': Marker, 'MarkerArray': MarkerArray}),
        ('geometry_msgs', {'Point': Point}),
        ('std_msgs', {'ColorRGBA': ColorRGBA}),
    )
    for package, classes in stand_ins:
        pkg = sys.modules.get(package) or types.ModuleType(package)
        msg = getattr(pkg, 'msg', None) or types.ModuleType(f'{package}.msg')
        for name, cls in classes.items():
            setattr(msg, name, cls)
        pkg.msg = msg
        sys.modules[package] = pkg
        sys.modules[f'{package}.msg'] = msg


_install_fake_marker_messages()
