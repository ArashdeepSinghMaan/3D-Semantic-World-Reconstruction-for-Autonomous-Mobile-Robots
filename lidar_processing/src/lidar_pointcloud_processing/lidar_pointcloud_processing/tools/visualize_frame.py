"""Visualize a saved frame before/after processing with Open3D.

Views::

    raw        all valid input points, grey
    removed    kept points grey, points removed by the filters red
    filtered   full-resolution filtered cloud, coloured by height
    voxelized  voxelized cloud, coloured by height
    compare    filtered (left) and voxelized (right) side by side

Example::

    ros2 run lidar_pointcloud_processing visualize_frame \\
        --frames ~/data/grass_frames --index 0 --view removed
"""

import argparse
import sys

import numpy as np

from ..filters import finite_mask
from ..pipeline import PointCloudPipeline
from ..pointcloud_utils import to_open3d, xyz_from_array
from .bag_io import load_frames
from .offline_experiments import _load_config


def height_colors(xyz):
    """Map z to a blue -> green -> red ramp using the 2nd-98th percentile."""
    if len(xyz) == 0:
        return np.zeros((0, 3))
    lo, hi = np.percentile(xyz[:, 2], [2, 98])
    t = np.clip((xyz[:, 2] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.column_stack([np.clip(2 * t - 1, 0, 1), 1 - np.abs(2 * t - 1),
                            np.clip(1 - 2 * t, 0, 1)])


def build_geometries(frame, config, view, offset=None):
    result = PointCloudPipeline(config).process(frame['cloud'], frame['height'], frame['width'])
    xyz_in = xyz_from_array(frame['cloud'])
    finite = finite_mask(xyz_in)

    if view == 'raw':
        xyz = xyz_in[finite]
        return [to_open3d(xyz, np.full((len(xyz), 3), 0.6))]
    if view == 'removed':
        kept = xyz_in[result.keep_mask]
        removed = xyz_in[finite & ~result.keep_mask]
        return [to_open3d(kept, np.full((len(kept), 3), 0.6)),
                to_open3d(removed, np.tile([0.9, 0.1, 0.1], (len(removed), 1)))]

    filtered = xyz_from_array(result.filtered)
    filtered = filtered[finite_mask(filtered)]
    if view == 'filtered':
        return [to_open3d(filtered, height_colors(filtered))]
    if result.voxelized is None:
        raise SystemExit('voxel.enabled is false in the config; nothing to show')
    voxel = xyz_from_array(result.voxelized)
    if view == 'voxelized':
        return [to_open3d(voxel, height_colors(voxel))]

    # compare: shift the voxelized cloud sideways by the scene width.
    if offset is None:
        extent = filtered[:, 1].max() - filtered[:, 1].min() if len(filtered) else 10.0
        offset = float(extent) * 1.1
    shifted = voxel + np.array([0.0, offset, 0.0])
    return [to_open3d(filtered, height_colors(filtered)),
            to_open3d(shifted, height_colors(voxel))]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Visualize a saved frame with Open3D')
    parser.add_argument('--frames', required=True)
    parser.add_argument('--index', type=int, default=0, help='Position in the frames list')
    parser.add_argument('--config', default='')
    parser.add_argument('--view', default='compare',
                        choices=['raw', 'removed', 'filtered', 'voxelized', 'compare'])
    parser.add_argument('--point-size', type=float, default=2.0)
    args = parser.parse_args(argv)

    try:
        import open3d as o3d
    except ImportError:
        print('Open3D is required: pip install open3d', file=sys.stderr)
        return 1

    frames = load_frames(args.frames)
    frame = frames[args.index]
    geometries = build_geometries(frame, _load_config(args.config), args.view)
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name=f'{frame["file"]} - {args.view}')
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    visualizer.get_render_option().point_size = args.point_size
    visualizer.run()
    visualizer.destroy_window()
    return 0


if __name__ == '__main__':
    sys.exit(main())
