"""Phase 1 experiments on frames saved by ``inspect_bag --save-frames``.

The experiments use the same ``PointCloudPipeline`` as the ROS node, loaded
from the same YAML, so results transfer directly to the live system.

Subcommands::

    info             layout and quality of the saved frames
    pipeline         run the configured pipeline, report per-stage counts/timing
    range-sweep      point retention for several max ranges + range histogram
    voxel-sweep      point count, time and geometric error per voxel size
    outlier-compare  none vs statistical vs radius on the voxelized cloud
    kdtree-bench     KD-tree vs brute-force nearest-neighbour search

Example::

    ros2 run lidar_pointcloud_processing offline_experiments voxel-sweep \\
        --frames ~/data/grass_frames --sizes 0.02 0.05 0.1 0.2 --csv voxel.csv
"""

import argparse
import copy
import csv
from pathlib import Path
import sys
import time

import numpy as np

from ..analysis import aggregate_quality, cloud_quality, format_table
from ..filters import finite_mask, nonzero_mask, radius_outlier_mask, statistical_outlier_mask
from ..pipeline import load_config_yaml, PipelineConfig, PointCloudPipeline
from ..pointcloud_utils import xyz_from_array
from ..spatial_search import benchmark_knn, KDTreeIndex
from ..voxelization import voxel_downsample
from .bag_io import load_frames


def _default_config_path():
    """Package YAML: installed share directory if available, else source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        path = Path(get_package_share_directory('lidar_pointcloud_processing'))
        candidate = path / 'config' / 'pointcloud_processing.yaml'
        if candidate.exists():
            return candidate
    except Exception:  # noqa: BLE001 - ament not available or package not installed
        pass
    candidate = Path(__file__).resolve().parents[2] / 'config' / 'pointcloud_processing.yaml'
    return candidate if candidate.exists() else None


def _load_config(path):
    path = path or _default_config_path()
    if path is None:
        print('No config file found; using built-in defaults', file=sys.stderr)
        return PipelineConfig()
    config, unknown = load_config_yaml(path)
    if unknown:
        print(f'Ignoring unknown config keys: {unknown}', file=sys.stderr)
    print(f'Config: {path}', file=sys.stderr)
    return config


def _valid_xyz(frame):
    xyz = xyz_from_array(frame['cloud'])
    return xyz[finite_mask(xyz) & nonzero_mask(xyz)]


def _prefiltered_xyz(frame, config):
    """xyz after invalid / range / crop filtering (the voxel stage's input)."""
    cfg = copy.deepcopy(config)
    cfg.voxel.enabled = False
    cfg.outlier.method = 'none'
    cfg.output.keep_organized = False
    result = PointCloudPipeline(cfg).process(frame['cloud'], frame['height'], frame['width'])
    return xyz_from_array(result.filtered)


def _print_rows(rows, csv_path=''):
    if not rows:
        return
    keys = list(rows[0].keys())
    widths = {k: max(len(k), *(len(_fmt(r[k])) for r in rows)) for k in keys}
    print('  '.join(k.rjust(widths[k]) for k in keys))
    for row in rows:
        print('  '.join(_fmt(row[k]).rjust(widths[k]) for k in keys))
    if csv_path:
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nWrote {csv_path}')


def _fmt(value):
    if isinstance(value, float):
        return f'{value:,.3f}'
    if isinstance(value, (int, np.integer)):
        return f'{value:,}'
    return str(value)


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_info(args, frames, _config):
    first = frames[0]
    print(f'{len(frames)} frames; first: {first["file"]}, '
          f'{first["height"]} x {first["width"]}, frame_id {first.get("frame_id", "?")}')
    print('fields: ' + ', '.join(
        f'{n}:{first["cloud"].dtype.fields[n][0]}' for n in first['cloud'].dtype.names))
    print()
    print(format_table('Quality', aggregate_quality([cloud_quality(f['cloud']) for f in frames])))
    return 0


def cmd_pipeline(args, frames, config):
    pipeline = PointCloudPipeline(config)
    for warning in config.warnings():
        print(f'WARNING: {warning}', file=sys.stderr)
    rows = []
    for frame in frames:
        stats = pipeline.process(frame['cloud'], frame['height'], frame['width']).stats
        row = {'frame': frame['file']}
        row.update({k: v for k, v in stats.as_dict().items() if k != 'timings_ms'})
        row.update({f'{k}_ms': v for k, v in stats.timings_ms.items()})
        rows.append(row)
    _print_rows(rows, args.csv)
    print('\n' + format_table('Mean over frames', {
        k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'frame'}))
    return 0


def cmd_range_sweep(args, frames, _config):
    valid = [_valid_xyz(f) for f in frames]
    ranges = [np.linalg.norm(v, axis=1) for v in valid]
    total_valid = sum(len(r) for r in ranges)
    rows = []
    for max_range in args.max_ranges:
        kept = sum(int(np.count_nonzero((r >= args.min_range) & (r <= max_range)))
                   for r in ranges)
        rows.append({'max_range_m': float(max_range),
                     'mean_points': kept / len(frames),
                     'fraction_of_valid': kept / max(total_valid, 1)})
    _print_rows(rows, args.csv)

    all_ranges = np.concatenate(ranges)
    edges = np.arange(0.0, max(args.max_ranges) + args.bin_size, args.bin_size)
    counts, _ = np.histogram(all_ranges, bins=edges)
    print(f'\nRange histogram (mean points per frame per {args.bin_size} m bin); '
          f'the fall-off with distance is the LiDAR density problem:')
    peak = max(counts.max(), 1)
    for lo, count in zip(edges[:-1], counts):
        per_frame = count / len(frames)
        bar = '█' * int(40 * count / peak)
        print(f'  {lo:6.1f}-{lo + args.bin_size:<6.1f} {per_frame:>10,.0f}  {bar}')
    return 0


def cmd_voxel_sweep(args, frames, config):
    rows = []
    inputs = [_prefiltered_xyz(f, config) for f in frames]
    rng = np.random.default_rng(0)
    for size in args.sizes:
        counts, times, mean_err, p95_err = [], [], [], []
        for xyz in inputs:
            t0 = time.perf_counter()
            grid = voxel_downsample(xyz, size)
            times.append((time.perf_counter() - t0) * 1e3)
            counts.append(len(grid.centroids))
            # Geometric loss: distance from original points to their nearest
            # voxel centroid (one-sided Chamfer distance), on a sample.
            sample = xyz[rng.choice(len(xyz), size=min(len(xyz), args.error_samples),
                                    replace=False)]
            distances, _ = KDTreeIndex(grid.centroids).query_knn(sample, 1)
            mean_err.append(float(distances.mean()))
            p95_err.append(float(np.percentile(distances, 95)))
        row = {
            'voxel_m': float(size),
            'input_points': float(np.mean([len(x) for x in inputs])),
            'voxel_points': float(np.mean(counts)),
            'reduction': float(np.mean(counts) / max(np.mean([len(x) for x in inputs]), 1)),
            'time_ms': float(np.mean(times)),
            'err_mean_cm': float(np.mean(mean_err)) * 100,
            'err_p95_cm': float(np.mean(p95_err)) * 100,
        }
        if args.compare_open3d:
            row['open3d_points'], row['open3d_ms'] = _open3d_voxel(inputs, size)
        rows.append(row)
    _print_rows(rows, args.csv)
    return 0


def _open3d_voxel(inputs, size):
    from ..pointcloud_utils import to_open3d
    counts, times = [], []
    for xyz in inputs:
        pcd = to_open3d(xyz)
        t0 = time.perf_counter()
        down = pcd.voxel_down_sample(float(size))
        times.append((time.perf_counter() - t0) * 1e3)
        counts.append(len(down.points))
    return float(np.mean(counts)), float(np.mean(times))


def cmd_outlier_compare(args, frames, config):
    rows = []
    for frame in frames:
        xyz = _prefiltered_xyz(frame, config)
        if config.voxel.enabled:
            xyz = voxel_downsample(xyz, config.voxel.voxel_size).centroids
        ranges = np.linalg.norm(xyz, axis=1)
        for method in ('none', 'statistical', 'radius'):
            t0 = time.perf_counter()
            if method == 'statistical':
                keep = statistical_outlier_mask(xyz, args.nb_neighbors, args.std_ratio,
                                                args.backend)
            elif method == 'radius':
                keep = radius_outlier_mask(xyz, args.radius, args.min_points, args.backend)
            else:
                keep = np.ones(len(xyz), dtype=bool)
            elapsed = (time.perf_counter() - t0) * 1e3
            removed = ~keep
            rows.append({
                'frame': frame['file'], 'method': method, 'points': len(xyz),
                'removed': int(removed.sum()),
                'removed_pct': 100.0 * removed.mean() if len(xyz) else 0.0,
                # Distance bias: where are the removed points?
                'removed_median_range_m': float(np.median(ranges[removed]))
                if removed.any() else float('nan'),
                'kept_median_range_m': float(np.median(ranges[keep])) if keep.any() else 0.0,
                'time_ms': elapsed,
            })
    _print_rows(rows, args.csv)
    print('\nIf removed_median_range_m is much larger than kept_median_range_m, the filter '
          'is mostly removing sparse far points rather than noise.')
    return 0


def cmd_kdtree_bench(args, frames, config):
    rows = []
    for frame in frames:
        xyz = _prefiltered_xyz(frame, config)
        if args.voxel_size > 0:
            xyz = voxel_downsample(xyz, args.voxel_size).centroids
        result = benchmark_knn(xyz, n_queries=args.queries, k=args.k, radius=args.radius)
        rows.append({'frame': frame['file'], **result})
    _print_rows(rows, args.csv)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Phase 1 point-cloud experiments on saved frames',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    def add(name, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument('--frames', required=True, help='Directory from inspect_bag --out-dir')
        p.add_argument('--limit', type=int, default=0, help='Use at most N frames')
        p.add_argument('--config', default='', help='Pipeline YAML (default: package config)')
        p.add_argument('--csv', default='', help='Also write results to this CSV file')
        return p

    add('info', 'Layout and quality of saved frames')
    add('pipeline', 'Run the configured pipeline')
    p = add('range-sweep', 'Point retention vs max range')
    p.add_argument('--min-range', type=float, default=0.5)
    p.add_argument('--max-ranges', type=float, nargs='+', default=[10, 20, 30, 50])
    p.add_argument('--bin-size', type=float, default=5.0)
    p = add('voxel-sweep', 'Point count / time / error vs voxel size')
    p.add_argument('--sizes', type=float, nargs='+', default=[0.02, 0.05, 0.1, 0.2])
    p.add_argument('--error-samples', type=int, default=50000)
    p.add_argument('--compare-open3d', action='store_true')
    p = add('outlier-compare', 'Compare outlier filters')
    p.add_argument('--nb-neighbors', type=int, default=20)
    p.add_argument('--std-ratio', type=float, default=2.0)
    p.add_argument('--radius', type=float, default=0.2)
    p.add_argument('--min-points', type=int, default=5)
    p.add_argument('--backend', choices=['scipy', 'open3d'], default='scipy')
    p = add('kdtree-bench', 'KD-tree vs brute force')
    p.add_argument('--queries', type=int, default=1000)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--radius', type=float, default=0.5)
    p.add_argument('--voxel-size', type=float, default=0.0,
                   help='Voxelize first (0 = use the full filtered cloud)')
    return parser.parse_args(argv)


COMMANDS = {
    'info': cmd_info,
    'pipeline': cmd_pipeline,
    'range-sweep': cmd_range_sweep,
    'voxel-sweep': cmd_voxel_sweep,
    'outlier-compare': cmd_outlier_compare,
    'kdtree-bench': cmd_kdtree_bench,
}


def main(argv=None):
    args = parse_args(argv)
    frames = load_frames(args.frames, args.limit)
    config = _load_config(args.config)
    return COMMANDS[args.command](args, frames, config)


if __name__ == '__main__':
    sys.exit(main())
