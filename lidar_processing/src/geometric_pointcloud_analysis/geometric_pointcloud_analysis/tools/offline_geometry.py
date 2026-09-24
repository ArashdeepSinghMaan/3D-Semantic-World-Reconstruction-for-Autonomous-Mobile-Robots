"""Phase 2 experiments on frames saved by Phase 1's ``inspect_bag --save-frames``.

Each frame is first preprocessed with the Phase 1 pipeline (Phase 1 YAML),
exactly as the live system does, then analysed with the Phase 2 pipeline
(Phase 2 YAML).

Subcommands::

    run             per-frame ground / cluster / timing summary
    clusters        describe every cluster of one frame
    ransac-sweep    distance threshold vs inliers, RMS, tilt, repeatability over seeds
    ground-compare  ransac vs grid: ground fraction per range band, cells, time
    cluster-sweep   tolerance vs number / size of clusters, time
    normals-k       neighbour count vs time, curvature, stability, ground agreement
    boxes           AABB vs PCA vs upright box volume
    surface         mesh one frame with Open3D (poisson | ball_pivoting | alpha)
    view            Open3D viewer: ground, clusters, boxes

Example::

    ros2 run geometric_pointcloud_analysis offline_geometry ground-compare \\
        --frames ~/data/grass_frames --csv ground.csv
"""

import argparse
import copy
import csv
from pathlib import Path
import sys
import time

from lidar_pointcloud_processing.pipeline import (
    load_config_yaml as load_phase1_yaml,
    PipelineConfig,
    PointCloudPipeline,
)
from lidar_pointcloud_processing.pointcloud_utils import xyz_from_array
from lidar_pointcloud_processing.tools.bag_io import load_frames
import numpy as np

from ..bounding_boxes import compute_box
from ..clustering import euclidean_clusters
from ..normals import estimate_normals
from ..pipeline import GeometryConfig, GeometryPipeline, load_config_yaml
from ..ransac import fit_plane_ransac, normalize


# ─── Configuration and data ─────────────────────────────────────────────────

def _share_or_source(package, relative):
    try:
        from ament_index_python.packages import get_package_share_directory
        candidate = Path(get_package_share_directory(package)) / relative
        if candidate.exists():
            return candidate
    except Exception:  # noqa: BLE001 - not installed / ament not available
        pass
    source_root = Path(__file__).resolve().parents[3]
    candidate = source_root / package / relative
    return candidate if candidate.exists() else None


def load_configs(args):
    phase1_path = args.phase1_config or _share_or_source(
        'lidar_pointcloud_processing', 'config/pointcloud_processing.yaml')
    phase2_path = args.config or _share_or_source(
        'geometric_pointcloud_analysis', 'config/geometric_analysis.yaml')
    phase1 = load_phase1_yaml(phase1_path)[0] if phase1_path else PipelineConfig()
    if phase2_path:
        phase2, unknown = load_config_yaml(phase2_path)
        unknown = [k for k in unknown if not k.startswith('markers')]
        if unknown:
            print(f'Ignoring unknown geometry keys: {unknown}', file=sys.stderr)
    else:
        phase2 = GeometryConfig()
    print(f'Phase 1 config: {phase1_path or "defaults"}\n'
          f'Phase 2 config: {phase2_path or "defaults"}', file=sys.stderr)
    return phase1, phase2


def preprocessed_frames(args, phase1):
    """Yield (name, xyz) after Phase 1 preprocessing (voxelized if enabled)."""
    pipeline = PointCloudPipeline(phase1)
    for frame in load_frames(args.frames, args.limit):
        result = pipeline.process(frame['cloud'], frame['height'], frame['width'])
        cloud = result.voxelized if result.voxelized is not None else result.filtered
        xyz = xyz_from_array(cloud)
        yield frame['file'], xyz[np.isfinite(xyz).all(axis=1)]


def with_overrides(config, **sections):
    config = copy.deepcopy(config)
    for section, values in sections.items():
        for key, value in values.items():
            setattr(getattr(config, section), key, value)
    return config.validate()


def print_rows(rows, csv_path=''):
    if not rows:
        print('(no rows)')
        return
    keys = list(rows[0])
    text = [[_fmt(r.get(k, '')) for k in keys] for r in rows]
    widths = [max(len(k), *(len(t[i]) for t in text)) for i, k in enumerate(keys)]
    print('  '.join(k.rjust(w) for k, w in zip(keys, widths)))
    for t in text:
        print('  '.join(v.rjust(w) for v, w in zip(t, widths)))
    if csv_path:
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nWrote {csv_path}')


def _fmt(value):
    if isinstance(value, (float, np.floating)):
        return f'{value:,.3f}'
    if isinstance(value, (int, np.integer)):
        return f'{value:,}'
    return str(value)


def _timed(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1e3


# ─── Commands ───────────────────────────────────────────────────────────────

def cmd_run(args, phase1, phase2):
    pipeline = GeometryPipeline(phase2)
    rows = []
    for name, xyz in preprocessed_frames(args, phase1):
        r = pipeline.process(xyz)
        row = {'frame': name}
        row.update({k: v for k, v in r.stats.items()
                    if k not in ('cluster_pairs',)})
        row.update({f'{k}_ms': v for k, v in r.timings_ms.items()})
        row['total_ms'] = r.total_ms
        rows.append(row)
    print_rows(rows, args.csv)
    return 0


def cmd_clusters(args, phase1, phase2):
    frames = list(preprocessed_frames(args, phase1))
    name, xyz = frames[min(args.index, len(frames) - 1)]
    r = GeometryPipeline(phase2).process(xyz)
    print(f'{name}: {len(xyz):,} points, {r.stats.get("ground_points", 0):,} ground, '
          f'{len(r.clusters)} clusters')
    rows = []
    for c in r.clusters:
        dims = c.box.extent if c.box is not None else c.dimensions
        rows.append({'id': c.id, 'points': c.point_count,
                     'cx': c.centroid[0], 'cy': c.centroid[1], 'cz': c.centroid[2],
                     'size_1': dims[0], 'size_2': dims[1], 'size_3': dims[2],
                     'height_max': c.height_max, 'shape': c.shape,
                     **{k: v for k, v in c.features.items()},
                     'curvature': c.mean_curvature})
    print_rows(rows, args.csv)
    return 0


def cmd_ransac_sweep(args, phase1, phase2):
    g, r = phase2.ground, phase2.ransac
    up = normalize(g.reference_normal)
    rows = []
    for name, xyz in preprocessed_frames(args, phase1):
        for threshold in args.thresholds:
            normals, fractions, rms, iters, times = [], [], [], [], []
            for seed in range(args.seeds):
                plane, ms = _timed(
                    fit_plane_ransac, xyz, threshold, r.max_iterations, r.probability,
                    reference_normal=up, max_angle_deg=g.max_angle_deg, refine=r.refine,
                    score_sample_size=r.score_sample_size, min_inliers=g.min_inliers,
                    rng=seed)
                times.append(ms)
                if plane is None:
                    continue
                normals.append(plane.normal)
                fractions.append(plane.inlier_count / len(xyz))
                rms.append(plane.rms)
                iters.append(plane.iterations)
            if not normals:
                rows.append({'frame': name, 'threshold_m': threshold, 'found': 0})
                continue
            normals = np.array(normals)
            spread = np.degrees(np.arccos(np.clip(np.abs(normals @ normals.T), -1, 1))).max()
            rows.append({
                'frame': name, 'threshold_m': threshold, 'found': len(normals),
                'inlier_pct': 100 * float(np.mean(fractions)),
                'rms_cm': 100 * float(np.mean(rms)),
                'tilt_deg': float(np.degrees(np.arccos(np.clip(abs(normals.mean(0) @ up)
                                                               / np.linalg.norm(
                                                                   normals.mean(0)), 0, 1)))),
                'normal_spread_deg': float(spread),
                'iterations': float(np.mean(iters)),
                'time_ms': float(np.mean(times)),
            })
    print_rows(rows, args.csv)
    print('\nnormal_spread_deg: largest angle between planes found with different random '
          'seeds (repeatability).\nA large jump in inlier_pct as the threshold grows means '
          'low obstacles or vegetation are being absorbed into the ground.')
    return 0


def cmd_ground_compare(args, phase1, phase2):
    bands = list(zip(args.range_bands[:-1], args.range_bands[1:]))
    rows = []
    for name, xyz in preprocessed_frames(args, phase1):
        ranges = np.linalg.norm(xyz[:, :2], axis=1)
        for method in ('ransac', 'grid'):
            config = with_overrides(phase2, ground={'method': method},
                                    clustering={'enabled': False},
                                    normals={'enabled': False})
            r = GeometryPipeline(config).process(xyz)
            mask = r.ground_mask
            row = {'frame': name, 'method': method, 'ground_pct': 100 * mask.mean(),
                   'time_ms': r.timings_ms.get('ground', 0.0)}
            for lo, hi in bands:
                band = (ranges >= lo) & (ranges < hi)
                row[f'ground_pct_{lo:g}-{hi:g}m'] = (100 * mask[band].mean()
                                                     if band.any() else float('nan'))
            if r.ground is not None and r.ground.cells:
                row['cells_accepted'] = (f'{sum(c.accepted for c in r.ground.cells)}/'
                                         f'{len(r.ground.cells)}')
            rows.append(row)
    print_rows(rows, args.csv)
    print('\nOn flat ground the per-band ground percentage stays roughly constant. If it '
          'falls with range for\nransac but not for grid, a single plane cannot follow the '
          'terrain (slopes, undulating grass).')
    return 0


def cmd_cluster_sweep(args, phase1, phase2):
    rows = []
    for name, xyz in preprocessed_frames(args, phase1):
        config = with_overrides(phase2, clustering={'enabled': False},
                                normals={'enabled': False})
        r = GeometryPipeline(config).process(xyz)
        points = xyz[~r.ground_mask]
        c = phase2.clustering
        for tolerance in args.tolerances:
            result, ms = _timed(euclidean_clusters, points, tolerance, c.min_points,
                                c.max_points)
            sizes = result.sizes
            rows.append({
                'frame': name, 'tolerance_m': tolerance, 'clusters': len(sizes),
                'largest': int(sizes.max()) if len(sizes) else 0,
                'median': float(np.median(sizes)) if len(sizes) else 0.0,
                'clustered_pct': 100 * float(sizes.sum()) / max(len(points), 1),
                'rejected_small_pts': result.rejected_small,
                'rejected_large_pts': result.rejected_large,
                'pairs': result.pair_count, 'time_ms': ms,
            })
    print_rows(rows, args.csv)
    print('\nToo small a tolerance fragments objects (many clusters, many rejected small '
          'points);\ntoo large merges neighbouring objects (few, very large clusters).')
    return 0


def cmd_normals_k(args, phase1, phase2):
    n = phase2.normals
    rows = []
    for name, xyz in preprocessed_frames(args, phase1):
        config = with_overrides(phase2, clustering={'enabled': False},
                                normals={'enabled': False})
        ground = GeometryPipeline(config).process(xyz).ground
        results = {}
        for k in args.ks:
            results[k], ms = _timed(estimate_normals, xyz, k, n.radius, n.min_neighbors,
                                    n.orientation, n.viewpoint)
            results[k].time_ms = ms
        reference = results[max(args.ks)]
        for k in args.ks:
            res = results[k]
            both = res.valid & reference.valid
            angle = np.degrees(np.arccos(np.clip(np.abs(np.einsum(
                'ij,ij->i', res.normals[both], reference.normals[both])), 0, 1)))
            row = {'frame': name, 'k': k, 'time_ms': res.time_ms,
                   'valid_pct': 100 * res.valid.mean(),
                   'mean_curvature': float(np.nanmean(res.curvature)),
                   f'angle_to_k{max(args.ks)}_deg': float(np.mean(angle)) if both.any() else 0}
            if ground is not None and ground.plane is not None and ground.mask.any():
                gm = ground.mask & res.valid
                cosines = np.abs(res.normals[gm] @ ground.plane.normal)
                row['ground_normals_within_15deg_pct'] = 100 * float(
                    np.mean(cosines >= np.cos(np.radians(15))))
            rows.append(row)
    print_rows(rows, args.csv)
    print('\nSmall k: noisy normals (low agreement on the ground). Large k: smoother but '
          'slower,\nand normals blur across edges (curvature rises at object borders).')
    return 0


def cmd_boxes(args, phase1, phase2):
    rows = []
    for name, xyz in preprocessed_frames(args, phase1):
        config = with_overrides(phase2, boxes={'enabled': False}, normals={'enabled': False})
        r = GeometryPipeline(config).process(xyz)
        volumes = {kind: [] for kind in ('aabb', 'pca', 'upright')}
        times = {kind: 0.0 for kind in volumes}
        for c in r.clusters:
            points = xyz[r.labels == c.id]
            for kind in volumes:
                box, ms = _timed(compute_box, points, kind, r.up)
                volumes[kind].append(box.volume)
                times[kind] += ms
        if not r.clusters:
            continue
        upright = np.array(volumes['upright'])
        for kind, values in volumes.items():
            rows.append({'frame': name, 'type': kind, 'clusters': len(values),
                         'mean_volume_m3': float(np.mean(values)),
                         'volume_vs_upright': float(np.mean(np.array(values) /
                                                            np.maximum(upright, 1e-9))),
                         'time_ms_total': times[kind]})
    print_rows(rows, args.csv)
    print('\nSmaller volume = tighter box. AABB inflates rotated objects; PCA can tilt; '
          'upright keeps the box vertical.')
    return 0


def cmd_surface(args, phase1, phase2):
    from ..surface_analysis import reconstruct_surface
    frames = list(preprocessed_frames(args, phase1))
    name, xyz = frames[min(args.index, len(frames) - 1)]
    normals = None
    if args.method != 'alpha':
        n = phase2.normals
        normals = estimate_normals(xyz, n.neighbors, n.radius, n.min_neighbors,
                                   n.orientation, n.viewpoint).normals
    mesh, stats = reconstruct_surface(xyz, args.method, normals, poisson_depth=args.depth,
                                      ball_radii=tuple(args.radii), alpha=args.alpha)
    import open3d as o3d
    o3d.io.write_triangle_mesh(args.out, mesh)
    print(f'{name}: {stats}\nWrote {args.out}')
    return 0


def cmd_view(args, phase1, phase2):
    import open3d as o3d
    from ..bounding_boxes import EDGES
    from ..markers import label_colors

    frames = list(preprocessed_frames(args, phase1))
    name, xyz = frames[min(args.index, len(frames) - 1)]
    r = GeometryPipeline(phase2).process(xyz)
    colors = label_colors(r.labels, unlabeled=(0.55, 0.55, 0.55))
    colors[r.ground_mask] = (0.25, 0.75, 0.35)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)]
    for c in r.clusters:
        if c.box is None:
            continue
        lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector(c.box.corners()),
                                     o3d.utility.Vector2iVector(list(EDGES)))
        lines.paint_uniform_color((1.0, 1.0, 1.0))
        geometries.append(lines)
    print(f'{name}: {len(r.clusters)} clusters, {r.stats.get("ground_points", 0):,} ground '
          f'points (green)')
    o3d.visualization.draw_geometries(geometries, window_name=f'{name} - geometry')
    return 0


COMMANDS = {
    'run': cmd_run, 'clusters': cmd_clusters, 'ransac-sweep': cmd_ransac_sweep,
    'ground-compare': cmd_ground_compare, 'cluster-sweep': cmd_cluster_sweep,
    'normals-k': cmd_normals_k, 'boxes': cmd_boxes, 'surface': cmd_surface, 'view': cmd_view,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Phase 2 geometric experiments on saved frames',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    def add(name, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument('--frames', required=True, help='Directory from Phase 1 inspect_bag')
        p.add_argument('--limit', type=int, default=0)
        p.add_argument('--config', default='', help='Phase 2 YAML (default: package config)')
        p.add_argument('--phase1-config', default='', help='Phase 1 YAML (default: package)')
        p.add_argument('--csv', default='')
        return p

    add('run', 'Per-frame summary')
    p = add('clusters', 'Describe clusters of one frame')
    p.add_argument('--index', type=int, default=0)
    p = add('ransac-sweep', 'RANSAC distance threshold sweep')
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.03, 0.05, 0.08, 0.12, 0.2])
    p.add_argument('--seeds', type=int, default=5)
    p = add('ground-compare', 'Single plane vs grid')
    p.add_argument('--range-bands', type=float, nargs='+', default=[0, 10, 20, 30, 100])
    p = add('cluster-sweep', 'Clustering tolerance sweep')
    p.add_argument('--tolerances', type=float, nargs='+', default=[0.1, 0.15, 0.25, 0.4, 0.6])
    p = add('normals-k', 'Normal neighbourhood size sweep')
    p.add_argument('--ks', type=int, nargs='+', default=[10, 20, 30, 50])
    add('boxes', 'Bounding box types')
    p = add('surface', 'Mesh one frame with Open3D')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--method', choices=['poisson', 'ball_pivoting', 'alpha'], default='poisson')
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--radii', type=float, nargs='+', default=[0.1, 0.2, 0.4])
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--out', default='surface.ply')
    p = add('view', 'Open3D viewer')
    p.add_argument('--index', type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    phase1, phase2 = load_configs(args)
    return COMMANDS[args.command](args, phase1, phase2)


if __name__ == '__main__':
    sys.exit(main())
