"""Shape descriptors for clusters and the ground, plus optional offline meshing.

Eigenvalue features (l1 >= l2 >= l3 of a covariance matrix):

* linearity  = (l1 - l2) / l1   -> poles, trunks, wires
* planarity  = (l2 - l3) / l1   -> walls, ground, boards
* scattering = l3 / l1          -> vegetation, noise
* verticality = 1 - |e3 . up|   -> 0 for horizontal surfaces, 1 for vertical ones

These features describe *local* surface structure and are computed per point
from its normal-estimation neighbourhood. A cluster's shape is the mean of its
points' local features. Computing them from the covariance of the whole
cluster instead measures the cluster's outline: a 6 x 2 m wall is elongated,
so it would come out "linear". ``eigen_features`` (whole cluster) is kept as
a fallback when normals are disabled.

These describe geometry without semantics; Phase 5+ can compare them with
semantic labels (e.g. "tree" clusters should be linear or scattered).

Surface reconstruction (Poisson, ball pivoting, alpha shapes) is not run per
frame: it is far too slow for 10 Hz and sparse single scans make poor meshes.
``reconstruct_surface`` is for the offline tool, typically on accumulated or
dense clouds.
"""

import time

import numpy as np


def eigen_features(points, up=(0.0, 0.0, 1.0)):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return {'linearity': np.nan, 'planarity': np.nan, 'scattering': np.nan,
                'verticality': np.nan}
    centered = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    l3, l2, l1 = np.maximum(values, 0.0)
    if l1 <= 0:
        return {'linearity': 0.0, 'planarity': 0.0, 'scattering': 0.0, 'verticality': 0.0}
    up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    return {
        'linearity': float((l1 - l2) / l1),
        'planarity': float((l2 - l3) / l1),
        'scattering': float(l3 / l1),
        'verticality': float(1.0 - abs(vectors[:, 0] @ up)),
    }


def local_features(eigenvalues_ascending, normals, up=(0.0, 0.0, 1.0)):
    """Per-point features from (N, 3) ascending eigenvalues and (N, 3) normals."""
    values = np.maximum(np.asarray(eigenvalues_ascending, dtype=np.float64), 0.0)
    l3, l2, l1 = values[:, 0], values[:, 1], values[:, 2]
    up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    with np.errstate(invalid='ignore', divide='ignore'):
        return {
            'linearity': (l1 - l2) / l1,
            'planarity': (l2 - l3) / l1,
            'scattering': l3 / l1,
            'verticality': 1.0 - np.abs(np.asarray(normals) @ up),
        }


def mean_local_features(per_point, index):
    """Mean of per-point features over ``index`` (NaN-aware)."""
    out = {}
    for name, values in per_point.items():
        selected = values[index]
        out[name] = float(np.nanmean(selected)) if np.isfinite(selected).any() else float('nan')
    return out


def dominant_shape(features):
    """'linear', 'planar' or 'scattered' (largest of the three features)."""
    names = ('linear', 'planar', 'scattered')
    values = [features['linearity'], features['planarity'], features['scattering']]
    if not np.all(np.isfinite(values)):
        return 'unknown'
    return names[int(np.argmax(values))]


def ground_statistics(ground_result, reference_up):
    """Slope of the global ground plane and roughness of the ground points."""
    plane = ground_result.plane
    if plane is None:
        return {}
    up = np.asarray(reference_up, dtype=np.float64) / np.linalg.norm(reference_up)
    stats = {
        'ground_slope_deg': float(np.degrees(np.arccos(np.clip(abs(plane.normal @ up), 0, 1)))),
        'ground_plane_rms_m': plane.rms,
        'ground_sensor_height_m': float(plane.d),  # distance of the origin above the plane
    }
    heights = ground_result.height[ground_result.mask]
    if heights.size:
        stats['ground_roughness_m'] = float(np.std(heights))
    if ground_result.cells:
        accepted = sum(c.accepted for c in ground_result.cells)
        stats['grid_cells'] = len(ground_result.cells)
        stats['grid_cells_accepted'] = accepted
    return stats


def reconstruct_surface(xyz, method='poisson', normals=None, poisson_depth=8,
                        density_quantile=0.05, ball_radii=(0.1, 0.2, 0.4), alpha=0.5):
    """Mesh a point cloud with Open3D (offline use). Returns (mesh, stats)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(xyz, np.float64)))
    start = time.perf_counter()
    if method in ('poisson', 'ball_pivoting'):
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(np.nan_to_num(np.asarray(normals)))
        else:
            pcd.estimate_normals()
            pcd.orient_normals_towards_camera_location(np.zeros(3))
    if method == 'poisson':
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=int(poisson_depth))
        densities = np.asarray(densities)
        if density_quantile > 0 and densities.size:
            # Poisson extrapolates a closed surface; remove poorly supported parts.
            mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_quantile))
    elif method == 'ball_pivoting':
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(list(ball_radii)))
    elif method == 'alpha':
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, float(alpha))
    else:
        raise ValueError(f'Unknown surface method {method!r}')
    mesh.compute_vertex_normals()
    return mesh, {
        'method': method,
        'vertices': len(mesh.vertices),
        'triangles': len(mesh.triangles),
        'time_ms': (time.perf_counter() - start) * 1e3,
    }
