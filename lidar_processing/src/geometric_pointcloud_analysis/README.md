# geometric_pointcloud_analysis

Phase 2 of *3D Semantic World Reconstruction for Autonomous Mobile Robots*.
This package turns the preprocessed LiDAR cloud from Phase 1 into structured
geometry:

* ground vs non-ground points
* object clusters
* surface normals and curvature
* bounding boxes
* shape descriptors
* height above the ground

It does this without any semantic labels.

```text
/lidar/points/voxelized (Phase 1)
        │
        ▼
 ground ── constrained RANSAC plane ──[grid: plane per cell]──► /geometry/ground
        │                                                        /geometry/non_ground
        ▼ non-ground
 Euclidean clustering ──► /geometry/clusters (cluster_id, rgb)
        │
        ▼ per cluster
 centroid · extent · box (upright | pca | aabb) · shape features · height above ground
        │
 normals + curvature (local PCA) ──► /geometry/normals
        │
        └──► /geometry/markers (boxes, labels, ground plane / cells, normals)
             /diagnostics      (counts, slope, roughness, per-stage timing)
```

## Design decisions

This section notes where the package deliberately differs from, or adds to,
the Phase 2 plan.

* **Input is `/lidar/points/voxelized`.** Phase 1 publishes
  `/lidar/points/filtered` and `/lidar/points/voxelized`; there is no
  `/lidar/points/processed`. The topic is a parameter, and the node also
  accepts organized clouds containing NaN rows.
* **The angle constraint acts inside RANSAC, not afterwards.** Hypotheses
  tilted more than `ground.max_angle_deg` from "up" are rejected while
  sampling. Checking only the winning plane is weaker: when a wall has more
  support than the floor, the wall wins and the floor is never considered.
  `test_angle_constraint_rejects_dominant_wall` demonstrates this.
* **"Up" can come from TF.** Set `ground.reference_frame` (e.g. `base_link`),
  and `ground.reference_normal` is rotated into the cloud frame on every
  frame. The node does not assume the sensor's z axis is vertical.
* **There is a second ground method, `grid`.** It fits one constrained plane
  per ground cell, validated against the global plane for both tilt and
  height offset. The Phase 2 plan predicts that a single plane fails on
  undulating grass. On a synthetic rolling scene it does:
  | method | ground recall |
  |---|---|
  | `ransac` | 22 % |
  | `grid` | 75 % |

  The missed ground reappears as fake 20 m "clusters".
  `offline_geometry ground-compare` measures the same effect on your data
  without needing ground truth.
* **Cluster shape comes from local features.** Linearity, planarity,
  scattering and verticality are averaged over the points' normal
  neighbourhoods. The covariance of the *whole* cluster measures its
  outline instead: a 6 × 2 m wall would read as "linear".
* **Upright boxes are the default.** The vertical axis is "up", and the yaw
  is the minimum-area rectangle (rotating calipers). For a rotated 4 × 1 × 1 m
  object this recovers the exact box. PCA boxes can tilt, and AABB inflates
  rotated objects: 7.5 m³ and 10.8 m³ for the same object.
* **Normals are computed on non-ground points by default.** The KD-tree
  query costs about 5 µs per point per core and dominates the stage. Ground
  normals are nearly the plane normal. Set `normals.target: all` for terrain
  work, or `normals.max_points` to query a random subset. Neighbours are
  still searched in the full cloud, so accuracy is unchanged.
* **Eigen-decomposition is closed-form.** A vectorized 3 × 3 symmetric solver
  is 3× faster than batched `np.linalg.eigh`. It falls back to `eigh` for
  degenerate neighbourhoods, and matches `eigh` to 1e-14 on eigenvalues.
* **Surface reconstruction is offline only.** Poisson, ball pivoting and
  alpha shapes are far too slow for 10 Hz, and single sparse scans make
  poor meshes. `offline_geometry surface` meshes a frame with Open3D.
* **Phase 1 code is reused.** This package depends on
  `lidar_pointcloud_processing` for PointCloud2 conversion, configuration
  coercion, saved-frame loading and preprocessing. The offline tool runs the
  real Phase 1 pipeline before Phase 2.

## Installation

Phase 1 must be in the same workspace.

```bash
cd ~/ros2_ws/src   # contains lidar_pointcloud_processing/ and geometric_pointcloud_analysis/
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install --packages-select lidar_pointcloud_processing geometric_pointcloud_analysis
source install/setup.bash
```

Open3D is optional. It is only needed for `offline_geometry surface` and
`offline_geometry view` (`pip install open3d`).

## Running with the bag

```bash
# Terminal 1
ros2 bag play ~/data/grass.db3 --clock --remap /tf_static:=/tf_static_raw

# Terminal 2: Phase 1 + Phase 2 (+ RViz)
ros2 launch geometric_pointcloud_analysis geometric_analysis.launch.py \
    rviz:=true fixed_frame:=<cloud frame_id>
```

Launch arguments:

| Argument | Meaning |
|---|---|
| `with_phase1` (default `true`) | also start the Phase 1 node |
| `config_file` | Phase 2 YAML |
| `phase1_config_file` | Phase 1 YAML |
| `use_sim_time` | use the bag's `/clock` |
| `rviz`, `fixed_frame` | start RViz2 with this fixed frame |
| `log_level` | node log level |

Useful at runtime:

```bash
ros2 param set /geometric_analysis ground.method grid
ros2 param set /geometric_analysis clustering.tolerance 0.35
ros2 param set /geometric_analysis log_clusters true      # print every cluster
ros2 param set /geometric_analysis markers.normals_max 1000
ros2 topic echo /diagnostics --field status[0].values
```

Invalid values are rejected with a reason, and the running pipeline is left
unchanged. Topics and QoS settings are fixed at startup.

## Topics

| Topic | Type | Content |
|---|---|---|
| `/lidar/points/voxelized` (in) | `PointCloud2` | Phase 1 output (any cloud with x, y, z) |
| `/geometry/ground` | `PointCloud2` | ground points, all input fields preserved |
| `/geometry/non_ground` | `PointCloud2` | everything else, all input fields preserved |
| `/geometry/clusters` | `PointCloud2` | `x y z cluster_id rgb` (colour by cluster in RViz) |
| `/geometry/normals` | `PointCloud2` | `x y z normal_x normal_y normal_z curvature` |
| `/geometry/markers` | `MarkerArray` | boxes, labels (id, shape, size), ground plane or grid cells (green accepted, red rejected), normals |
| `/diagnostics` | `DiagnosticArray` | point counts, ground slope, plane RMS, roughness, sensor height, cluster counts, per-stage timing; WARN if ground not found or over budget |

Cluster ids are sorted by size, so id 0 is the largest cluster. All output
is in the input cloud's frame and keeps its header.

## Parameters

Every parameter is documented in `config/geometric_analysis.yaml`. The main
ones:

| Parameter | Default | Notes |
|---|---|---|
| `ground.method` | `ransac` | `grid` for uneven terrain |
| `ground.reference_normal` / `reference_frame` | `[0,0,1]` / `""` | "up", optionally via TF |
| `ground.max_angle_deg` | 20 | plane tilt limit during sampling |
| `ransac.distance_threshold` | 0.08 | inlier and ground classification band (m) |
| `ransac.score_sample_size` | 20000 | score hypotheses on a subset for speed |
| `ransac.seed` | -1 | set ≥ 0 for reproducible experiments |
| `grid.cell_size` / `max_tilt_deg` / `max_height_offset` | 5.0 / 25 / 0.4 | per-cell plane validation |
| `clustering.tolerance` | 0.25 | keep ≥ 1.5 × Phase 1 voxel size |
| `clustering.min_points` / `max_points` | 20 / 50000 | `max_points: 0` = unlimited |
| `clustering.min_height_above_ground` | off | e.g. 0.2 drops low clutter |
| `normals.neighbors` / `radius` | 30 / 0 | `radius > 0` = hybrid kNN + radius |
| `normals.target` / `max_points` | `non_ground` / 0 | cost control, see design notes |
| `boxes.type` | `upright` | `pca`, `aabb` |
| `process_every_n` | 1 | skip frames if over budget |

## Phase 2 experiments

Save frames once with the Phase 1 tool. Then every experiment runs Phase 1
preprocessing followed by Phase 2 analysis, using the package YAMLs unless
you pass `--config` or `--phase1-config`. Add `--csv file.csv` to any
command.

```bash
ros2 run lidar_pointcloud_processing inspect_bag ~/data/grass.db3 \
    --save-frames 20 --out-dir ~/data/grass_frames

G="ros2 run geometric_pointcloud_analysis offline_geometry"
F=--frames=$HOME/data/grass_frames

$G run $F                 # per-frame ground %, slope, roughness, clusters, stage timing
$G clusters $F --index 0  # every cluster: size, height, shape features
$G ransac-sweep $F        # threshold 3–20 cm: inliers, RMS, tilt, repeatability over seeds
$G ground-compare $F      # ransac vs grid, ground % per range band 0-10/10-20/20-30/30+ m
$G cluster-sweep $F       # tolerance 0.10–0.60 m: fragmentation vs merging
$G normals-k $F           # k = 10/20/30/50: time, curvature, stability, ground agreement
$G boxes $F               # AABB vs PCA vs upright tightness
$G surface $F --method poisson --out frame0.ply   # Open3D
$G view $F --index 0      # Open3D: ground green, clusters coloured, boxes
```

What to look for:

* **`ground-compare`.** On flat ground, the ground percentage per range band
  is roughly constant. If it falls with range for `ransac` but not for
  `grid`, a single plane cannot follow the terrain.
* **`clusters`.** Wide (> 10 m), flat clusters with verticality ≈ 0 and
  height ≈ 0 are ground that the ground stage missed, not objects.
* **`ransac-sweep`.** A jump in inlier percentage as the threshold grows
  means grass or low obstacles are being absorbed into the ground.
  `normal_spread_deg` shows how repeatable RANSAC is across random seeds.
* **`cluster-sweep`.** Too small a tolerance produces many clusters and many
  rejected small points. Too large a tolerance produces a few giant merged
  clusters.
* **`normals-k`.** Small k gives noisy normals, visible as lower ground
  agreement. Large k is slower and blurs edges. Local shape labels also
  depend on k, because dimensionality depends on scale.

## Performance

These numbers are from one CPU core in a sandbox, on a ~100k-point synthetic
scene.

| Stage | Time |
|---|---|
| ground: `ransac` | 20 ms |
| ground: `grid` | 85 ms |
| clustering | 10 ms |
| normals: `non_ground` (default) | 77 ms |
| normals: all points | ~790 ms |

The KD-tree queries use all cores (`workers=-1`), so a multi-core machine is
several times faster on the normals stage. If `/diagnostics` warns about the
100 ms budget, use `normals.max_points`, `normals.neighbors: 20` or
`process_every_n: 2`, or disable publishing unused outputs.

## Tests

```bash
cd ~/ros2_ws/src/geometric_pointcloud_analysis
python3 -m pytest test -q    # runs with or without ROS; needs Phase 1 as a sibling or installed
```

The tests use synthetic outdoor scenes with ground truth: flat, 8° slope and
rolling terrain, with a crate, pole, bush and wall. They cover:

* RANSAC accuracy, the angle constraint, degenerate input and adaptive iterations
* grid vs single plane on rolling terrain
* cluster separation, chaining and size filters
* normals on planes and spheres, orientation, and radius limits
* box containment, tightness and quaternions
* shape features
* per-object cluster purity
* the TF-style "up" override
* config validation and YAML ↔ parameter consistency
* markers and diagnostics

## Package layout

```text
geometric_pointcloud_analysis/
├── geometric_pointcloud_analysis/
│   ├── geometric_analysis_node.py  ROS node: I/O, parameters, TF "up", publishing
│   ├── pipeline.py                 config dataclasses + GeometryPipeline (no ROS)
│   ├── ransac.py                   constrained, adaptive, batched RANSAC; LSQ refinement
│   ├── ground.py                   single-plane and grid ground segmentation, heights
│   ├── clustering.py               Euclidean clustering (radius graph components)
│   ├── normals.py                  local-PCA normals, curvature, closed-form 3x3 eigensolver
│   ├── bounding_boxes.py           AABB, PCA OBB, upright min-area OBB, quaternions
│   ├── surface_analysis.py         shape features, ground stats, offline meshing
│   ├── markers.py                  RViz MarkerArray, cluster colours
│   ├── diagnostics.py              rolling timings, DiagnosticArray
│   └── tools/offline_geometry.py   Phase 2 experiments
├── config/geometric_analysis.yaml
├── launch/geometric_analysis.launch.py
├── rviz/geometric_analysis.rviz
└── test/
```

## What Phase 3 consumes

Phase 3 (registration) can use:

* `/geometry/non_ground`, or ground-removed clouds, which give ICP stable
  structure instead of featureless ground
* `normals.estimate_normals`, for point-to-plane ICP
* `ransac.fit_plane_ransac`, for ground-plane constraints

Later phases use the cluster boxes and height above ground: semantic fusion
(Phase 6) and terrain reasoning (Phase 11).
