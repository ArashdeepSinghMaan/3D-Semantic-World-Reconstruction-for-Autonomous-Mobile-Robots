# lidar_pointcloud_processing

Phase 1 of *3D Semantic World Reconstruction for Autonomous Mobile Robots*:
a ROS 2 Humble package that turns the raw Ouster stream
(`/os_cloud_node/points`) into a clean, configurable, measurable 3D
representation, plus the tools to inspect the dataset and run the Phase 1
experiments offline.

```text
/os_cloud_node/points (PointCloud2)
        │
        ▼  decode: structured NumPy view of msg.data (all fields, zero-copy)
┌───────────────────────────────────────────────┐
│ invalid: non-finite + Ouster zero points      │
│ range:   min_range ≤ ‖p‖ ≤ max_range          │
│ crop box (self-filter, optional)              │
│ [outlier, if apply_to: filtered]              │──► /lidar/points/filtered
│                                               │    full resolution, ALL fields
│ voxel grid: centroid + averaged fields        │    (t, ring, reflectivity, …)
│ [outlier, if apply_to: voxelized]             │──► /lidar/points/voxelized
└───────────────────────────────────────────────┘    x, y, z, intensity, point_count
        │
        └──► /diagnostics  (counts per stage, per-stage timing, budget check)
```

## Design decisions

* **The algorithms don't depend on ROS.** `pipeline.py`, `filters.py`,
  `voxelization.py`, `spatial_search.py` and `analysis.py` import no ROS code.
  The node only handles ROS I/O. The offline tools run the *same* pipeline
  from the *same* YAML on saved frames.
* **Two outputs.** Voxelization destroys per-point time `t`, `ring` and the
  organized image structure. Phase 3 and Phase 7 will need `t` for
  motion deskewing on a walking robot. So the full-resolution filtered cloud
  keeps every field, and `output.keep_organized: true` even keeps the H × W
  layout, with removed points set to NaN.
* **Zero points are handled explicitly.** Ouster drivers encode "no return" as
  `(0, 0, 0)`, not NaN, so a finiteness check alone misses them. They are
  counted as a separate stage.
* **No per-frame KD-tree in the node.** Nothing consumes it in Phase 1, so it
  would only add latency. `spatial_search.KDTreeIndex` is the reusable
  component for later phases, and `offline_experiments kdtree-bench`
  measures why it exists.
* **Fast conversion.** Decoding uses `np.frombuffer` with a dtype built from
  `msg.fields`/`point_step`/`row_step`, handling padding, row padding and big
  endian. Encoding assigns `array.array('B')` to `msg.data`, which takes
  rclpy's fast path. Assigning bytes or a list makes rclpy validate every
  byte in Python.
* **Validated, runtime-tunable configuration.** Pipeline parameters are
  generated from dataclass defaults, so the YAML and the node can't drift
  apart; a test enforces this. `ros2 param set` changes are validated as a
  whole before being applied.

## Installation

```bash
cd ~/ros2_ws/src
# copy or clone this package here
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
pip install -r src/lidar_pointcloud_processing/requirements.txt
colcon build --symlink-install --packages-select lidar_pointcloud_processing
source install/setup.bash
```

`requirements.txt` pins `numpy<2`, because ROS 2 Humble's Python packages are
built against NumPy 1.x. `rosbags` is needed for `inspect_bag`, and `open3d`
only for `visualize_frame` and the optional `open3d` backends.

## Step 1: close the Phase 0 LiDAR checklist offline

No ROS system and no `metadata.yaml` are needed. The tool reads the `.db3`
directly.

```bash
ros2 run lidar_pointcloud_processing inspect_bag ~/data/grass.db3 \
    --compare-topics /stereo/frame_left/image_raw/compressed \
                     /stereo/frame_right/image_raw/compressed \
                     /unitree/body_odom \
    --save-frames 20 --save-stride 30 --out-dir ~/data/grass_frames \
    --report-json ~/data/grass_lidar_report.json
```

It reports:

* every topic with its type and message count
* the PointCloud2 layout: fields, `frame_id`, H × W, `is_dense`, `point_step`
* LiDAR header-stamp rate, jitter, gaps and non-monotonic stamps
* bag receive time minus header stamp
* valid and zero fractions, the range distribution, the number of rings, and
  the sweep duration from `t`
* for each compare topic, the offset of its nearest stamp to every LiDAR
  stamp. This is the first camera–LiDAR and LiDAR–pose synchronization
  measurement for `docs/synchronization.md`.

Header stamps are read straight from the CDR bytes, so the multi-megabyte
clouds are only fully decoded every `--quality-stride` frames.

## Step 2: run live with the bag

```bash
# Terminal 1: --clock is required for use_sim_time
ros2 bag play ~/data/grass.db3 --clock --remap /tf_static:=/tf_static_raw

# Terminal 2 (static TF republisher from Phase 0, when you need TF)
ros2 run <your_phase0_package> static_tf_republisher

# Terminal 3
ros2 launch lidar_pointcloud_processing pointcloud_processing.launch.py \
    rviz:=true inspector:=true fixed_frame:=<frame_id from the layout log>
```

`ros2 bag play` needs `metadata.yaml`. If you only have the `.db3`, run
`ros2 bag reindex <bag_dir>` once to regenerate it.

Launch arguments: `config_file`, `use_sim_time` (default `true`), `rviz`,
`inspector`, `fixed_frame` (default `os_sensor`), `log_level`.

In RViz2, *Filtered* is enabled by default. Enable *Raw* and *Voxelized* to
compare the three clouds.

Monitor the node:

```bash
ros2 topic hz /lidar/points/voxelized
ros2 topic echo /diagnostics --field status[0].values
```

### Tuning at runtime

```bash
ros2 param set /pointcloud_processing voxel.voxel_size 0.1
ros2 param set /pointcloud_processing range_filter.max_range 30.0
ros2 param set /pointcloud_processing outlier.method statistical
ros2 param set /pointcloud_processing crop_box.enabled true
```

Invalid combinations (for example `min_range > max_range`) are rejected with
a reason, and the running pipeline is left unchanged.

## Topics

| Topic | Type | Content |
|---|---|---|
| `/os_cloud_node/points` (in) | `sensor_msgs/PointCloud2` | raw Ouster cloud |
| `/lidar/points/filtered` | `sensor_msgs/PointCloud2` | all input fields; unorganized, or H × W with NaNs if `keep_organized` |
| `/lidar/points/voxelized` | `sensor_msgs/PointCloud2` | `x y z` + averaged fields + `point_count` |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | per-stage counts, per-stage ms, rolling mean/p95/max; WARN if over budget |

The output header (stamp and `frame_id`) is copied unchanged from the input.

## Parameters

See `config/pointcloud_processing.yaml`; every entry is commented there.

| Group | Parameters | Notes |
|---|---|---|
| I/O (startup only) | `input_topic`, `filtered_topic`, `voxelized_topic`, `diagnostics_topic`, `input_qos_reliability`, `queue_depth`, `stats_window` | `queue_depth: 1` drops stale frames instead of lagging |
| Node (runtime) | `publish_filtered`, `publish_voxelized`, `publish_diagnostics`, `log_every_n_frames`, `processing_budget_ms` | budget = 100 ms at 10 Hz |
| `filters.*` | `remove_nonfinite`, `remove_zero`, `zero_epsilon` | |
| `range_filter.*` | `enabled`, `min_range`, `max_range` | Euclidean range in the sensor frame |
| `crop_box.*` | `enabled`, `min_xyz`, `max_xyz`, `negative` | `negative: true` removes the inside (robot body) |
| `voxel.*` | `enabled`, `voxel_size`, `min_points_per_voxel`, `average_fields`, `include_point_count` | missing fields are skipped with a warning |
| `outlier.*` | `method` (`none`/`statistical`/`radius`), `apply_to` (`voxelized`/`filtered`), `backend` (`scipy`/`open3d`), `nb_neighbors`, `std_ratio`, `radius`, `min_points` | the radius count includes the point itself |
| `output.*` | `keep_organized` | |

ROS 2 YAML cannot express an empty list, so use `average_fields: [""]` to
average nothing. Pipeline parameters accept `30` and `30.0` interchangeably.

## Step 3: Phase 1 experiments

All experiments run on the frames saved in Step 1, using the package YAML
unless you pass `--config`. Add `--csv results.csv` to any of them.

```bash
E="ros2 run lidar_pointcloud_processing offline_experiments"
F=--frames=$HOME/data/grass_frames

$E info $F                          # layout and quality summary
$E pipeline $F                      # per-frame counts and per-stage timing (Exp. 1–2)
$E range-sweep $F --max-ranges 10 20 30 50     # retention + range histogram (Exp. 3)
$E voxel-sweep $F --sizes 0.02 0.05 0.1 0.2    # points / ms / geometric error (Exp. 4)
$E outlier-compare $F --radius 0.2 --min-points 5   # none vs statistical vs radius (Exp. 5)
$E kdtree-bench $F --queries 1000 --k 10       # KD-tree vs brute force (Exp. 6)
```

* `voxel-sweep` reports geometric loss as the distance from each original
  point to its nearest voxel centroid (a one-sided Chamfer distance), with the
  mean and p95 in cm. Add `--compare-open3d` to cross-check against Open3D.
* `outlier-compare` reports the median range of removed vs kept points. If
  removed points are much farther away, the filter is removing sparse
  distant geometry, not noise. This happens because LiDAR density falls with
  range.

Visual inspection with Open3D:

```bash
ros2 run lidar_pointcloud_processing visualize_frame $F --index 0 --view removed
# views: raw | removed | filtered | voxelized | compare
```

Use `--view raw` to measure the robot body's extent before enabling
`crop_box`.

## Performance

On a synthetic worst-case 128 × 2048 scan (262k points, random ranges, so
almost every point lands in its own voxel), one frame takes about 65 ms
single-threaded on a laptop-class CPU. That is under the 100 ms budget at
10 Hz. Real terrain merges many points per voxel and is faster. Outlier removal on the full-resolution cloud
(`apply_to: filtered`) costs hundreds of ms and will not keep up at 10 Hz,
which is why the default is `voxelized`.

## Tests

```bash
cd ~/ros2_ws/src/lidar_pointcloud_processing
python3 -m pytest test -q          # runs with or without ROS installed
# or: colcon test --packages-select lidar_pointcloud_processing
```

The tests cover decoding edge cases (padding, row padding, big endian, short
buffers), encode/decode round trips, every filter, voxel centroids against
hand-computed values, KD-tree vs brute-force agreement, pipeline count
consistency, organized output, config coercion and validation, and YAML ↔
parameter consistency.

## Package layout

```text
lidar_pointcloud_processing/
├── lidar_pointcloud_processing/
│   ├── pointcloud_node.py       ROS node: I/O, parameters, diagnostics
│   ├── pointcloud_inspector.py  ROS node: live layout / timing / quality report
│   ├── pipeline.py              config dataclasses + PointCloudPipeline (no ROS)
│   ├── pointcloud_utils.py      PointCloud2 ⇄ NumPy, layout description
│   ├── filters.py               invalid, zero, range, crop box, outlier masks
│   ├── voxelization.py          voxel grid (centroids, averaged fields, point→voxel map)
│   ├── spatial_search.py        KDTreeIndex, brute-force baselines, benchmark
│   ├── analysis.py              stamp / sync / quality statistics
│   ├── diagnostics.py           FrameStats, RollingStats, DiagnosticArray
│   └── tools/
│       ├── bag_io.py            direct .db3 access, saved-frame I/O
│       ├── inspect_bag.py       offline dataset inspection
│       ├── offline_experiments.py
│       └── visualize_frame.py
├── config/pointcloud_processing.yaml
├── launch/pointcloud_processing.launch.py
├── rviz/pointcloud_processing.rviz
├── test/
├── package.xml  setup.py  setup.cfg  requirements.txt
```

## Out of scope for Phase 1

This package does no RANSAC, clustering, ICP, deskewing, camera projection or
TF transformation. Clouds stay in their own `frame_id`. Those are Phases 2–7.
What Phase 2 consumes: `/lidar/points/voxelized` for geometry, and the
`PointCloudPipeline` and `KDTreeIndex` classes directly for offline work.
