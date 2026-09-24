# Phase 8 — Persistent 3D Semantic Mapping

C++ ROS 2 Humble package implementing a persistent voxel-based semantic world map.

## Pipeline

```text
Phase 7 world-frame semantic cloud
                |
                v
       voxel spatial indexing
                |
                v
      semantic score accumulation
                |
                v
       persistent voxel map
                |
                v
     /semantic_map/points
```

## Build

```bash
cd ~/your_ros2_ws/src
unzip persistent_semantic_mapping_cpp.zip
cd ..
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select persistent_semantic_mapping
source install/setup.bash
```

## Run

```bash
ros2 launch persistent_semantic_mapping persistent_semantic_mapping.launch.py
```

## Input contract

The input PointCloud2 must contain:
- `x`: FLOAT32
- `y`: FLOAT32
- `z`: FLOAT32
- `class_id`: INT32
- `confidence`: FLOAT32

The expected input is Phase 7 world-frame output.

## Output

`/semantic_map/points` is a persistent voxel-center PointCloud2 containing the current winning semantic class and normalized class confidence per voxel.

`/semantic_map/voxel_count` publishes the current number of stored voxels.

Optional MarkerArray visualization can be enabled in YAML.

## Key configuration

```yaml
voxel:
  resolution: 0.10
  min_confidence: 0.0
  max_classes_per_voxel: 16
publish:
  rate_hz: 2.0
  max_voxels: 0
```

`max_voxels: 0` means publish all voxels. This is separate from map storage.

## Important architectural choice

The persistent map is keyed in the **world/map frame**, not in the sensor frame. Phase 7 is therefore a prerequisite.

The voxel map is the primary Phase 8 representation. A GridMap bridge can be added later when terrain reasoning needs 2.5D layers such as elevation, traversability, slope, roughness, semantic labels, and negative-obstacle evidence.
