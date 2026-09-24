# Multi-Observation Semantic Fusion — Phase 9

C++ ROS 2 Humble package for confidence-aware semantic fusion over repeated observations of a persistent world-frame voxel map.

## Pipeline

Phase 8 world semantic cloud -> voxel lookup -> confidence/evidence update -> normalized semantic confidence -> fused voxel map.

## Build

```bash
sudo apt install libpcl-dev ros-humble-pcl-conversions
cd ~/your_ros2_ws/src
unzip multi_observation_semantic_fusion_cpp.zip
cd ..
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select multi_observation_semantic_fusion
source install/setup.bash
```

## Run

```bash
ros2 launch multi_observation_semantic_fusion multi_observation_semantic_fusion.launch.py
```

## Fusion model

For an observation with class `c` and confidence `q`:

`E_c <- E_c + q`

The published confidence is the normalized evidence of the winning class. A configurable exponential temporal decay can be enabled with `fusion.decay_seconds`.

## Input contract

The input `sensor_msgs/PointCloud2` must contain `x`, `y`, `z`, `class_id`, and `confidence`. Coordinates are expected to already be in the world/map frame.

## Output

`/semantic_fusion/map` — one representative point per voxel with winning class and fused confidence.

`/semantic_fusion/voxel_count` — current number of occupied semantic voxels.

## Phase boundary

Phase 9 focuses on repeated semantic evidence. Pose optimization, loop closure, occupancy mapping, traversability, navigation, and GPU acceleration are intentionally outside this package.
