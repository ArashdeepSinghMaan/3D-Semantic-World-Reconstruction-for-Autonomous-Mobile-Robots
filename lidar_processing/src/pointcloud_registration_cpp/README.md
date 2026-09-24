# Phase 3 — Point-Cloud Registration (C++ / ROS 2 Humble)

This package implements consecutive-frame LiDAR registration using PCL.

## Pipeline

```text
Phase 1 processed cloud
        |
        +-------------------+
        |                   |
   Frame t             Frame t+1
        |                   |
      Source              Target
        |                   |
        +-------- ICP ------+
                   |
                   v
             T_source_target
                   |
                   v
          aligned source cloud
```

## Implemented

- PCL point-to-point ICP
- PCL point-to-plane ICP with normals
- Coarse-to-fine multi-scale ICP
- Configurable initial transformation
- Voxel downsampling
- Registration fitness
- Inlier RMSE
- Translation/rotation metrics
- Processing time
- ROS 2 PointCloud2 input/output
- RViz2 visualization

## Build dependencies

Ubuntu / ROS 2 Humble:

```bash
sudo apt install \
  libpcl-dev \
  ros-humble-pcl-conversions \
  ros-humble-tf2 \
  ros-humble-tf2-ros
```

## Build

```bash
cd ~/your_ws
colcon build --packages-select pointcloud_registration
source install/setup.bash
```

## Run

```bash
ros2 launch pointcloud_registration registration.launch.py
```

Custom YAML:

```bash
ros2 launch pointcloud_registration registration.launch.py \
  config_file:=/path/to/registration.yaml
```

## Input

Default:

```text
/lidar/points/processed
```

## Outputs

```text
/registration/aligned
/registration/target
/registration/metrics
```

## ICP modes

### Point-to-point

```yaml
icp:
  method: point_to_point
```

### Point-to-plane

```yaml
icp:
  method: point_to_plane
```

For point-to-plane ICP, PCL estimates local surface normals using a KdTree.

## Multi-scale

```yaml
multiscale:
  enabled: true
  voxel_sizes: [0.20, 0.10, 0.05]
  correspondence_factors: [2.5, 2.0, 1.5]
  iterations: [40, 30, 20]
```

The transformation is carried from the coarse scale to the finer scale.

## Initial transform

```yaml
initial_transform:
  translation_xyz:
    x: 0.0
    y: 0.0
    z: 0.0
  rotation_rpy_rad:
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
```

Zero is useful for the baseline ICP experiment.

Later this can be replaced by an odometry/IMU/TF-derived motion prior.

## Important interpretation

The output is a relative transformation between consecutive LiDAR observations.

It is NOT yet:

- a world pose
- a globally optimized trajectory
- loop closure
- global registration
- SLAM
- a persistent semantic map

Those are deliberately later phases.

## Experiments

1. Point-to-point + zero initial guess
2. Point-to-point + non-zero initial guess
3. Point-to-plane
4. Multi-scale point-to-plane
5. Flat vs vegetation vs sparse geometry
6. Different overlap and inter-frame motion

Record:

```text
converged
fitness
inlier RMSE
translation
rotation
processing time
```

The important lesson is:

```text
ICP convergence does not guarantee correct registration.
```
