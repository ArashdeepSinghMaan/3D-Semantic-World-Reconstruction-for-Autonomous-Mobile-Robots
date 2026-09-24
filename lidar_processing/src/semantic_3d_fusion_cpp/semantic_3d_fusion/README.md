# semantic_3d_fusion — Phase 6

C++ ROS 2 Humble implementation of semantic 3D fusion.

## Pipeline

```text
                 Camera
                   |
                   v
        +-----------------------+
        | semantic/class_map    |
        | semantic/confidence   |
        +-----------+-----------+
                    |
                    v
LiDAR ---> Transform ---> Project ---> Pixel sample
  |                                      |
  +--------------------------------------+
                    |
                    v
       (x,y,z,class_id,confidence,time)
                    |
                    v
          /semantic_3d/points
```

## Dependencies

```bash
sudo apt install \
  ros-humble-rclcpp \
  ros-humble-sensor-msgs \
  ros-humble-std-msgs \
  ros-humble-tf2 \
  ros-humble-tf2-ros \
  ros-humble-tf2-sensor-msgs \
  libeigen3-dev
```

## Build

```bash
cd ~/your_ros2_ws
colcon build --packages-select semantic_3d_fusion --symlink-install
source install/setup.bash
```

## Run

```bash
ros2 launch semantic_3d_fusion semantic_3d_fusion.launch.py
```

Or specify another YAML:

```bash
ros2 launch semantic_3d_fusion semantic_3d_fusion.launch.py \
  config:=/absolute/path/to/semantic_3d_fusion.yaml
```

## Default topics

| Purpose | Topic | Type |
|---|---|---|
| LiDAR | `/lidar/points/processed` | `sensor_msgs/PointCloud2` |
| Class map | `/semantic/class_map` | `std_msgs/Int32MultiArray` |
| Confidence | `/semantic/confidence_map` | `std_msgs/Float32MultiArray` |
| Camera info | `/stereo/frame_left/camera_info` | `sensor_msgs/CameraInfo` |
| Semantic cloud | `/semantic_3d/points` | `sensor_msgs/PointCloud2` |
| Unassigned | `/semantic_3d/unassigned_points` | `sensor_msgs/PointCloud2` |

## Quick test

First run Phase 5 and publish valid class/confidence maps. Then run this package.

Check:

```bash
ros2 topic echo /semantic_3d/points --once
ros2 topic hz /semantic_3d/points
ros2 topic info /semantic_3d/points
```

Inspect fields:

```bash
ros2 topic echo /semantic_3d/points --once
```

The output PointCloud2 contains `x`, `y`, `z`, `class_id`, `confidence`, and `timestamp`.

## Critical calibration convention

The configured extrinsic is **LiDAR -> camera**:

```text
P_camera = T_camera_lidar P_lidar
```

Do not accidentally enter the inverse camera->LiDAR transform.

If TF is used, the node performs the equivalent lookup:

```text
lookupTransform(camera_frame, lidar_frame, stamp)
```

## Phase boundary

This package creates a camera-frame semantic 3D observation cloud. It does not yet create a persistent world map. World-frame accumulation begins in Phase 7.
