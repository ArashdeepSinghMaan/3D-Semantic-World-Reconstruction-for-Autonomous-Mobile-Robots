# world_frame_reconstruction — Phase 7

ROS 2 Humble C++ package for transforming Phase 6 semantic 3D observations into a common world coordinate frame and accumulating them over time.

## Pipeline

```text
Phase 6 Semantic PointCloud2
          |
          v
  cloud timestamp + frame_id
          |
          v
   TF lookup at timestamp
          |
          v
 T_world_sensor(t)
          |
          v
    Transform XYZ
          |
          v
World-frame semantic cloud
          |
          +----> current frame
          |
          +----> bounded accumulation
```

## Build

```bash
cd ~/your_ros2_ws/src
unzip world_frame_reconstruction_cpp.zip
cd ..
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select world_frame_reconstruction
source install/setup.bash
```

If PCL is missing:

```bash
sudo apt install libpcl-dev
```

## Run

```bash
ros2 launch world_frame_reconstruction world_frame_reconstruction.launch.py
```

## Important input contract

The input cloud should be produced by Phase 6 and contain:

```text
x          FLOAT32
y          FLOAT32
z          FLOAT32
class_id   INT32
confidence FLOAT32
```

Its `header.stamp` must correspond to the observation time and its `header.frame_id` must identify the sensor frame.

## Parameters

See `config/world_frame_reconstruction.yaml`.

The most important parameter is:

```yaml
frames:
  world: map
```

Change this to the actual world/map frame used by the dataset.

## Why TF is used instead of manually multiplying transforms

The conceptual equation is:

```text
P_world = T_world_robot * T_robot_lidar * P_lidar
```

ROS TF already stores and composes those transformations. Looking up:

```text
world <- lidar
```

at the cloud timestamp is therefore equivalent to applying the complete transform chain, provided the TF tree is correct.

## Current limitation

Accumulation is a bounded point buffer, not a spatially fused map. The same physical surface can therefore appear multiple times. This is deliberate: Phase 8 will introduce persistent spatial map integration.
