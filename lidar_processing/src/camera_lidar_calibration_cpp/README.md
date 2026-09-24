# Phase 4 — Camera-LiDAR Calibration

C++ / ROS 2 Humble package for connecting LiDAR 3D geometry to camera pixels.

## Core pipeline

```text
LiDAR Point
    |
    v
T_camera_lidar
    |
    v
Camera Coordinates
    |
    v
Perspective Projection
    |
    v
Image Pixel
```

Mathematically:

```text
P_camera = T_camera_lidar P_lidar

u = fx * X/Z + cx
v = fy * Y/Z + cy
```

## Implemented

- Camera intrinsics from `sensor_msgs/CameraInfo`
- Optional explicit intrinsics
- Configurable camera-LiDAR extrinsic transform
- LiDAR -> camera coordinate transformation
- Perspective projection
- Distortion handling for common ROS models
- Minimum depth filtering
- Image boundary filtering
- Pixel-level depth-buffer occlusion handling
- RGB overlay visualization
- Depth-colored LiDAR projection
- Projected camera-frame PointCloud2 output
- ROS 2 configurable topics and parameters

## Important coordinate convention

The package expects:

```text
T_camera_lidar

P_camera = T_camera_lidar * P_lidar
```

The transform must therefore map coordinates from the LiDAR frame into the camera optical coordinate frame.

For ROS optical frames the usual convention is:

```text
X = right
Y = down
Z = forward
```

Do not blindly assume a sensor body frame is an optical frame. Verify the TF tree and calibration convention.

## Build dependencies

```bash
sudo apt install \
  libopencv-dev \
  libpcl-dev \
  ros-humble-cv-bridge \
  ros-humble-pcl-conversions \
  ros-humble-tf2 \
  ros-humble-tf2-ros
```

Build:

```bash
cd ~/your_ws
colcon build --packages-select camera_lidar_calibration
source install/setup.bash
```

Run:

```bash
ros2 launch camera_lidar_calibration calibration.launch.py
```

## Topics

Default configuration:

```text
Input:
  /os_cloud_node/points
  /stereo/frame_left/image_raw
  /stereo/frame_left/camera_info

Output:
  /camera_lidar/overlay
  /camera_lidar/projected_points
```

Change all topics through:

```text
config/calibration.yaml
```

No dataset topic is hardcoded in the C++ implementation.

## CameraInfo

When:

```yaml
calibration:
  use_camera_info: true
```

the node obtains:

```text
image width
image height
fx
fy
cx
cy
distortion model
distortion coefficients
```

from `CameraInfo`.

This is preferred to manually copying intrinsics into source code.

## Raw vs rectified images

For a rectified image topic:

```yaml
apply_distortion: false
```

For a raw distorted image:

```yaml
apply_distortion: true
```

The correct setting depends on the actual image topic and camera pipeline.

## Extrinsic calibration

The current implementation accepts an explicit configurable extrinsic:

```yaml
extrinsic:
  translation_xyz:
    x: ...
    y: ...
    z: ...

  rotation_rpy_rad:
    roll: ...
    pitch: ...
    yaw: ...
```

This represents:

```text
T_camera_lidar
```

not the inverse.

## Occlusion

Multiple LiDAR points may project to the same image pixel.

A depth buffer keeps the closest return.

Parameters:

```yaml
projection:
  occlusion:
    enabled: true
    window_size: 1
    depth_tolerance_m: 0.15
```

This is a visibility approximation, not a full mesh/ray-tracing solution.

## Phase 4 experiments

### Experiment 1 — Identity transform

Set:

```text
T_camera_lidar = Identity
```

Only as a sanity check.

### Experiment 2 — Dataset/reference calibration

Use the known dataset calibration.

Verify:

- projected points fall on the correct image structures
- no systematic translation
- no systematic rotation
- correct depth ordering

### Experiment 3 — Perturbation

Intentionally perturb:

```text
translation
roll
pitch
yaw
```

and observe how the overlay moves.

This builds intuition for calibration sensitivity.

### Experiment 4 — Compare calibration

Later compare:

```text
Reference calibration
        vs
Our estimated calibration
```

using reprojection error and visual alignment.

## Reprojection error

For a known corresponding 3D/2D pair:

```text
e = sqrt(
    (u_est - u_ref)^2 +
    (v_est - v_ref)^2
)
```

This should become a primary calibration metric.

## Current boundary

This package performs:

```text
intrinsics
extrinsics
3D -> 2D projection
visibility filtering
visualization
```

It does not yet estimate the extrinsic automatically.

The next calibration-estimation implementation can use:

- target-board correspondences
- plane/edge correspondences
- optimization
- PnP-style formulations
- nonlinear refinement

The key experimental requirement is to compare the independently estimated transform against the dataset-provided reference.
