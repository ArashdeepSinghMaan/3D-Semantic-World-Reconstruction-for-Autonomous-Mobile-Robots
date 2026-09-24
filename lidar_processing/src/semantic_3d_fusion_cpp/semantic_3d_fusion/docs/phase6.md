# Phase 6 — Semantic 3D Fusion

## Objective

Transfer semantic information from the camera image into 3D LiDAR points.

For every valid LiDAR point:

1. Transform LiDAR coordinates into the camera frame.
2. Project the 3D point into the image.
3. Check image bounds and depth.
4. Sample the semantic class map.
5. Sample the confidence map.
6. Reject low-confidence/invalid labels according to configuration.
7. Emit a semantic 3D point.

Conceptually:

```text
LiDAR point (x,y,z)
       |
       v
T_camera_lidar
       |
       v
Camera point (X,Y,Z)
       |
       v
u = fx X/Z + cx
v = fy Y/Z + cy
       |
       v
(class_id, confidence)
       |
       v
(x,y,z,class_id,confidence,timestamp)
```

## Output

The main output is a `sensor_msgs/PointCloud2` on `/semantic_3d/points` with fields:

- `x`, `y`, `z`: original LiDAR coordinates
- `class_id`: `int32`
- `confidence`: `float32`
- `timestamp`: `float64` seconds

The output header retains the LiDAR frame and timestamp.

A second optional output `/semantic_3d/unassigned_points` contains points that project successfully but cannot receive a valid semantic label.

## Important interface note

The current Phase 5 package publishes class/confidence maps using `std_msgs/msg/Int32MultiArray` and `Float32MultiArray`. Those messages do not contain a ROS header. This Phase 6 node therefore uses the most recent maps and stamps the fused point with the LiDAR timestamp. It does **not** claim exact temporal synchronization.

For research-grade synchronization, the next interface should carry a header and image timestamp, preferably with a custom semantic-map message or `sensor_msgs/Image` plus metadata. Until then, the current implementation is suitable for validating the spatial projection/fusion mechanism.

## Calibration modes

### Manual extrinsic

Set `extrinsic.use_tf: false` and configure the LiDAR-to-camera transform directly.

The transform must satisfy:

```text
P_camera = R_camera_lidar * P_lidar + t_camera_lidar
```

### TF-based extrinsic

Set `extrinsic.use_tf: true` and configure `extrinsic.lidar_frame` and `extrinsic.camera_frame`. The node requests the transform at each LiDAR timestamp.

This mode is preferred once the static calibration is correctly published in TF.

## Camera model

With `camera.use_camera_info: true`, the node uses the latest `CameraInfo`. It prefers the rectified projection matrix `P` when available and falls back to `K`.

The current projection assumes a rectified pinhole image. Distortion is intentionally not applied in Phase 6 because Phase 4 can provide rectified camera geometry.

## Semantic sampling

Default: nearest-neighbour pixel sampling.

Optional: `pixel_radius > 0` with `nearest_pixel: false` searches a local square neighborhood and chooses the highest-confidence semantic sample. This can make the fusion more robust near mask boundaries, but it is not a replacement for true probabilistic fusion.

## Experiments

1. Reference calibration + perfect synthetic semantic map.
2. Reference calibration + Phase 5 predictions.
3. Perturb extrinsic translation and measure semantic projection degradation.
4. Perturb extrinsic rotation and measure degradation.
5. Compare nearest-pixel and local-window sampling.
6. Sweep confidence thresholds.
7. Measure projected/assigned/unknown ratios and runtime.
8. Compare semantic point clouds against RGB overlay.

## What Phase 6 does NOT do

- No world-frame transformation.
- No persistent map.
- No semantic voting over multiple observations.
- No loop closure.
- No Ceres/GTSAM optimization.
- No traversability reasoning.
- No navigation.
- No automatic camera-LiDAR calibration.

Those belong to later phases.
