# Phase 4 — Camera-LiDAR Calibration

## 1. Objective

Phase 4 establishes the geometric relationship between camera pixels and LiDAR points.

```text
LiDAR
  |
  | T_camera_lidar
  v
Camera coordinates
  |
  | projection
  v
Image pixel
```

This is essential for later semantic 3D fusion.

The camera can provide:

```text
class
mask
confidence
```

while LiDAR provides:

```text
3D position
depth
geometry
```

The calibration allows these observations to refer to the same physical location.

---

## 2. Intrinsics

The pinhole model is:

```text
u = fx X/Z + cx
v = fy Y/Z + cy
```

where:

```text
fx, fy = focal lengths
cx, cy = principal point
```

The implementation obtains these from ROS `CameraInfo` when configured.

---

## 3. Extrinsics

The rigid transformation is:

```text
P_camera = T_camera_lidar P_lidar
```

with:

```text
T = [ R t ]
    [ 0 1 ]
```

where:

```text
R = rotation
t = translation
```

This transform is currently configurable.

That is deliberate: we want to separate the projection implementation from the later calibration-estimation experiment.

---

## 4. Coordinate frames

One of the most important Phase 4 lessons is that:

```text
LiDAR frame
camera body frame
camera optical frame
```

are not necessarily identical.

ROS optical coordinates generally use:

```text
X = right
Y = down
Z = forward
```

A transform that is numerically correct in one convention can produce a completely incorrect image overlay if the frame convention is misunderstood.

---

## 5. Projection pipeline

For each LiDAR point:

```text
P_lidar
   |
   v
P_camera = T_camera_lidar P_lidar
   |
   v
check Z > minimum depth
   |
   v
normalize X/Z, Y/Z
   |
   v
apply distortion if required
   |
   v
u,v
   |
   v
check image bounds
```

---

## 6. Occlusion

Different 3D points can project to the same pixel.

Example:

```text
LiDAR
  |
  +---- near object
  |
  +--------- far object
                  \
                   image pixel
```

The nearer point should normally be visible.

The package therefore implements a configurable depth-buffer approximation.

This is not a full geometric visibility solution. Later stages can replace it with more sophisticated visibility reasoning if required.

---

## 7. Visualization

The package publishes:

```text
/camera_lidar/overlay
```

which contains:

```text
RGB image
+
projected LiDAR points
```

It also publishes:

```text
/camera_lidar/projected_points
```

containing valid projected points represented in the camera coordinate frame.

---

## 8. Critical experiment

The FusionPortable dataset provides calibration information.

We should **not simply accept it as the final calibration**.

The intended experiment is:

```text
Dataset reference calibration
             |
             v
       projection
             |
             v
        reference
```

versus:

```text
Our independently estimated calibration
             |
             v
       projection
             |
             v
       estimated
```

Then compare:

```text
reprojection error
visual alignment
translation difference
rotation difference
```

This makes Phase 4 an actual calibration study rather than merely a projection implementation.

---

## 9. Calibration sensitivity

Intentionally perturb the extrinsic parameters:

```text
+ translation
- translation
+ roll
+ pitch
+ yaw
```

and observe the overlay.

This demonstrates why even small extrinsic errors can produce significant image misalignment, especially at longer LiDAR ranges.

---

## 10. Phase 4 boundary

Implemented:

```text
CameraInfo
intrinsics
extrinsic transform
LiDAR -> camera
projection
distortion handling
image bounds
occlusion approximation
visualization
```

Not yet implemented:

```text
automatic extrinsic estimation
nonlinear calibration optimization
reprojection-error optimization
target-board calibration
continuous calibration
```

Those should be added as the experimental calibration-estimation part of Phase 4 rather than mixed into the basic projection pipeline.

---

## 11. Connection to Phase 5

After calibration:

```text
LiDAR point
    |
    +--> 3D geometry
    |
    +--> camera pixel
              |
              v
       semantic prediction
              |
              v
      semantic 3D point
```

That is the foundation of Phase 5/6 semantic fusion.
