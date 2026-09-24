# Phase 7 — World-Frame Reconstruction

## Objective
Transform semantic 3D observations from the sensor frame into a common world coordinate frame using the ROS 2 TF tree and the pose available at each observation timestamp.

Conceptually:

`P_world = T_world_robot * T_robot_lidar * P_lidar`

The implementation asks TF for the composed transform directly:

`T_world_sensor(t) = TF(world <- sensor, t)`

This lets ROS TF compose intermediate robot/base/sensor transforms.

## Input

The input is the Phase 6 semantic point cloud. It should contain:

- x, y, z: FLOAT32
- class_id: INT32
- confidence: FLOAT32

The input `header.frame_id` identifies the sensor frame. A `source_override` parameter can be used when the recorded frame ID needs to be overridden.

## Output

Two PointCloud2 topics are published:

- `/world_reconstruction/points`: transformed observations from the current frame.
- `/world_reconstruction/accumulated`: bounded accumulation in the world frame.

Semantic fields are preserved.

## TF synchronization

For every input cloud, the node calls:

`lookupTransform(world_frame, source_frame, cloud_timestamp)`

Therefore the transform is requested at the sensor observation time rather than using the latest robot pose.

This is important for a moving robot.

## Accumulation

Accumulation is bounded by `accumulation.max_points`. The oldest observations are removed once the limit is reached.

This is intentionally a simple temporal accumulation stage. It does not yet perform:

- voxel-map integration
- duplicate removal
- semantic Bayesian fusion
- loop closure
- pose-graph optimization
- ICP correction
- persistent map serialization

Those belong to later phases.

## Experiment

1. Start the TF tree / recorded bag.
2. Start Phase 6 semantic fusion.
3. Start this package.
4. Visualize `/world_reconstruction/points` and `/world_reconstruction/accumulated` in RViz2 with `map` as the fixed frame.
5. Move through a static scene.
6. Check whether static structures remain spatially consistent as the robot moves.

### Useful measurements

Record:

- cloud timestamp
- TF lookup success/failure
- number of input points
- transformed points
- accumulated points
- processing time
- frame-to-frame displacement

## Expected TF tree

A typical chain is:

`map -> odom -> base_link -> lidar`

or an equivalent dataset-specific chain.

Do not assume these names. Inspect the actual TF tree from the FusionPortable bag before configuring the package.

## Phase 7 boundary

Phase 7 establishes the world-frame observation pipeline. It is not yet a persistent semantic map. Phase 8 should introduce a spatial map representation and map integration strategy.
