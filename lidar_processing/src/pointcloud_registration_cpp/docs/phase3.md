# Phase 3 — Point-Cloud Registration

## Objective

Understand how consecutive LiDAR observations are aligned.

```text
Frame t       -> Point Cloud A
Frame t + 1   -> Point Cloud B
                       |
                       v
                     ICP
                       |
                       v
                 Transformation T
```

The estimated rigid transform is:

```text
T = [ R  t ]
    [ 0  1 ]
```

such that:

```text
P_target ≈ T P_source
```

## Why this phase matters

Phase 2 describes geometry inside a single observation.

Phase 3 establishes the relationship between observations.

This is the first step toward persistent world reconstruction.

## Point-to-point ICP

The point-to-point objective is conceptually:

```text
E = sum ||p_i - T q_i||²
```

It is the baseline method.

## Point-to-plane ICP

With target normals:

```text
E = sum (n_i^T (p_i - T q_i))²
```

This uses local surface orientation and is particularly useful around structured surfaces.

## Multi-scale ICP

Registration is performed coarse-to-fine:

```text
0.20 m -> 0.10 m -> 0.05 m
```

The coarse stages establish broad alignment and the fine stages refine it.

## Initial transformation

ICP is a local method.

A good initial guess increases the chance that the correct correspondences fall within the correspondence threshold.

The package therefore makes the initial transform configurable.

Future source:

```text
wheel odometry
IMU
TF
visual odometry
```

## Metrics

The package reports:

- convergence state
- fitness score
- inlier RMSE
- translation magnitude
- rotation magnitude
- change relative to the initial guess
- processing time
- source point count
- target point count

## Failure modes to investigate

- insufficient overlap
- large motion
- incorrect initial guess
- repetitive geometry
- mostly planar geometry
- vegetation
- sparse clouds
- dynamic objects
- unsuitable correspondence distance

A key lesson:

```text
convergence != correctness
```

## Phase boundary

Included:

```text
ICP
point-to-point
point-to-plane
multi-scale
initial transform
metrics
visualization
```

Deferred:

```text
global registration
loop closure
trajectory optimization
odometry fusion
IMU fusion
Ceres/GTSAM
persistent mapping
```

## Next phase

Phase 4 will establish the camera-LiDAR geometric relationship required to transfer image semantics into 3D.
