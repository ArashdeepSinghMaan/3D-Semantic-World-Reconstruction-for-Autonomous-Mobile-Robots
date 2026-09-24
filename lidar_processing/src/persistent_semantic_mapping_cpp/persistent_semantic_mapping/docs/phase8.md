# Phase 8 — Persistent 3D Semantic Mapping

## Objective
Convert world-frame semantic observations into a persistent spatial representation.

## Representation
This implementation uses a configurable 3D voxel map. Each voxel stores:
- accumulated semantic class scores
- accumulated confidence
- observation count
- confidence maximum
- elevation statistics
- last observation timestamp

The published map is a semantic voxel-center PointCloud2 with fields:
`x, y, z, class_id, confidence`.

## Update rule
For each world-frame semantic point:
1. Compute voxel key by floor(position / resolution).
2. Ignore invalid points and confidence below the configured threshold.
3. Add the observation confidence to that class's score.
4. Accumulate geometry/elevation statistics.
5. Select the class with maximum accumulated score as the current voxel label.

This is deliberately an accumulation baseline, not yet a Bayesian/Dirichlet semantic fusion model.

## Experiments
1. Static scene: move robot around a structure and verify the same structure occupies stable voxels.
2. Resolution sweep: 0.05, 0.10, 0.20, 0.30 m.
3. Confidence threshold sweep.
4. Observe class stability after repeated observations.
5. Measure voxel count and publication time as map size grows.

## Boundary
Not implemented here:
- ray-based free-space occupancy
- probabilistic occupancy grids
- temporal decay
- loop closure / pose graph optimization
- semantic Bayesian fusion
- GridMap conversion
- terrain reasoning
- navigation
- GPU acceleration

Those belong to later phases.
