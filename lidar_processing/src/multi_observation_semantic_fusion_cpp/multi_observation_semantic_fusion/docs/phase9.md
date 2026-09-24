# Phase 9 — Multi-Observation Semantic Fusion

## Objective
Fuse repeated observations of the same world voxel without overwriting earlier evidence.

## Method
Each voxel stores a non-negative evidence value for every semantic class. For an observation with class `c` and confidence `q`, the update is:

`E_c <- E_c + q`

A small configurable prior prevents zero-probability classes. The published semantic confidence is:

`p(c|voxel) = E_c / sum_k E_k`

The winning class is the class with maximum posterior evidence.

## Temporal model
`fusion.decay_seconds = 0` disables decay. If positive, old evidence is exponentially decayed before a new observation is applied. This is useful for dynamic environments but should be experimentally evaluated rather than enabled blindly.

## Geometry
The voxel also maintains a running mean elevation/height and observation count. This separates spatial persistence from semantic fusion while keeping enough information for later terrain reasoning.

## Input
Phase 8 persistent/world-frame semantic cloud containing `x`, `y`, `z`, `class_id`, `confidence`.

## Output
A voxelized semantic map containing `x`, `y`, `z`, winning `class_id`, and fused `confidence`.

## Experiments
1. Repeat TREE observations with 0.91, 0.83, 0.95 and inspect confidence.
2. Alternate TREE/GRASS observations and inspect class probabilities indirectly through the winning class and confidence.
3. Compare voxel resolutions 0.05/0.10/0.20 m.
4. Compare no decay vs 10/30/60 s decay on moving/dynamic objects.
5. Measure voxel count, update rate and publish time.
6. Validate that repeated observations do not duplicate map geometry.

## Boundary
No pose optimization, loop closure, occupancy ray tracing, traversability, navigation, or GPU acceleration is included in Phase 9.
