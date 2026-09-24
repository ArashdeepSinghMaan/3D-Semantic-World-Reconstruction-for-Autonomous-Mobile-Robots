# Phase 5 — Semantic Perception

## Objective

Generate semantic information from camera observations while keeping the semantic interface independent from the mapping system.

```text
RGB Image
    |
    v
Semantic Model
    |
    +-- class
    +-- confidence
    +-- bounding box
    +-- mask
```

## Architecture

```text
Camera Image
      |
      v
+------------------+
| Inference        |
+------------------+
      |
      v
+------------------+
| Post-processing  |
+------------------+
      |
      +--> detections
      |
      +--> class map
      |
      +--> confidence map
      |
      v
+------------------+
| Visualization    |
+------------------+
```

## Model interface

The package accepts an ONNX model path through YAML.

The model backend can be selected as:

```text
opencv
cuda
cuda_fp16
```

The implementation uses OpenCV DNN as the model abstraction layer.

This keeps the ROS node independent from the semantic model itself.

## Important limitation

The current decoder is intentionally a generic YOLO-style detection decoder for common ONNX output:

```text
[1, attributes, candidates]
```

It does not pretend that every YOLO segmentation export has the same output structure.

For an actual segmentation model, the next model-specific adapter should decode:

```text
detection tensor
+
mask coefficients
+
prototype masks
```

and populate `Detection.mask`.

This keeps model-specific tensor interpretation separate from the semantic ROS interface.

## Semantic output

Every pixel can be represented by:

```text
class_id
confidence
```

The node publishes:

```text
/semantic/class_map
/semantic/confidence_map
```

The detection interface additionally contains:

```text
class
confidence
bounding box
```

## Why keep the interface model-independent?

Later the model can change:

```text
YOLO
DeepLab
SegFormer
SAM-derived model
foundation model
custom segmentation network
```

without changing the mapping system.

The mapping system should consume:

```text
semantic class
semantic confidence
pixel/region
```

rather than knowing how the prediction was generated.

## Pixel maps

The current fallback builds pixel maps from detection boxes.

If segmentation masks are available, masks override the box-level assignment.

For a true semantic segmentation model, the ideal output is:

```text
class_map[y,x]
confidence_map[y,x]
```

directly from the model.

## Phase 5 experiments

1. Run model on a single image.
2. Verify class IDs.
3. Verify confidence values.
4. Verify bounding boxes.
5. Verify segmentation masks.
6. Measure inference time.
7. Measure post-processing time.
8. Measure end-to-end image rate.
9. Test different confidence thresholds.
10. Test day/night/fog/rain images.

## Important design boundary

Phase 5 does not know anything about:

```text
LiDAR
ICP
world frame
GridMap
navigation
```

It only produces semantic observations.

The next phase can consume those observations and associate them with 3D geometry using the Camera-LiDAR calibration from Phase 4.

## Phase 5 → Phase 6

```text
RGB
 |
 v
Semantic perception
 |
 +---- class
 +---- confidence
 +---- mask
 |
 v
Camera-LiDAR projection
 |
 v
3D semantic observations
 |
 v
Semantic fusion
```
