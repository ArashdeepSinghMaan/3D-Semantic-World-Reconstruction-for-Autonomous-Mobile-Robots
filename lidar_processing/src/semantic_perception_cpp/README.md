# Phase 5 — Semantic Perception

ROS 2 Humble / C++ semantic perception interface.

## Purpose

The package turns RGB images into a model-independent semantic representation:

```text
class
confidence
mask
bounding box
```

The mapping system does not need to know how the semantic model was trained.

## Package

```text
semantic_perception/
├── CMakeLists.txt
├── package.xml
├── include/
│   └── semantic_perception/
│       ├── types.hpp
│       ├── inference.hpp
│       ├── postprocess.hpp
│       └── visualization.hpp
├── src/
│   ├── inference.cpp
│   ├── postprocess.cpp
│   ├── visualization.cpp
│   └── semantic_perception_node.cpp
├── config/
│   └── semantic_perception.yaml
├── launch/
│   └── semantic_perception.launch.py
├── rviz/
│   └── semantic_perception.rviz
├── docs/
│   └── phase5.md
└── README.md
```

## Dependencies

```bash
sudo apt install \
  libopencv-dev \
  ros-humble-cv-bridge
```

## Build

```bash
cd ~/your_ws
colcon build --packages-select semantic_perception
source install/setup.bash
```

## Configure model

Edit:

```text
config/semantic_perception.yaml
```

Set:

```yaml
model:
  path: /absolute/path/to/model.onnx
```

Example:

```yaml
model:
  path: /home/user/models/semantic.onnx
  backend: opencv
  target: cpu
```

For a CUDA-enabled OpenCV build:

```yaml
backend: cuda
```

or:

```yaml
backend: cuda_fp16
```

## Run

```bash
ros2 launch semantic_perception semantic_perception.launch.py
```

## Input

Default:

```text
/stereo/frame_left/image_raw
```

## Outputs

```text
/semantic/detections
/semantic/class_map
/semantic/confidence_map
/semantic/visualization
```

## Class configuration

Classes are configuration, not hardcoded into the inference algorithm:

```yaml
classes:
  names:
    - road
    - floor
    - bushes
    - trees
    - stairs
    - trench
    - puddle
    - rubble
    - mud
    - gravel
    - grass
    - dirt
```

Replace this list with the exact class ordering of the model.

## Model compatibility

The included ONNX decoder supports a common YOLO detection tensor:

```text
[1, attributes, candidates]
```

It is not a universal decoder for all YOLO segmentation exports.

For a segmentation ONNX model, the model-specific adapter must decode:

```text
detection tensor
mask coefficients
prototype tensor
```

and populate:

```cpp
Detection::mask
```

The ROS semantic interface remains unchanged.

## Performance

The node reports:

```text
inference_time_ms
postprocess_time_ms
```

Use these to identify whether performance is limited by:

```text
model inference
post-processing
visualization
message serialization
```

## Phase 5 boundary

This package does not perform:

```text
LiDAR fusion
camera-LiDAR calibration
ICP
world mapping
navigation
```

It only generates semantic observations.

That separation is intentional.


## Recommended Phase 5 validation

Before connecting this node to Phase 4, validate the semantic model independently:

```text
single image
    ↓
model
    ↓
class IDs
confidence
boxes/masks
    ↓
visualization
```

Only after the model output is verified should it be connected to:

```text
Camera-LiDAR projection
        ↓
3D semantic fusion
```

This prevents a model decoding error from being mistaken for a calibration or fusion error.
