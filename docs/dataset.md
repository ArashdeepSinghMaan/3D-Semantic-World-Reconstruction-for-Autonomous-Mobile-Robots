
# Phase 0 — Dataset Understanding

## FusionPortable V2 — Legged Robot Dataset

This project uses the **FusionPortable V2 dataset** for developing and evaluating a 3D semantic world reconstruction pipeline for autonomous mobile robots.

The dataset provides multi-modal sensor observations from a quadruped platform, including:

* LiDAR
* Stereo cameras
* Event cameras
* Multiple IMUs
* Robot body odometry
* Joint/foot state
* Camera calibration information
* Static TF information
* LiDAR-derived range/intensity imagery

The purpose of Phase 0 is to understand the available data, sensor rates, coordinate frames, calibration information, timestamps, and ROS 2 bag structure before implementing the reconstruction pipeline.

Dataset reference:

https://fusionportable.github.io/dataset/fusionportable_v2_data/

---

# 1. Phase 0 Objectives

The objectives of this phase are:

1. Understand the sensor suite available in the dataset.
2. Identify the ROS 2 topics required for the project.
3. Understand message types and approximate sensor frequencies.
4. Identify camera calibration information.
5. Understand the available robot-state and pose information.
6. Understand the LiDAR point-cloud representation.
7. Understand the TF/static-TF structure.
8. Establish a reliable ROS 2 bag replay procedure.
9. Validate timestamp synchronization between sensors.
10. Establish the data required for subsequent project phases.

The output of this phase is a documented and reproducible dataset interface that can be used by all later phases.

---

# 2. Selected ROS 2 Bag

The initial development sequence is stored as:

```text
grass.db3
```

The provided ROS bag metadata is:

| Property                  | Value                    |
| ------------------------- | ------------------------ |
| Storage                   | SQLite3                  |
| ROS distribution metadata | `rosbags`                |
| Duration                  | 301.995101670 s          |
| Duration                  | ~5.03 min                |
| Total messages            | 638,117                  |
| Start time                | `1690962427004361528 ns` |
| Compression               | None                     |
| Database                  | `grass.db3`              |

The bag contains a single database file:

```text
grass.db3
```

---

# 3. Sensor Overview

The bag contains the following major sensor groups.

```text
                         grass.db3
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
      LiDAR               Cameras              IMUs
        │                    │                    │
        │              ┌─────┴─────┐       ┌──────┼──────┐
        │              │           │       │      │      │
        ▼              ▼           ▼       ▼      ▼      ▼
   PointCloud2      Frame       Davis     STIM   Unitree  LiDAR
                   Camera       Camera    300     IMU      IMU
        │
        ▼
  Range / Intensity
     Images

        ┌─────────────────────────────────────────┐
        │ Robot State / Pose / Calibration / TF   │
        └─────────────────────────────────────────┘
```

---

# 4. LiDAR

The primary LiDAR point-cloud topic is:

```text
/os_cloud_node/points
```

Message type:

```text
sensor_msgs/msg/PointCloud2
```

Message count:

```text
3020
```

Given the approximately 302-second recording duration, this corresponds to approximately:

```text
3020 / 302 ≈ 10 Hz
```

This will be one of the primary inputs for the 3D reconstruction pipeline.

## LiDAR-derived image topics

The bag also contains:

```text
/os_image_node/range_image
/os_image_node/nearir_image
/os_image_node/signal_image
/os_image_node/reflec_image
```

All use:

```text
sensor_msgs/msg/Image
```

with approximately:

```text
3020 messages
```

These provide additional information derived from the LiDAR sensor and may be useful later for:

* Range visualization
* Intensity analysis
* Reflectivity analysis
* Sensor debugging
* LiDAR-camera association experiments

They are not required for the first implementation of the semantic reconstruction pipeline.

---

# 5. Stereo Frame Cameras

The dataset provides left and right frame-camera streams.

## Left camera

```text
/stereo/frame_left/image_raw/compressed
```

Message type:

```text
sensor_msgs/msg/CompressedImage
```

Message count:

```text
6040
```

Camera calibration:

```text
/stereo/frame_left/camera_info
```

Message type:

```text
sensor_msgs/msg/CameraInfo
```

Message count:

```text
6040
```

## Right camera

```text
/stereo/frame_right/image_raw/compressed
```

Message type:

```text
sensor_msgs/msg/CompressedImage
```

Message count:

```text
6040
```

Camera calibration:

```text
/stereo/frame_right/camera_info
```

Message type:

```text
sensor_msgs/msg/CameraInfo
```

Message count:

```text
6040
```

Approximate camera frequency:

```text
6040 / 302 ≈ 20 Hz
```

These frame cameras will be the primary image sources for semantic perception and camera–LiDAR fusion.

---

# 6. Davis Camera

The dataset also contains a Davis event-camera system.

The right Davis camera provides conventional image data:

```text
/stereo/davis_right/image_raw/compressed
```

Type:

```text
sensor_msgs/msg/CompressedImage
```

Message count:

```text
6038
```

Camera calibration:

```text
/stereo/davis_right/camera_info
```

Type:

```text
sensor_msgs/msg/CameraInfo
```

Message count:

```text
6040
```

The Davis camera also provides event data:

```text
/stereo/davis_right/events
/stereo/davis_left/events
```

Type:

```text
dvs_msgs/msg/EventArray
```

Message counts:

```text
/stereo/davis_right/events → 9060
/stereo/davis_left/events  → 2891
```

For the initial 3D semantic reconstruction system, event-camera data will be treated as optional.

The primary pipeline will initially use:

```text
Frame Camera + LiDAR
```

The event camera can be investigated in a later extension.

---

# 7. Camera Chunk Data

The bag contains:

```text
/stereo/frame_left/image_chunk_data
/stereo/frame_right/image_chunk_data
/stereo/davis_right/image_chunk_data
```

with:

```text
geometry_msgs/msg/PointStamped
```

These topics contain approximately one message per frame-camera sample.

They should be investigated during dataset exploration to determine their exact purpose and whether they contain useful timestamp/chunk synchronization information.

They will not initially be treated as primary perception inputs.

---

# 8. IMU Sensors

The dataset contains multiple IMU streams.

## STIM300 IMU

```text
/stim300/imu/data_raw
```

Type:

```text
sensor_msgs/msg/Imu
```

Messages:

```text
60390
```

Approximate frequency:

```text
60390 / 302 ≈ 200 Hz
```

---

## Unitree IMU

```text
/unitree/imu/data_raw
```

Type:

```text
sensor_msgs/msg/Imu
```

Messages:

```text
14911
```

Approximate frequency:

```text
14911 / 302 ≈ 49 Hz
```

---

## Davis IMU

```text
/stereo/davis_right/imu/data_raw
```

Type:

```text
sensor_msgs/msg/Imu
```

Messages:

```text
300969
```

Approximate frequency:

```text
300969 / 302 ≈ 1000 Hz
```

---

## LiDAR/OS IMU

```text
/os_cloud_node/imu/data_raw
```

Type:

```text
sensor_msgs/msg/Imu
```

Messages:

```text
30196
```

Approximate frequency:

```text
30196 / 302 ≈ 100 Hz
```

---

# 9. Robot State

The dataset contains several topics related to the quadruped's state.

## Body odometry

```text
/unitree/body_odom
```

Type:

```text
nav_msgs/msg/Odometry
```

Messages:

```text
14911
```

Approximate frequency:

```text
~49 Hz
```

This is potentially important for:

* Robot pose
* Motion estimation
* World-frame transformation
* Sensor observation accumulation
* Mapping experiments

---

## Joint state

```text
/unitree/joint_state
```

Type:

```text
sensor_msgs/msg/JointState
```

Messages:

```text
14911
```

This provides robot joint information.

It may become useful for understanding the quadruped's kinematic configuration and potentially investigating body/sensor motion.

---

## Foot state

```text
/unitree/foot_state
```

Type:

```text
trajectory_msgs/msg/MultiDOFJointTrajectory
```

Messages:

```text
14911
```

This may be useful later for studying:

* Foot contact
* Legged locomotion
* Body motion
* Terrain interaction

It is not required for the first version of the world reconstruction pipeline.

---

## Unitree robot state

```text
/unitree/robot_state
```

Type:

```text
unitree_legged_msgs/msg/HighState
```

Messages:

```text
14911
```

This provides additional low-level robot state information.

---

# 10. TF Information

The bag contains:

```text
/tf_static
```

Type:

```text
tf2_msgs/msg/TFMessage
```

Message count:

```text
90600
```

This topic is especially important because the project depends heavily on correct relationships between:

```text
LiDAR
Camera
Robot body
Other sensor frames
```

The static transformations determine the spatial relationship between these coordinate frames.

However, the recorded `/tf_static` data does not currently provide the replay behavior we want for this project.

Therefore, the bag will be replayed using a controlled static-TF remapping and republishing mechanism.

---

# 11. Static TF Replay Strategy

## Problem

The recorded topic is:

```text
/tf_static
```

However, ROS 2 static transforms require appropriate QoS behavior, particularly transient-local durability, so that late subscribers can receive previously published static transforms.

For this project, we will not directly depend on the recorded `/tf_static` publisher behavior.

Instead, the recorded topic will be renamed during playback:

```text
/tf_static → /tf_static_raw
```

and a dedicated node will subscribe to:

```text
/tf_static_raw
```

and republish corrected static transforms on:

```text
/tf_static
```

The resulting architecture is:

```text
                     ROS BAG
                       │
                       │ recorded
                       ▼
                  /tf_static
                       │
                       │ ros2 bag play remap
                       ▼
                 /tf_static_raw
                       │
                       ▼
          StaticTFRepublisher Node
                       │
                       │ StaticTransformBroadcaster
                       ▼
                  /tf_static
                       │
                       ▼
                  TF2 System
```

---

# 12. Static TF Republisher

The project uses a dedicated static TF republisher.

The node:

```text
static_tf_republisher
```

subscribes to:

```text
/tf_static_raw
```

and republishes the transforms using:

```text
tf2_ros.StaticTransformBroadcaster
```

The subscriber intentionally uses:

```text
BEST_EFFORT
VOLATILE
```

to be permissive when reading the recorded bag.

The republished transforms are sent through the standard ROS 2 static transform mechanism.

The node also maintains a set of previously observed transforms and avoids repeatedly publishing identical transforms.

---

# 13. Why the Remapping Is Important

The recorded topic must **not** simply be subscribed to and republished on the same topic.

Incorrect:

```text
/tf_static
    │
    ▼
Node
    │
    ▼
/tf_static
    │
    └──────→ loop
```

Instead:

```text
Recorded bag:

/tf_static
     │
     │ remap
     ▼
/tf_static_raw
     │
     ▼
Republisher
     │
     ▼
/tf_static
```

This avoids a publish/subscribe loop.

---

# 14. Bag Playback

The bag should be played with:

```bash
ros2 bag play grass.db3 \
  --remap /tf_static:=/tf_static_raw
```

The static TF republisher should be running separately:

```bash
ros2 run <package_name> static_tf_republisher
```

The exact package name will be defined in the project workspace.

The intended runtime architecture is:

```text
Terminal 1
──────────

ros2 bag play grass.db3 \
    --remap /tf_static:=/tf_static_raw


Terminal 2
──────────

ros2 run <package_name> static_tf_republisher


Terminal 3
──────────

ros2 topic echo /tf_static


Terminal 4
──────────

ros2 run <our_processing_node>
```

---

# 15. Primary Topics for This Project

Not every topic in the dataset is required for the initial implementation.

The core pipeline will initially use:

| Purpose            | Topic                                      | Type                              |
| ------------------ | ------------------------------------------ | --------------------------------- |
| RGB                | `/stereo/frame_left/image_raw/compressed`  | `sensor_msgs/msg/CompressedImage` |
| RGB                | `/stereo/frame_right/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` |
| Camera calibration | `/stereo/frame_left/camera_info`           | `sensor_msgs/msg/CameraInfo`      |
| Camera calibration | `/stereo/frame_right/camera_info`          | `sensor_msgs/msg/CameraInfo`      |
| LiDAR              | `/os_cloud_node/points`                    | `sensor_msgs/msg/PointCloud2`     |
| Robot pose         | `/unitree/body_odom`                       | `nav_msgs/msg/Odometry`           |
| Static transforms  | `/tf_static`                               | `tf2_msgs/msg/TFMessage`          |
| IMU                | `/stim300/imu/data_raw`                    | `sensor_msgs/msg/Imu`             |

The initial reconstruction pipeline therefore becomes:

```text
                     Camera
                       │
                       ▼
              Semantic Perception
                       │
                       │
LiDAR ────────────────┼─────────────── Robot Pose
  │                   │                    │
  ▼                   ▼                    ▼
3D Geometry      Camera–LiDAR          World Pose
                  Association              │
  │                   │                    │
  └───────────────────┼────────────────────┘
                      ▼
              Semantic 3D Fusion
                      │
                      ▼
              World Reconstruction
                      │
                      ▼
              Persistent 3D Map
```

---

# 16. Secondary Topics

The following topics will be retained for future experiments:

```text
/stereo/davis_right/events
/stereo/davis_left/events

/stereo/davis_right/image_raw/compressed
/stereo/davis_right/camera_info

/stereo/davis_right/imu/data_raw

/os_cloud_node/imu/data_raw
/os_image_node/range_image
/os_image_node/nearir_image
/os_image_node/signal_image
/os_image_node/reflec_image

/unitree/joint_state
/unitree/foot_state
/unitree/robot_state
/unitree/imu/data_raw
```

These can support future research into:

* Event-based perception
* Multi-IMU fusion
* Legged-state estimation
* Terrain interaction
* LiDAR intensity/reflectivity
* Multi-modal perception

---

# 17. Approximate Sensor Frequencies

Based on the bag duration and message counts:

| Sensor / Topic          | Messages | Approx. Frequency |
| ----------------------- | -------: | ----------------: |
| STIM300 IMU             |   60,390 |           ~200 Hz |
| Davis IMU               |  300,969 |          ~1000 Hz |
| LiDAR IMU               |   30,196 |           ~100 Hz |
| Unitree IMU             |   14,911 |            ~49 Hz |
| Unitree body odometry   |   14,911 |            ~49 Hz |
| Unitree joint state     |   14,911 |            ~49 Hz |
| Unitree foot state      |   14,911 |            ~49 Hz |
| Unitree robot state     |   14,911 |            ~49 Hz |
| Frame left image        |    6,040 |            ~20 Hz |
| Frame right image       |    6,040 |            ~20 Hz |
| Frame left camera info  |    6,040 |            ~20 Hz |
| Frame right camera info |    6,040 |            ~20 Hz |
| LiDAR point cloud       |    3,020 |            ~10 Hz |
| LiDAR range image       |    3,020 |            ~10 Hz |
| LiDAR near-IR           |    3,020 |            ~10 Hz |
| LiDAR signal            |    3,020 |            ~10 Hz |
| LiDAR reflectivity      |    3,020 |            ~10 Hz |

These frequencies are approximate and are calculated from:

```text
message_count / bag_duration
```

They should be verified from actual message timestamps before being used for synchronization or performance assumptions.

---

# 18. Initial Sensor Synchronization Model

The primary sensor streams have different frequencies:

```text
Camera       ~20 Hz
LiDAR        ~10 Hz
Body odom    ~49 Hz
STIM300      ~200 Hz
Davis IMU    ~1000 Hz
```

Therefore, the project cannot simply assume one-to-one message correspondence.

Instead, the reconstruction pipeline will use timestamps.

For each LiDAR frame:

```text
LiDAR timestamp
       │
       ├── Find nearest camera frame
       │
       ├── Find corresponding robot pose
       │
       └── Find relevant TF
```

Conceptually:

```text
LiDAR(t)
   │
   ├──────────── Camera(t ± Δt)
   │
   ├──────────── Pose(t ± Δt)
   │
   └──────────── TF(t)
```

The synchronization error will eventually be measured and documented.

---

# 19. Coordinate-Frame Investigation

Before Phase 4, we must explicitly document the coordinate frames.

The following relationships need to be extracted from the bag:

```text
LiDAR frame
Camera frame
Robot/body frame
IMU frame
World/odom frame
```

We should not assume frame names or transform directions from topic names alone.

Phase 0 will therefore include:

```bash
ros2 topic echo /tf_static
```

and TF inspection using appropriate ROS 2 tools.

The resulting TF tree will be documented in:

```text
docs/coordinate_frames.md
```

---

# 20. Calibration Data

Camera calibration is available through:

```text
/stereo/frame_left/camera_info
/stereo/frame_right/camera_info
/stereo/davis_right/camera_info
```

These `sensor_msgs/msg/CameraInfo` messages contain the camera model information required to understand:

* Image dimensions
* Intrinsic matrix
* Distortion parameters
* Rectification information
* Projection matrix

The provided calibration will be treated as a **reference calibration**.

Importantly, this project will independently investigate and estimate camera–LiDAR calibration.

Therefore:

```text
Provided Calibration
        │
        │ reference
        ▼
Compare
        ▲
        │
Our Calibration
```

The goal is not simply to use the provided calibration.

The goal is to understand, estimate, validate, and compare calibration.

---

# 21. Phase 0 Data Flow

The complete Phase 0 understanding is:

```text
                   grass.db3
                       │
                       ▼
                 ROS 2 Bag
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
       Camera        LiDAR         Robot
          │            │            │
          ▼            ▼            ▼
    CameraInfo     PointCloud2    Odometry
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
                  TF / Frames
                       │
                       ▼
                Synchronization
                       │
                       ▼
             Dataset Interface
                       │
                       ▼
              Phase 1 Processing
```

---

# 22. Phase 0 Validation Checklist

Before moving to Phase 1, the following must be verified.

### Dataset

* [x] ROS 2 bag identified
* [x] SQLite3 storage identified
* [x] Bag duration identified
* [x] Message count identified
* [x] Primary sensor topics identified

### LiDAR

* [x] PointCloud2 topic identified
* [x] LiDAR frequency approximately established
* [ ] Point fields inspected
* [ ] Point coordinate frame verified
* [ ] Point cloud visualization verified

### Cameras

* [x] Left camera identified
* [x] Right camera identified
* [x] CameraInfo topics identified
* [ ] Intrinsic parameters extracted
* [ ] Distortion model identified
* [ ] Image resolution verified

### Robot State

* [x] Body odometry identified
* [x] IMU streams identified
* [ ] Odometry frame verified
* [ ] Pose convention verified

### TF

* [x] `/tf_static` identified
* [x] Static TF remapping strategy defined
* [ ] TF tree inspected
* [ ] Sensor-frame relationships documented

### Synchronization

* [x] Sensor frequencies approximately identified
* [ ] Actual timestamp distributions inspected
* [ ] Camera–LiDAR synchronization verified
* [ ] LiDAR–pose synchronization verified

### Calibration

* [x] Camera calibration topics identified
* [x] Dataset calibration treated as reference
* [ ] Camera intrinsics extracted
* [ ] Camera–LiDAR transform identified
* [ ] Independent calibration pipeline planned

---

# 23. Phase 0 Deliverables

At the end of Phase 0, the project should contain:

```text
docs/
├── dataset.md
├── coordinate_frames.md
└── synchronization.md
```

and a reproducible bag replay setup:

```text
scripts/
└── play_grass.sh
```

along with:

```text
static_tf_republisher
```

for reliable static TF publication.

---

# 24. Transition to Phase 1

Once Phase 0 is complete, the next phase will begin with the LiDAR data.

The first processing pipeline will be:

```text
/os_cloud_node/points
          │
          ▼
      PointCloud2
          │
          ▼
   ROS → NumPy/Open3D
          │
          ▼
    Inspect fields
          │
          ▼
    Remove invalid points
          │
          ▼
    Range filtering
          │
          ▼
    Voxel downsampling
          │
          ▼
       Visualization
```

Phase 1 will then introduce:

* Open3D
* PCL concepts
* PointCloud2 structure
* Coordinate frames
* Voxel grids
* KD-trees
* Outlier filtering
* Point-cloud visualization

The objective is to establish a strong geometric foundation before introducing semantic perception or camera–LiDAR fusion.
