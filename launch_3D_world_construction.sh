#!/bin/bash

# ============================================================
# 3D Semantic World Reconstruction
# Phase 9 - Full Pipeline Launcher
# ============================================================

set -e

WORKSPACE="/media/NewVolume/Quadruped/lidar_processing"

echo "============================================================"
echo "  3D Semantic World Reconstruction"
echo "  Phase 9 - Full Pipeline"
echo "============================================================"

# ------------------------------------------------------------
# Source ROS 2
# ------------------------------------------------------------

source /opt/ros/humble/setup.bash

# ------------------------------------------------------------
# Source workspace
# ------------------------------------------------------------

if [ ! -f "$WORKSPACE/install/setup.bash" ]; then
    echo "[ERROR] Workspace has not been built."
    echo "Run:"
    echo "  cd $WORKSPACE"
    echo "  colcon build --symlink-install"
    exit 1
fi

source "$WORKSPACE/install/setup.bash"

echo "[INFO] ROS 2 Humble sourced"
echo "[INFO] Workspace sourced"
echo ""

# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------

launch_node()
{
    PACKAGE=$1
    EXECUTABLE=$2
    TITLE=$3

    echo "[START] $PACKAGE -> $EXECUTABLE"

    gnome-terminal \
        --title="$TITLE" \
        -- bash -c "
            source /opt/ros/humble/setup.bash
            source $WORKSPACE/install/setup.bash

            echo '================================================'
            echo ' $TITLE'
            echo '================================================'
            echo ''

            ros2 run $PACKAGE $EXECUTABLE

            echo ''
            echo '================================================'
            echo ' $TITLE stopped'
            echo '================================================'
            read
        "
}

# ============================================================
# PIPELINE
# ============================================================

echo ""
echo "Starting Phase 9 pipeline..."
echo ""

# ------------------------------------------------------------
# Phase 1 / World reconstruction
# ------------------------------------------------------------

launch_node \
    world_frame_reconstruction \
    world_frame_reconstruction_node \
    "World Frame Reconstruction"

sleep 1


# ------------------------------------------------------------
# Camera-LiDAR calibration / projection
# ------------------------------------------------------------

launch_node \
    camera_lidar_calibration \
    camera_lidar_projection_node \
    "Camera LiDAR Projection"

sleep 1


# ------------------------------------------------------------
# Semantic perception
# ------------------------------------------------------------

launch_node \
    semantic_perception \
    semantic_perception_node \
    "Semantic Perception"

sleep 1


# ------------------------------------------------------------
# Semantic 3D fusion
# ------------------------------------------------------------

launch_node \
    semantic_3d_fusion \
    semantic_3d_fusion_node \
    "Semantic 3D Fusion"

sleep 1


# ------------------------------------------------------------
# Multi-observation semantic fusion
# ------------------------------------------------------------

launch_node \
    multi_observation_semantic_fusion \
    multi_observation_semantic_fusion_node \
    "Multi Observation Semantic Fusion"

sleep 1


# ------------------------------------------------------------
# Persistent semantic mapping
# ------------------------------------------------------------

launch_node \
    persistent_semantic_mapping \
    persistent_semantic_mapping_node \
    "Persistent Semantic Mapping"

sleep 1


# ------------------------------------------------------------
# Point cloud registration
# ------------------------------------------------------------

launch_node \
    pointcloud_registration \
    pointcloud_registration_node \
    "Point Cloud Registration"

sleep 1


# ------------------------------------------------------------
# Terrain reasoning
# ------------------------------------------------------------

launch_node \
    terrain_reasoning \
    negative_obstacle_detector \
    "Negative Obstacle Detector"


echo ""
echo "============================================================"
echo "  All Phase 9 nodes launched"
echo "============================================================"
echo ""
echo "Use the individual terminals to monitor each node."
echo "Close the terminals to stop individual nodes."
echo ""
