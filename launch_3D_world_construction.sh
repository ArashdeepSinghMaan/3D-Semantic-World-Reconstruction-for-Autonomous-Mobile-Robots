#!/usr/bin/env bash

set -euo pipefail

SESSION_NAME="semantic_world"
WORKSPACE_DIR="/media/hitech/NewVolume/Quadruped/lidar_processing"
SETUP_CMD="source /opt/ros/humble/setup.bash && source $WORKSPACE_DIR/install/setup.bash"

# ------------------------------------------------------------
# Kill existing session
# ------------------------------------------------------------

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing session '$SESSION_NAME'..."
    tmux kill-session -t "$SESSION_NAME"
fi

# ------------------------------------------------------------
# Create tmux session
# ------------------------------------------------------------

tmux new-session -d -s "$SESSION_NAME" -n nodes

# ------------------------------------------------------------
# Create panes
# ------------------------------------------------------------

attempt_split() {
    if ! tmux split-window -t "$SESSION_NAME:nodes" -l 10 2>/dev/null; then
        echo "Vertical split failed, trying horizontal split..."
        if ! tmux split-window -t "$SESSION_NAME:nodes" -h -l 20 2>/dev/null; then
            echo "Warning: Cannot split further."
            return 1
        fi
    fi

    return 0
}

# 8 nodes -> 8 panes
for i in {1..7}; do
    attempt_split || break
done

# ------------------------------------------------------------
# Tiled layout
# ------------------------------------------------------------

tmux select-layout -t "$SESSION_NAME:nodes" tiled

# ------------------------------------------------------------
# Phase-9 ROS2 nodes
# ------------------------------------------------------------

commands=(
    "ros2 run world_frame_reconstruction world_frame_reconstruction_node"
    "ros2 run camera_lidar_calibration camera_lidar_projection_node"
    "ros2 run semantic_perception semantic_perception_node"
    "ros2 run semantic_3d_fusion semantic_3d_fusion_node"
    "ros2 run multi_observation_semantic_fusion multi_observation_semantic_fusion_node"
    "ros2 run persistent_semantic_mapping persistent_semantic_mapping_node"
    "ros2 run pointcloud_registration pointcloud_registration_node"
    "ros2 run terrain_reasoning negative_obstacle_detector"
)

# ------------------------------------------------------------
# Start each node in its own pane
# ------------------------------------------------------------

for idx in "${!commands[@]}"; do

    if tmux list-panes \
        -t "$SESSION_NAME:nodes" \
        -F "#{pane_index}" 2>/dev/null |
        grep -qx "$idx"
    then

        tmux send-keys \
            -t "$SESSION_NAME:nodes.$idx" \
            "$SETUP_CMD && ${commands[$idx]}" \
            C-m

        echo "Started: ${commands[$idx]}"

    else

        echo "Pane $idx does not exist:"
        echo "  ${commands[$idx]}"

    fi

done

# ------------------------------------------------------------
# Enable mouse support
# ------------------------------------------------------------

tmux set-option -g mouse on

# ------------------------------------------------------------
# Attach
# ------------------------------------------------------------

tmux select-window -t "$SESSION_NAME:nodes"

tmux attach-session -t "$SESSION_NAME"
