from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("camera_lidar_calibration"),
        "config",
        "calibration.yaml",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=default_config,
            description="Camera-LiDAR calibration configuration",
        ),

        Node(
            package="camera_lidar_calibration",
            executable="camera_lidar_projection_node",
            name="camera_lidar_projection",
            output="screen",
            parameters=[LaunchConfiguration("config_file")],
        ),
    ])
