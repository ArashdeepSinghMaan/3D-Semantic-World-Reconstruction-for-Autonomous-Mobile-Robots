from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("pointcloud_registration"),
        "config",
        "registration.yaml",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=default_config,
            description="Path to registration YAML",
        ),
        Node(
            package="pointcloud_registration",
            executable="registration_node",
            name="pointcloud_registration",
            output="screen",
            parameters=[LaunchConfiguration("config_file")],
        ),
    ])
