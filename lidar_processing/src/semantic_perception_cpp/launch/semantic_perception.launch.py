from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("semantic_perception"),
        "config",
        "semantic_perception.yaml",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=default_config,
            description="Semantic perception configuration YAML",
        ),

        Node(
            package="semantic_perception",
            executable="semantic_perception_node",
            name="semantic_perception",
            output="screen",
            parameters=[LaunchConfiguration("config_file")],
        ),
    ])
