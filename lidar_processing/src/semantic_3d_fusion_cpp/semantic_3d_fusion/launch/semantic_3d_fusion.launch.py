from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('semantic_3d_fusion')
    default_config = os.path.join(pkg, 'config', 'semantic_3d_fusion.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config),
        Node(
            package='semantic_3d_fusion',
            executable='semantic_fusion_node',
            name='semantic_3d_fusion',
            output='screen',
            parameters=[LaunchConfiguration('config')],
        ),
    ])
