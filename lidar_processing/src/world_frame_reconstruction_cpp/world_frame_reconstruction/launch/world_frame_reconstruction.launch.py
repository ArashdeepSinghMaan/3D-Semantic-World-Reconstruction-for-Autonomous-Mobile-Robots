from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('world_frame_reconstruction')
    config = os.path.join(pkg, 'config', 'world_frame_reconstruction.yaml')
    return LaunchDescription([
        Node(
            package='world_frame_reconstruction',
            executable='world_frame_reconstruction_node',
            name='world_frame_reconstruction',
            output='screen',
            parameters=[config],
        )
    ])
