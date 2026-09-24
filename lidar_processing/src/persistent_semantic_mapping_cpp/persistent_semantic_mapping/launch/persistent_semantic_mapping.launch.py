from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory('persistent_semantic_mapping')
    return LaunchDescription([Node(package='persistent_semantic_mapping', executable='persistent_semantic_mapping_node', name='persistent_semantic_mapping', output='screen', parameters=[os.path.join(pkg, 'config', 'persistent_semantic_mapping.yaml')])])
