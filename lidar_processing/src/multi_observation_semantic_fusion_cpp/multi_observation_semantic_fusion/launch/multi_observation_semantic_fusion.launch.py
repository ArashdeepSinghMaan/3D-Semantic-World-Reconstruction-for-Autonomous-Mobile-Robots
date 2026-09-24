from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    cfg=os.path.join(get_package_share_directory('multi_observation_semantic_fusion'),'config','multi_observation_semantic_fusion.yaml')
    return LaunchDescription([Node(package='multi_observation_semantic_fusion',executable='multi_observation_semantic_fusion_node',name='multi_observation_semantic_fusion',output='screen',parameters=[cfg])])
