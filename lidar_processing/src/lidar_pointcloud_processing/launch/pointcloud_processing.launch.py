"""Launch the point-cloud processing node, optionally with the inspector and RViz2.

    ros2 launch lidar_pointcloud_processing pointcloud_processing.launch.py \
        rviz:=true fixed_frame:=<frame_id from the layout log>

Play the bag with --clock so use_sim_time works:

    ros2 bag play <bag> --clock --remap /tf_static:=/tf_static_raw
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare('lidar_pointcloud_processing')

    config_file = LaunchConfiguration('config_file')
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    log_level = LaunchConfiguration('log_level')

    arguments = [
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution(
                [package_share, 'config', 'pointcloud_processing.yaml']),
            description='Parameter YAML for the processing and inspector nodes'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use /clock from `ros2 bag play --clock`'),
        DeclareLaunchArgument(
            'inspector', default_value='false',
            description='Also run the PointCloud2 inspector node'),
        DeclareLaunchArgument(
            'rviz', default_value='false', description='Start RViz2'),
        DeclareLaunchArgument(
            'fixed_frame', default_value='os_sensor',
            description='RViz2 fixed frame; use the frame_id printed in the layout log'),
        DeclareLaunchArgument(
            'log_level', default_value='info', description='Node log level'),
    ]

    processing = Node(
        package='lidar_pointcloud_processing',
        executable='pointcloud_processing_node',
        name='pointcloud_processing',
        output='screen',
        parameters=[config_file, {'use_sim_time': use_sim_time}],
        arguments=['--ros-args', '--log-level', log_level],
    )

    inspector = Node(
        package='lidar_pointcloud_processing',
        executable='pointcloud_inspector',
        name='pointcloud_inspector',
        output='screen',
        parameters=[config_file, {'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('inspector')),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d', PathJoinSubstitution([package_share, 'rviz', 'pointcloud_processing.rviz']),
            '-f', LaunchConfiguration('fixed_frame'),
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription(arguments + [processing, inspector, rviz])
