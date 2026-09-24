"""Launch Phase 2 geometric analysis, optionally with Phase 1 preprocessing and RViz2.

    ros2 launch geometric_pointcloud_analysis geometric_analysis.launch.py \
        with_phase1:=true rviz:=true fixed_frame:=<cloud frame_id>

Play the bag with --clock:  ros2 bag play <bag> --clock --remap /tf_static:=/tf_static_raw
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare('geometric_pointcloud_analysis')
    phase1_share = FindPackageShare('lidar_pointcloud_processing')
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)

    arguments = [
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution([share, 'config', 'geometric_analysis.yaml'])),
        DeclareLaunchArgument(
            'phase1_config_file',
            default_value=PathJoinSubstitution(
                [phase1_share, 'config', 'pointcloud_processing.yaml'])),
        DeclareLaunchArgument('with_phase1', default_value='true',
                              description='Also start the Phase 1 preprocessing node'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('fixed_frame', default_value='os_sensor'),
        DeclareLaunchArgument('log_level', default_value='info'),
    ]

    phase1 = Node(
        package='lidar_pointcloud_processing',
        executable='pointcloud_processing_node',
        name='pointcloud_processing',
        output='screen',
        parameters=[LaunchConfiguration('phase1_config_file'), {'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('with_phase1')),
    )
    geometry = Node(
        package='geometric_pointcloud_analysis',
        executable='geometric_analysis_node',
        name='geometric_analysis',
        output='screen',
        parameters=[LaunchConfiguration('config_file'), {'use_sim_time': use_sim_time}],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([share, 'rviz', 'geometric_analysis.rviz']),
                   '-f', LaunchConfiguration('fixed_frame')],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )
    return LaunchDescription(arguments + [phase1, geometry, rviz])
