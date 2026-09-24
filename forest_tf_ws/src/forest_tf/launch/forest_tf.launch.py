from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='forest_tf',
            executable='static_tf_republisher',
            name='static_tf_republisher',
            output='screen',
        ),
        Node(
            package='forest_tf',
            executable='forest_tf_node',
            name='forest_tf_node',
            output='screen',
            parameters=[{
                'odom_topic': '/unitree/body_odom',
                'world_frame': 'world',
                'body_frame': 'body_imu',
                'sensor_frame': 'ouster00',
                'use_header_frame_as_parent': False,
            }],
        ),
    ])
