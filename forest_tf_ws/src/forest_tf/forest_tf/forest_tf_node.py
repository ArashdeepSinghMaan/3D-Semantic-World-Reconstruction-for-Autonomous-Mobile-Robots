#!/usr/bin/env python3
"""
Publish the dynamic TF for the Forest dataset.

Input:
  /unitree/body_odom : nav_msgs/msg/Odometry

The recorded odometry has:
  header.frame_id = unitree_body_odom
  child_frame_id  = empty

We interpret the pose as the pose of body_imu in the odometry/world
coordinate system and publish:

  world (or configured odom_frame) -> ouster00

using the calibrated fixed transform:

  body_imu -> ouster00

The existing static tree is then used to reach the cameras and body_imu.
"""

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


# Calibration obtained by composing the recorded static transforms:
# ouster00 -> frame_cam00 -> body_imu
# and then inverting to obtain body_imu -> ouster00.
BODY_TO_OUSTER_T = (0.03595682, -0.01764054, 0.03291798)
BODY_TO_OUSTER_Q = (-0.0006598046, 0.0018554325, -0.0041559715, 0.999989425)


def quat_to_matrix(x, y, z, w):
    """Return a 3x3 rotation matrix without numpy."""
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return (
        (1 - 2*(yy+zz), 2*(xy-wz),     2*(xz+wy)),
        (2*(xy+wz),     1 - 2*(xx+zz), 2*(yz-wx)),
        (2*(xz-wy),     2*(yz+wx),     1 - 2*(xx+yy)),
    )


def mat_to_quat(m):
    """Convert a 3x3 rotation matrix to x,y,z,w quaternion."""
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return x, y, z, w


def matmul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def matvec(a, v):
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


class ForestTFNode(Node):
    def __init__(self):
        super().__init__('forest_tf_node')

        self.declare_parameter('odom_topic', '/unitree/body_odom')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('body_frame', 'body_imu')
        self.declare_parameter('sensor_frame', 'ouster00')
        self.declare_parameter('use_header_frame_as_parent', False)

        self.odom_topic = self.get_parameter('odom_topic').value
        self.world_frame = self.get_parameter('world_frame').value
        self.body_frame = self.get_parameter('body_frame').value
        self.sensor_frame = self.get_parameter('sensor_frame').value
        self.use_header_frame = self.get_parameter(
            'use_header_frame_as_parent').value

        self.tf_broadcaster = TransformBroadcaster(self)
        self.count = 0

        self.sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            20
        )

        self.get_logger().info(
            f'Listening: {self.odom_topic} '
            f'({self.world_frame}/{self.sensor_frame})'
        )
        self.get_logger().info(
            'Interpreting /unitree/body_odom pose as the pose of body_imu.'
        )

    def odom_callback(self, msg: Odometry):
        # Recorded odometry:
        #   p_world_body, q_world_body
        p = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        )
        q = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )

        # Compose:
        # T_world_ouster = T_world_body * T_body_ouster
        r_wb = quat_to_matrix(*q)
        r_bo = quat_to_matrix(*BODY_TO_OUSTER_Q)

        r_wo = matmul(r_wb, r_bo)
        rotated_offset = matvec(r_wb, BODY_TO_OUSTER_T)

        p_wo = tuple(p[i] + rotated_offset[i] for i in range(3))
        q_wo = mat_to_quat(r_wo)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = (
            msg.header.frame_id if self.use_header_frame
            and msg.header.frame_id else self.world_frame
        )
        tf_msg.child_frame_id = self.sensor_frame

        tf_msg.transform.translation.x = p_wo[0]
        tf_msg.transform.translation.y = p_wo[1]
        tf_msg.transform.translation.z = p_wo[2]

        tf_msg.transform.rotation.x = q_wo[0]
        tf_msg.transform.rotation.y = q_wo[1]
        tf_msg.transform.rotation.z = q_wo[2]
        tf_msg.transform.rotation.w = q_wo[3]

        self.tf_broadcaster.sendTransform(tf_msg)

        self.count += 1
        if self.count == 1:
            self.get_logger().info(
                f'Publishing {tf_msg.header.frame_id} -> '
                f'{tf_msg.child_frame_id}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ForestTFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
