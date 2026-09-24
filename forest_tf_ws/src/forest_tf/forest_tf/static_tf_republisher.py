#!/usr/bin/env python3
"""
Repair/re-publish recorded /tf_static with ROS2 static-TF QoS.

IMPORTANT:
The input topic is intentionally /tf_static_raw.

When playing the bag, remap:
    /tf_static:=/tf_static_raw

This avoids a publish/subscribe loop because this node publishes the
corrected transforms on /tf_static.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster


class StaticTFRepublisher(Node):
    def __init__(self):
        super().__init__('static_tf_republisher')

        self.input_topic = '/tf_static_raw'
        self.broadcaster = StaticTransformBroadcaster(self)
        self.seen = set()
        self.received = 0

        # Volatile + BEST_EFFORT is intentionally permissive for recorded
        # bags whose /tf_static publisher has non-standard QoS.
        input_qos = QoSProfile(depth=100)
        input_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        input_qos.durability = DurabilityPolicy.VOLATILE

        self.sub = self.create_subscription(
            TFMessage,
            self.input_topic,
            self.callback,
            input_qos
        )

        self.get_logger().info(
            f'Subscribing to {self.input_topic} and republishing on /tf_static'
        )

    def callback(self, msg: TFMessage):
        new_transforms = []

        for tf in msg.transforms:
            key = (
                tf.header.frame_id,
                tf.child_frame_id,
                round(tf.transform.translation.x, 9),
                round(tf.transform.translation.y, 9),
                round(tf.transform.translation.z, 9),
                round(tf.transform.rotation.x, 9),
                round(tf.transform.rotation.y, 9),
                round(tf.transform.rotation.z, 9),
                round(tf.transform.rotation.w, 9),
            )

            if key not in self.seen:
                self.seen.add(key)
                new_transforms.append(tf)

        if new_transforms:
            self.broadcaster.sendTransform(new_transforms)
            self.received += len(new_transforms)

            self.get_logger().info(
                f'Published {len(new_transforms)} new static transforms '
                f'(total unique: {self.received})'
            )


def main(args=None):
    rclpy.init(args=args)
    node = StaticTFRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
