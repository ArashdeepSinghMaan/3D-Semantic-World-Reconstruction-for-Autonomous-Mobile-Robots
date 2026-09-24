"""ROS 2 node that reports what a PointCloud2 stream actually contains.

Logs the field layout of the first message, then periodically reports:
header-stamp rate / jitter / gaps, receive latency (meaningful only with
``use_sim_time`` and ``ros2 bag play --clock``), invalid / zero point counts,
range distribution and per-point time span. Nothing is assumed about the
driver beyond x, y, z.
"""

from collections import deque

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from .analysis import (
    aggregate_quality,
    cloud_quality,
    format_table,
    offset_statistics,
    stamp_statistics,
    stamp_to_sec,
)
from .pointcloud_utils import describe_layout, pointcloud2_to_array


class PointCloudInspector(Node):

    def __init__(self):
        super().__init__('pointcloud_inspector')
        self.declare_parameter('topic', '/os_cloud_node/points')
        self.declare_parameter('report_period_s', 5.0)
        self.declare_parameter('input_qos_reliability', 'best_effort')
        self.declare_parameter('quality_every_n', 1)
        self.declare_parameter('history', 3000)

        history = int(self.get_parameter('history').value)
        self._stamps = deque(maxlen=history)
        self._latencies = deque(maxlen=history)
        self._qualities = deque(maxlen=200)
        self._frame_ids = set()
        self._received = 0
        self._layout_logged = False
        self._quality_every = max(1, int(self.get_parameter('quality_every_n').value))

        reliability = str(self.get_parameter('input_qos_reliability').value).lower()
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=(ReliabilityPolicy.RELIABLE if reliability == 'reliable'
                         else ReliabilityPolicy.BEST_EFFORT),
            durability=DurabilityPolicy.VOLATILE,
        )
        topic = self.get_parameter('topic').value
        self.create_subscription(PointCloud2, topic, self._on_cloud, qos)
        self.create_timer(float(self.get_parameter('report_period_s').value), self._report)
        self.get_logger().info(f'Inspecting {topic}')

    def _on_cloud(self, msg):
        self._received += 1
        if not self._layout_logged:
            self.get_logger().info('\n' + describe_layout(msg))
            self._layout_logged = True
        self._frame_ids.add(msg.header.frame_id)

        stamp = stamp_to_sec(msg.header.stamp)
        self._stamps.append(stamp)
        now_ns = self.get_clock().now().nanoseconds
        if now_ns > 0:
            self._latencies.append(now_ns * 1e-9 - stamp)

        if self._received % self._quality_every == 0:
            try:
                self._qualities.append(cloud_quality(pointcloud2_to_array(msg)))
            except ValueError as exc:
                self.get_logger().error(f'Could not decode cloud: {exc}',
                                        throttle_duration_sec=5.0)

    def _report(self):
        if self._received == 0:
            self.get_logger().info('No messages received yet', throttle_duration_sec=30.0)
            return
        sections = [
            f'Messages received: {self._received}   frame_ids: {sorted(self._frame_ids)}',
            format_table('Header stamp statistics', stamp_statistics(list(self._stamps))),
        ]
        if self._latencies:
            sections.append(format_table(
                'Receive time - header stamp (clock-dependent)',
                offset_statistics(list(self._latencies))))
        if self._qualities:
            sections.append(format_table(
                f'Cloud quality over last {len(self._qualities)} sampled frames',
                aggregate_quality(list(self._qualities))))
        self.get_logger().info('\n' + '\n\n'.join(sections))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PointCloudInspector()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node._report()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
