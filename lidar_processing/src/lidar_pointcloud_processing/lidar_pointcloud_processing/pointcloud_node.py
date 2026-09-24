"""ROS 2 node: PointCloud2 in, filtered + voxelized PointCloud2 and diagnostics out.

The node only handles ROS I/O. All processing is delegated to
``PointCloudPipeline``, which has no ROS dependency.

Pipeline parameters (``filters.*``, ``range_filter.*``, ``crop_box.*``,
``voxel.*``, ``outlier.*``, ``output.*``) can be changed at runtime with
``ros2 param set``; they are validated as a whole before being applied.
Topic and QoS parameters are read once at startup.
"""

import time

from diagnostic_msgs.msg import DiagnosticArray
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from .diagnostics import build_diagnostic_array, RollingStats
from .pipeline import (
    config_from_dict,
    config_to_flat_dict,
    flat_to_nested,
    PipelineConfig,
    PointCloudPipeline,
)
from .pointcloud_utils import array_to_pointcloud2, describe_layout, pointcloud2_to_array

# name: (default, description, changeable at runtime)
NODE_PARAMETERS = {
    'input_topic': ('/os_cloud_node/points', 'Input sensor_msgs/PointCloud2 topic', False),
    'filtered_topic': ('/lidar/points/filtered',
                       'Full-resolution filtered cloud, all input fields preserved', False),
    'voxelized_topic': ('/lidar/points/voxelized', 'Voxel-downsampled cloud', False),
    'diagnostics_topic': ('/diagnostics', 'diagnostic_msgs/DiagnosticArray output', False),
    'input_qos_reliability': ('best_effort',
                              "Input QoS reliability: 'best_effort' or 'reliable'", False),
    'queue_depth': (1, 'Input queue depth. 1 drops stale frames instead of lagging', False),
    'publish_filtered': (True, 'Publish the filtered cloud', True),
    'publish_voxelized': (True, 'Publish the voxelized cloud', True),
    'publish_diagnostics': (True, 'Publish per-frame diagnostics', True),
    'log_every_n_frames': (50, 'Log a rolling summary every N frames (0 disables)', True),
    'stats_window': (100, 'Number of frames in the rolling statistics window', False),
    'processing_budget_ms': (100.0,
                             'Diagnostics WARN above this mean time (100 ms at 10 Hz)', True),
}


class PointCloudProcessingNode(Node):

    def __init__(self):
        super().__init__('pointcloud_processing')

        for name, (default, description, runtime) in NODE_PARAMETERS.items():
            suffix = '' if runtime else ' (read at startup)'
            self.declare_parameter(
                name, default, ParameterDescriptor(description=description + suffix))

        # Pipeline parameters are generated from the dataclass defaults so the
        # YAML, the node and the offline tools can never drift apart.
        # dynamic_typing lets YAML use 30 or 30.0 interchangeably; values are
        # coerced and validated by the pipeline configuration.
        self._pipeline_parameter_names = []
        for name, default in config_to_flat_dict(PipelineConfig()).items():
            self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))
            self._pipeline_parameter_names.append(name)

        config = self._build_config({})  # raises ValueError on invalid startup config
        self._pipeline = PointCloudPipeline(config)
        self._log_config(config)
        self._read_runtime_parameters({})

        self._rolling = RollingStats(window=self.get_parameter('stats_window').value)
        self._layout_logged = False
        self._missing_fields_warned = False

        reliability = str(self.get_parameter('input_qos_reliability').value).lower()
        if reliability not in ('best_effort', 'reliable'):
            raise ValueError(f"input_qos_reliability must be 'best_effort' or 'reliable', "
                             f'got {reliability!r}')
        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=int(self.get_parameter('queue_depth').value),
            reliability=(ReliabilityPolicy.BEST_EFFORT if reliability == 'best_effort'
                         else ReliabilityPolicy.RELIABLE),
            durability=DurabilityPolicy.VOLATILE,
        )
        # RELIABLE publishers match both RViz2's default (reliable) and
        # best-effort subscribers.
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._filtered_pub = self.create_publisher(
            PointCloud2, self.get_parameter('filtered_topic').value, output_qos)
        self._voxel_pub = self.create_publisher(
            PointCloud2, self.get_parameter('voxelized_topic').value, output_qos)
        self._diag_pub = self.create_publisher(
            DiagnosticArray, self.get_parameter('diagnostics_topic').value, 10)
        input_topic = self.get_parameter('input_topic').value
        self._subscription = self.create_subscription(
            PointCloud2, input_topic, self._on_cloud, input_qos)

        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.get_logger().info(
            f'Listening on {input_topic} ({reliability}, depth '
            f'{input_qos.depth}); publishing {self._filtered_pub.topic_name} and '
            f'{self._voxel_pub.topic_name}')

    # ─── Parameters ─────────────────────────────────────────────────────────

    def _build_config(self, overrides):
        flat = {name: self.get_parameter(name).value for name in self._pipeline_parameter_names}
        flat.update(overrides)
        config, _ = config_from_dict(flat_to_nested(flat))
        return config.validate()

    def _read_runtime_parameters(self, overrides):
        def value(name):
            return overrides.get(name, self.get_parameter(name).value)

        self._publish_filtered = bool(value('publish_filtered'))
        self._publish_voxelized = bool(value('publish_voxelized'))
        self._publish_diagnostics = bool(value('publish_diagnostics'))
        self._log_every = int(value('log_every_n_frames'))
        self._budget_ms = float(value('processing_budget_ms'))

    def _log_config(self, config):
        flat = config_to_flat_dict(config)
        self.get_logger().info(
            'Pipeline configuration:\n' + '\n'.join(f'  {k}: {v}' for k, v in flat.items()))
        for warning in config.warnings():
            self.get_logger().warn(warning)

    def _on_set_parameters(self, parameters):
        static = [p.name for p in parameters
                  if p.name in NODE_PARAMETERS and not NODE_PARAMETERS[p.name][2]]
        if static:
            return SetParametersResult(
                successful=False, reason=f'{static} can only be set at startup')

        pipeline_overrides = {p.name: p.value for p in parameters
                              if p.name in self._pipeline_parameter_names}
        runtime_overrides = {p.name: p.value for p in parameters if p.name in NODE_PARAMETERS}

        try:
            if pipeline_overrides:
                config = self._build_config(pipeline_overrides)
            if runtime_overrides:
                self._read_runtime_parameters(runtime_overrides)
        except (ValueError, TypeError) as exc:
            return SetParametersResult(successful=False, reason=str(exc))

        if pipeline_overrides:
            self._pipeline = PointCloudPipeline(config)
            self._rolling = RollingStats(window=self.get_parameter('stats_window').value)
            self._missing_fields_warned = False
            self.get_logger().info(f'Pipeline updated: {pipeline_overrides}')
            for warning in config.warnings():
                self.get_logger().warn(warning)
        return SetParametersResult(successful=True)

    # ─── Processing ─────────────────────────────────────────────────────────

    def _on_cloud(self, msg):
        t_start = time.perf_counter()
        if not self._layout_logged:
            self.get_logger().info('\n' + describe_layout(msg))
            self._layout_logged = True

        try:
            cloud = pointcloud2_to_array(msg)
            t_decoded = time.perf_counter()
            result = self._pipeline.process(cloud, msg.height, msg.width)
        except ValueError as exc:
            self.get_logger().error(f'Dropping cloud: {exc}', throttle_duration_sec=5.0)
            return

        if result.missing_fields and not self._missing_fields_warned:
            self.get_logger().warn(
                f'voxel.average_fields {list(result.missing_fields)} are not in the input '
                f'cloud and are skipped')
            self._missing_fields_warned = True

        t_processed = time.perf_counter()
        if self._publish_filtered:
            self._filtered_pub.publish(array_to_pointcloud2(
                result.filtered, msg.header, result.filtered_height,
                result.filtered_width, result.filtered_is_dense))
        if self._publish_voxelized and result.voxelized is not None:
            self._voxel_pub.publish(array_to_pointcloud2(
                result.voxelized, msg.header, is_dense=True))
        t_end = time.perf_counter()

        stats = result.stats
        stats.timings_ms = {
            'decode': (t_decoded - t_start) * 1e3,
            **stats.timings_ms,
            'encode_publish': (t_end - t_processed) * 1e3,
        }
        stats.total_ms = (t_end - t_start) * 1e3
        self._rolling.add(stats)

        if self._publish_diagnostics:
            self._diag_pub.publish(build_diagnostic_array(
                stats, self._rolling, msg.header.stamp,
                name=f'{self.get_name()}: point-cloud processing',
                hardware_id=msg.header.frame_id,
                budget_ms=self._budget_ms))

        if self._log_every > 0 and self._rolling.total_frames % self._log_every == 0:
            self.get_logger().info(self._rolling.format_summary())


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PointCloudProcessingNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
