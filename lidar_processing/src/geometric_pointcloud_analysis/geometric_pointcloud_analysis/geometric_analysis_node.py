"""ROS 2 node: point cloud in; ground, non-ground, clusters, normals, markers out.

The node handles ROS I/O only; all geometry is done by ``GeometryPipeline``.

"Up" for ground fitting comes from ``ground.reference_normal``. If
``ground.reference_frame`` is set (e.g. the robot base frame), that vector is
interpreted in the reference frame and rotated into the cloud's frame with
TF on every frame, so a pitched or rolled sensor still searches for the
right plane. On TF failure the node falls back to the vector as given in the
cloud frame and warns (throttled).

Pipeline parameters (``ground.*``, ``ransac.*``, ``grid.*``, ``clustering.*``,
``normals.*``, ``boxes.*``, ``features.*``) and the ``markers.*`` / publish
options can be changed at runtime with ``ros2 param set``.
"""

import time

from diagnostic_msgs.msg import DiagnosticArray
from lidar_pointcloud_processing.pointcloud_utils import (
    array_to_pointcloud2,
    describe_layout,
    pointcloud2_to_array,
    xyz_from_array,
)
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray

from .bounding_boxes import quaternion_rotate
from .diagnostics import build_diagnostic_array, RollingTimes
from .markers import build_marker_array, label_colors, pack_rgb
from .pipeline import (
    config_from_dict,
    config_to_flat_dict,
    flat_to_nested,
    GeometryConfig,
    GeometryPipeline,
)

# name: (default, description, changeable at runtime)
NODE_PARAMETERS = {
    'input_topic': ('/lidar/points/voxelized', 'Input sensor_msgs/PointCloud2', False),
    'ground_topic': ('/geometry/ground', 'Ground points (input fields preserved)', False),
    'non_ground_topic': ('/geometry/non_ground', 'Non-ground points (input fields)', False),
    'clusters_topic': ('/geometry/clusters', 'Clustered points: x y z cluster_id rgb', False),
    'normals_topic': ('/geometry/normals',
                      'Points with normal_x/y/z and curvature', False),
    'markers_topic': ('/geometry/markers', 'visualization_msgs/MarkerArray', False),
    'diagnostics_topic': ('/diagnostics', 'diagnostic_msgs/DiagnosticArray', False),
    'input_qos_reliability': ('best_effort', "'best_effort' or 'reliable'", False),
    'queue_depth': (1, 'Input queue depth; 1 drops stale frames', False),
    'stats_window': (100, 'Frames in the rolling statistics window', False),
    'process_every_n': (1, 'Process every Nth input frame', True),
    'publish_ground': (True, 'Publish ground cloud', True),
    'publish_non_ground': (True, 'Publish non-ground cloud', True),
    'publish_clusters': (True, 'Publish cluster cloud', True),
    'publish_normals': (True, 'Publish normals cloud', True),
    'publish_markers': (True, 'Publish RViz markers', True),
    'publish_diagnostics': (True, 'Publish diagnostics', True),
    'log_every_n_frames': (50, 'Rolling summary every N processed frames (0 = off)', True),
    'log_clusters': (False, 'Log a one-line description of every cluster', True),
    'processing_budget_ms': (100.0, 'Diagnostics WARN above this mean time', True),
    'markers.boxes': (True, 'Draw cluster bounding boxes', True),
    'markers.labels': (True, 'Draw cluster labels', True),
    'markers.ground': (True, 'Draw the ground plane / grid cells', True),
    'markers.ground_size': (20.0, 'Side length of the drawn global plane (m)', True),
    'markers.normals_max': (300, 'Number of normals drawn as lines (0 = none)', True),
    'markers.normal_length': (0.3, 'Length of drawn normals (m)', True),
}


class GeometricAnalysisNode(Node):

    def __init__(self):
        super().__init__('geometric_analysis')
        for name, (default, description, runtime) in NODE_PARAMETERS.items():
            suffix = '' if runtime else ' (read at startup)'
            self.declare_parameter(
                name, default, ParameterDescriptor(description=description + suffix))

        self._pipeline_names = []
        for name, default in config_to_flat_dict(GeometryConfig()).items():
            self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))
            self._pipeline_names.append(name)

        config = self._build_config({})
        self._pipeline = GeometryPipeline(config)
        self._log_config(config)
        self._read_runtime({})

        self._tf_buffer = None
        self._setup_tf(config)

        self._rolling = RollingTimes(self.get_parameter('stats_window').value)
        self._received = 0
        self._layout_logged = False

        reliability = str(self.get_parameter('input_qos_reliability').value).lower()
        if reliability not in ('best_effort', 'reliable'):
            raise ValueError("input_qos_reliability must be 'best_effort' or 'reliable'")
        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=int(self.get_parameter('queue_depth').value),
            reliability=(ReliabilityPolicy.BEST_EFFORT if reliability == 'best_effort'
                         else ReliabilityPolicy.RELIABLE),
            durability=DurabilityPolicy.VOLATILE)
        output_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                                reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.VOLATILE)

        def topic(name):
            return self.get_parameter(name).value

        self._pubs = {
            'ground': self.create_publisher(PointCloud2, topic('ground_topic'), output_qos),
            'non_ground': self.create_publisher(PointCloud2, topic('non_ground_topic'),
                                                output_qos),
            'clusters': self.create_publisher(PointCloud2, topic('clusters_topic'), output_qos),
            'normals': self.create_publisher(PointCloud2, topic('normals_topic'), output_qos),
            'markers': self.create_publisher(MarkerArray, topic('markers_topic'), output_qos),
            'diagnostics': self.create_publisher(DiagnosticArray, topic('diagnostics_topic'),
                                                 10),
        }
        self._subscription = self.create_subscription(
            PointCloud2, topic('input_topic'), self._on_cloud, input_qos)
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.get_logger().info(f'Listening on {topic("input_topic")} ({reliability})')

    # ─── Parameters ─────────────────────────────────────────────────────────

    def _build_config(self, overrides):
        flat = {name: self.get_parameter(name).value for name in self._pipeline_names}
        flat.update(overrides)
        config, _ = config_from_dict(flat_to_nested(flat))
        return config.validate()

    def _read_runtime(self, overrides):
        def value(name):
            return overrides.get(name, self.get_parameter(name).value)

        self._every_n = max(1, int(value('process_every_n')))
        self._publish = {key: bool(value(f'publish_{key}')) for key in
                         ('ground', 'non_ground', 'clusters', 'normals', 'markers',
                          'diagnostics')}
        self._log_every = int(value('log_every_n_frames'))
        self._log_clusters = bool(value('log_clusters'))
        self._budget_ms = float(value('processing_budget_ms'))
        self._marker_options = {
            'show_boxes': bool(value('markers.boxes')),
            'show_labels': bool(value('markers.labels')),
            'show_ground': bool(value('markers.ground')),
            'ground_size': float(value('markers.ground_size')),
            'normals_max': int(value('markers.normals_max')),
            'normal_length': float(value('markers.normal_length')),
        }

    def _log_config(self, config):
        self.get_logger().info('Geometry configuration:\n' + '\n'.join(
            f'  {k}: {v}' for k, v in config_to_flat_dict(config).items()))
        for warning in config.warnings():
            self.get_logger().warn(warning)

    def _setup_tf(self, config):
        if config.ground.reference_frame and self._tf_buffer is None:
            from tf2_ros import Buffer, TransformListener
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            self.get_logger().info(
                f'Ground reference {config.ground.reference_normal} is expressed in '
                f'"{config.ground.reference_frame}" and rotated into the cloud frame via TF')

    def _on_set_parameters(self, parameters):
        static = [p.name for p in parameters
                  if p.name in NODE_PARAMETERS and not NODE_PARAMETERS[p.name][2]]
        if static:
            return SetParametersResult(successful=False,
                                       reason=f'{static} can only be set at startup')
        pipeline_overrides = {p.name: p.value for p in parameters
                              if p.name in self._pipeline_names}
        runtime_overrides = {p.name: p.value for p in parameters if p.name in NODE_PARAMETERS}
        try:
            config = self._build_config(pipeline_overrides) if pipeline_overrides else None
            if runtime_overrides:
                self._read_runtime(runtime_overrides)
        except (ValueError, TypeError) as exc:
            return SetParametersResult(successful=False, reason=str(exc))
        if config is not None:
            self._pipeline = GeometryPipeline(config)
            self._setup_tf(config)
            self._rolling = RollingTimes(self.get_parameter('stats_window').value)
            self.get_logger().info(f'Geometry pipeline updated: {pipeline_overrides}')
            for warning in config.warnings():
                self.get_logger().warn(warning)
        return SetParametersResult(successful=True)

    # ─── Processing ─────────────────────────────────────────────────────────

    def _reference_up(self, frame_id):
        g = self._pipeline.config.ground
        if not g.reference_frame or self._tf_buffer is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(frame_id, g.reference_frame, Time())
        except Exception as exc:  # noqa: BLE001 - tf2 raises several exception types
            self.get_logger().warn(
                f'TF {g.reference_frame} -> {frame_id} unavailable ({exc}); using '
                f'ground.reference_normal in the cloud frame', throttle_duration_sec=10.0)
            return None
        r = tf.transform.rotation
        return quaternion_rotate((r.x, r.y, r.z, r.w), g.reference_normal)

    def _on_cloud(self, msg):
        self._received += 1
        if (self._received - 1) % self._every_n:
            return
        t_start = time.perf_counter()
        if not self._layout_logged:
            self.get_logger().info('\n' + describe_layout(msg))
            self._layout_logged = True
        try:
            cloud = pointcloud2_to_array(msg)
            xyz_all = xyz_from_array(cloud)
        except ValueError as exc:
            self.get_logger().error(f'Dropping cloud: {exc}', throttle_duration_sec=5.0)
            return
        valid = np.flatnonzero(np.isfinite(xyz_all).all(axis=1))
        cloud, xyz = cloud[valid], xyz_all[valid]
        t_decoded = time.perf_counter()

        result = self._pipeline.process(xyz, reference_normal=self._reference_up(
            msg.header.frame_id))
        t_processed = time.perf_counter()
        self._publish_outputs(msg.header, cloud, xyz, result)
        t_end = time.perf_counter()

        stage_ms = {'decode': (t_decoded - t_start) * 1e3, **result.timings_ms,
                    'encode_publish': (t_end - t_processed) * 1e3}
        total_ms = (t_end - t_start) * 1e3
        self._rolling.add(total_ms, stage_ms, result.stats)

        if self._publish['diagnostics']:
            self._pubs['diagnostics'].publish(build_diagnostic_array(
                result.stats, stage_ms, total_ms, self._rolling, msg.header.stamp,
                name=f'{self.get_name()}: geometric analysis',
                hardware_id=msg.header.frame_id, budget_ms=self._budget_ms))
        if result.stats.get('ground_status', 'ok') != 'ok':
            self.get_logger().warn(f'Ground not found: {result.stats["ground_status"]}',
                                   throttle_duration_sec=5.0)
        if self._log_clusters:
            self.get_logger().info('\n'.join(c.summary() for c in result.clusters) or
                                   'no clusters')
        if self._log_every > 0 and self._rolling.total_frames % self._log_every == 0:
            self.get_logger().info(self._rolling.format_summary())

    def _publish_outputs(self, header, cloud, xyz, result):
        ground = result.ground_mask
        if self._publish['ground']:
            self._pubs['ground'].publish(array_to_pointcloud2(cloud[ground], header))
        if self._publish['non_ground']:
            self._pubs['non_ground'].publish(array_to_pointcloud2(cloud[~ground], header))

        if self._publish['clusters']:
            clustered = np.flatnonzero(result.labels >= 0)
            out = np.empty(len(clustered), dtype=[('x', np.float32), ('y', np.float32),
                                                  ('z', np.float32), ('cluster_id', np.int32),
                                                  ('rgb', np.float32)])
            out['x'], out['y'], out['z'] = xyz[clustered].T
            out['cluster_id'] = result.labels[clustered]
            out['rgb'] = pack_rgb(label_colors(result.labels[clustered]))
            self._pubs['clusters'].publish(array_to_pointcloud2(out, header))

        if self._publish['normals'] and result.normals is not None:
            ok = np.flatnonzero(result.normals.valid)
            out = np.empty(len(ok), dtype=[(n, np.float32) for n in (
                'x', 'y', 'z', 'normal_x', 'normal_y', 'normal_z', 'curvature')])
            out['x'], out['y'], out['z'] = xyz[ok].T
            normals = result.normals.normals[ok]
            out['normal_x'], out['normal_y'], out['normal_z'] = normals.T
            out['curvature'] = result.normals.curvature[ok]
            self._pubs['normals'].publish(array_to_pointcloud2(out, header))

        if self._publish['markers']:
            self._pubs['markers'].publish(build_marker_array(
                result, header, xyz=xyz, **self._marker_options))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GeometricAnalysisNode()
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
