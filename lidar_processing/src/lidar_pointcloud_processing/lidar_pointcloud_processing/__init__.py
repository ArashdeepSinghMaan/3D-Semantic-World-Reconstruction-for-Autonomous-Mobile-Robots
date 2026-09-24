"""LiDAR point-cloud preprocessing for the 3D semantic world reconstruction project.

The algorithm modules (``pointcloud_utils``, ``filters``, ``voxelization``,
``spatial_search``, ``pipeline``, ``analysis``, ``diagnostics``) do not import
ROS at module level, so they can be used and tested on saved frames without a
running ROS system. Only ``pointcloud_node`` and ``pointcloud_inspector`` are
ROS nodes.
"""

__version__ = '0.1.0'
