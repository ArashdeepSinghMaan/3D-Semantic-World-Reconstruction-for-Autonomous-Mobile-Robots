"""RViz2 visualization: MarkerArray for boxes, labels, ground model and normals.

ROS message imports are lazy so the colour helpers can be used offline.
Each MarkerArray starts with a DELETEALL marker so objects from the previous
frame never linger when the number of clusters changes.
"""

import colorsys

import numpy as np

from .bounding_boxes import EDGES, rotation_to_quaternion
from .ground import plane_basis

_GOLDEN = 0.618033988749895


def cluster_color(cluster_id):
    """Distinct, stable RGB in [0, 1] for a cluster id (golden-ratio hue walk)."""
    hue = (0.1 + cluster_id * _GOLDEN) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.75, 0.95)


def pack_rgb(colors):
    """(N, 3) floats in [0, 1] -> float32 'rgb' field as expected by RViz / PCL."""
    c = np.clip(np.round(np.asarray(colors) * 255), 0, 255).astype(np.uint32)
    packed = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
    return packed.view(np.float32)


def label_colors(labels, unlabeled=(0.5, 0.5, 0.5)):
    labels = np.asarray(labels)
    colors = np.tile(np.asarray(unlabeled, dtype=np.float64), (len(labels), 1))
    for cluster_id in np.unique(labels[labels >= 0]):
        colors[labels == cluster_id] = cluster_color(int(cluster_id))
    return colors


def build_marker_array(result, header, xyz=None, show_boxes=True, show_labels=True,
                       show_ground=True, ground_size=20.0, normals_max=0,
                       normal_length=0.3, rng_seed=0):
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    def new_marker(ns, marker_id, marker_type):
        m = Marker()
        m.header = header
        m.ns = ns
        m.id = marker_id
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def point(p):
        return Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))

    def color(rgb, alpha=1.0):
        return ColorRGBA(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=float(alpha))

    array = MarkerArray()
    clear = Marker()
    clear.header = header
    clear.action = Marker.DELETEALL
    array.markers.append(clear)

    for cluster in result.clusters:
        if cluster.box is None:
            continue
        rgb = cluster_color(cluster.id)
        if show_boxes:
            m = new_marker('boxes', cluster.id, Marker.LINE_LIST)
            m.scale.x = 0.03
            m.color = color(rgb)
            corners = cluster.box.corners()
            m.points = [point(corners[i]) for edge in EDGES for i in edge]
            array.markers.append(m)
        if show_labels:
            m = new_marker('labels', cluster.id, Marker.TEXT_VIEW_FACING)
            top = cluster.box.center + result.up * (cluster.box.extent[2] / 2.0 + 0.3)
            m.pose.position = point(top)
            m.scale.z = 0.25
            m.color = color((1.0, 1.0, 1.0))
            e = cluster.box.extent
            m.text = f'#{cluster.id} {cluster.shape}\n{e[0]:.1f}x{e[1]:.1f}x{e[2]:.1f} m'
            array.markers.append(m)

    ground = result.ground
    if show_ground and ground is not None and ground.plane is not None:
        plane = ground.plane
        u, v = plane_basis(plane.normal)
        if ground.cells:
            m = new_marker('ground', 0, Marker.TRIANGLE_LIST)
            m.scale.x = m.scale.y = m.scale.z = 1.0
            s = ground.cell_size
            for cell in ground.cells:
                i, j = cell.index
                corners = []
                for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                    p = a * s * u + b * s * v - plane.d * plane.normal
                    t = -(cell.normal @ p + cell.d) / (cell.normal @ plane.normal)
                    corners.append(p + t * plane.normal)
                if cell.accepted:
                    rgba = color((0.2, 0.8, 0.3), 0.35)
                else:
                    rgba = color((0.9, 0.3, 0.2), 0.25)
                for k in (0, 1, 2, 0, 2, 3):
                    m.points.append(point(corners[k]))
                    m.colors.append(rgba)
            m.color = color((1.0, 1.0, 1.0))
            array.markers.append(m)
        else:
            m = new_marker('ground', 0, Marker.CUBE)
            m.pose.position = point(-plane.d * plane.normal)
            q = rotation_to_quaternion(np.column_stack([u, v, plane.normal]))
            m.pose.orientation.x, m.pose.orientation.y = float(q[0]), float(q[1])
            m.pose.orientation.z, m.pose.orientation.w = float(q[2]), float(q[3])
            m.scale.x = m.scale.y = float(ground_size)
            m.scale.z = 0.01
            m.color = color((0.2, 0.8, 0.3), 0.3)
            array.markers.append(m)

    normals = result.normals
    if normals_max > 0 and normals is not None and xyz is not None:
        valid = np.flatnonzero(normals.valid)
        if valid.size:
            rng = np.random.default_rng(rng_seed)
            chosen = rng.choice(valid, size=min(normals_max, valid.size), replace=False)
            m = new_marker('normals', 0, Marker.LINE_LIST)
            m.scale.x = 0.01
            m.color = color((0.3, 0.7, 1.0))
            for index in chosen:
                m.points.append(point(xyz[index]))
                m.points.append(point(xyz[index] + normal_length * normals.normals[index]))
            array.markers.append(m)

    return array
