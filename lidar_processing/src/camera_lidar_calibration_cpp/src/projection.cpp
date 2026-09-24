#include "camera_lidar_calibration/projection.hpp"

#include <cmath>
#include <stdexcept>

namespace camera_lidar_calibration
{

Eigen::Vector2d Projector::distort(
  const Eigen::Vector2d & normalized,
  const CameraCalibration & camera) const
{
  // ROS CameraInfo distortion ordering:
  // plumb_bob: k1, k2, t1, t2, k3
  // rational_polynomial may additionally contain k4, k5, k6.
  const auto & d = camera.distortion;

  const double x = normalized.x();
  const double y = normalized.y();
  const double r2 = x * x + y * y;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;

  const double k1 = d.size() > 0 ? d[0] : 0.0;
  const double k2 = d.size() > 1 ? d[1] : 0.0;
  const double p1 = d.size() > 2 ? d[2] : 0.0;
  const double p2 = d.size() > 3 ? d[3] : 0.0;
  const double k3 = d.size() > 4 ? d[4] : 0.0;

  if (camera.distortion_model == "plumb_bob" ||
      camera.distortion_model == "rational_polynomial") {
    const double radial =
      1.0 + k1 * r2 + k2 * r4 + k3 * r6;

    const double xd =
      x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);

    const double yd =
      y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

    return Eigen::Vector2d(xd, yd);
  }

  // For unknown models, leave the normalized coordinates unchanged.
  return normalized;
}

std::vector<ProjectedPoint> Projector::project(
  const std::vector<Eigen::Vector3d> & lidar_points,
  const CameraCalibration & camera,
  const ExtrinsicCalibration & extrinsic,
  double min_depth,
  bool apply_distortion) const
{
  if (!camera.valid()) {
    throw std::runtime_error(
      "Invalid camera calibration. CameraInfo or configured intrinsics are required.");
  }

  std::vector<ProjectedPoint> output;
  output.reserve(lidar_points.size());

  for (std::size_t i = 0; i < lidar_points.size(); ++i) {
    ProjectedPoint p;
    p.lidar_point = lidar_points[i];
    p.lidar_index = i;

    Eigen::Vector4d lidar_h;
    lidar_h << lidar_points[i], 1.0;

    const Eigen::Vector4d camera_h =
      extrinsic.T_camera_lidar * lidar_h;

    p.camera_point = camera_h.head<3>();
    p.depth = p.camera_point.z();

    if (!std::isfinite(p.depth) || p.depth <= min_depth) {
      output.push_back(p);
      continue;
    }

    const double x = p.camera_point.x() / p.camera_point.z();
    const double y = p.camera_point.y() / p.camera_point.z();

    Eigen::Vector2d normalized(x, y);
    if (apply_distortion) {
      normalized = distort(normalized, camera);
    }

    p.u = camera.fx * normalized.x() + camera.cx;
    p.v = camera.fy * normalized.y() + camera.cy;

    p.valid =
      std::isfinite(p.u) &&
      std::isfinite(p.v) &&
      p.u >= 0.0 &&
      p.u < static_cast<double>(camera.width) &&
      p.v >= 0.0 &&
      p.v < static_cast<double>(camera.height);

    output.push_back(p);
  }

  return output;
}

}  // namespace camera_lidar_calibration
