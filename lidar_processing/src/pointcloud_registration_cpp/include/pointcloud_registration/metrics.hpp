#pragma once

#include <cmath>

#include <Eigen/Dense>

namespace pointcloud_registration
{

struct TransformMetrics
{
  double translation_magnitude{0.0};
  double rotation_angle_deg{0.0};
  double delta_translation_m{0.0};
  double delta_rotation_deg{0.0};
};

inline double rotationAngleDeg(const Eigen::Matrix4f & transform)
{
  Eigen::Matrix3f R = transform.block<3, 3>(0, 0);
  double value = (static_cast<double>(R.trace()) - 1.0) / 2.0;
  value = std::max(-1.0, std::min(1.0, value));
  return std::acos(value) * 180.0 / M_PI;
}

inline TransformMetrics calculateTransformMetrics(
  const Eigen::Matrix4f & transform,
  const Eigen::Matrix4f & initial)
{
  TransformMetrics metrics;
  metrics.translation_magnitude = transform.block<3, 1>(0, 3).norm();
  metrics.rotation_angle_deg = rotationAngleDeg(transform);

  Eigen::Matrix4f delta = transform * initial.inverse();
  metrics.delta_translation_m = delta.block<3, 1>(0, 3).norm();
  metrics.delta_rotation_deg = rotationAngleDeg(delta);

  return metrics;
}

}  // namespace pointcloud_registration
