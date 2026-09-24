#include "semantic_3d_fusion/projection.hpp"

#include <cmath>

namespace semantic_3d_fusion
{

void Projector::setCamera(const CameraModel &camera)
{
  camera_ = camera;
}

void Projector::setExtrinsic(const Extrinsic &extrinsic)
{
  extrinsic_ = extrinsic;
}

Eigen::Vector3d Projector::transformToCamera(const Eigen::Vector3d &point_lidar) const
{
  return extrinsic_.rotation * point_lidar + extrinsic_.translation;
}

ProjectionResult Projector::project(const Eigen::Vector3d &point_lidar) const
{
  ProjectionResult result;
  const Eigen::Vector3d pc = transformToCamera(point_lidar);

  if (!std::isfinite(pc.x()) || !std::isfinite(pc.y()) || !std::isfinite(pc.z()) || pc.z() <= 0.0) {
    return result;
  }
  if (camera_.fx <= 0.0 || camera_.fy <= 0.0 || camera_.width <= 0 || camera_.height <= 0) {
    return result;
  }

  result.u = camera_.fx * pc.x() / pc.z() + camera_.cx;
  result.v = camera_.fy * pc.y() / pc.z() + camera_.cy;
  result.depth = pc.z();
  result.valid = insideImage(result.u, result.v, camera_.width, camera_.height);
  return result;
}

bool insideImage(double u, double v, int width, int height)
{
  return u >= 0.0 && v >= 0.0 && u < static_cast<double>(width) && v < static_cast<double>(height);
}

}  // namespace semantic_3d_fusion
