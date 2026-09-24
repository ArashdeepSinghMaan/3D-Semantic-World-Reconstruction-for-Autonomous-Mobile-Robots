#include "camera_lidar_calibration/calibration.hpp"

#include <Eigen/Geometry>

namespace camera_lidar_calibration
{

Eigen::Matrix3d eulerToRotation(
  double roll, double pitch, double yaw)
{
  const Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());

  return (rz * ry * rx).toRotationMatrix();
}

Eigen::Matrix4d makeTransform(
  double tx, double ty, double tz,
  double roll, double pitch, double yaw)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = eulerToRotation(roll, pitch, yaw);
  T(0, 3) = tx;
  T(1, 3) = ty;
  T(2, 3) = tz;
  return T;
}

}  // namespace camera_lidar_calibration
