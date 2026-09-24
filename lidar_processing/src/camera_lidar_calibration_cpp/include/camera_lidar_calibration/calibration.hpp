#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>

namespace camera_lidar_calibration
{

struct CameraCalibration
{
  int width{0};
  int height{0};

  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};

  std::vector<double> distortion;
  std::string distortion_model{"plumb_bob"};

  bool valid() const
  {
    return width > 0 && height > 0 &&
           fx > 0.0 && fy > 0.0;
  }
};

struct ExtrinsicCalibration
{
  Eigen::Matrix4d T_camera_lidar{Eigen::Matrix4d::Identity()};
};

Eigen::Matrix4d makeTransform(
  double tx, double ty, double tz,
  double roll, double pitch, double yaw);

Eigen::Matrix3d eulerToRotation(
  double roll, double pitch, double yaw);

}  // namespace camera_lidar_calibration
