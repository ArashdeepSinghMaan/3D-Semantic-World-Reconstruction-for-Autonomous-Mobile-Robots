#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "camera_lidar_calibration/calibration.hpp"

namespace camera_lidar_calibration
{

struct ProjectedPoint
{
  Eigen::Vector3d lidar_point;
  Eigen::Vector3d camera_point;

  double u{0.0};
  double v{0.0};
  double depth{0.0};

  std::size_t lidar_index{0};

  bool valid{false};
};

class Projector
{
public:
  std::vector<ProjectedPoint> project(
    const std::vector<Eigen::Vector3d> & lidar_points,
    const CameraCalibration & camera,
    const ExtrinsicCalibration & extrinsic,
    double min_depth,
    bool apply_distortion) const;

private:
  Eigen::Vector2d distort(
    const Eigen::Vector2d & normalized,
    const CameraCalibration & camera) const;
};

}  // namespace camera_lidar_calibration
