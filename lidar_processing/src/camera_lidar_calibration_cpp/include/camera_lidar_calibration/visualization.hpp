#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "camera_lidar_calibration/projection.hpp"

namespace camera_lidar_calibration
{

cv::Mat drawProjectedPoints(
  const cv::Mat & image,
  const std::vector<ProjectedPoint> & points,
  int radius,
  double alpha,
  bool color_by_depth);

cv::Mat drawCalibrationText(
  const cv::Mat & image,
  const std::string & text);

}  // namespace camera_lidar_calibration
