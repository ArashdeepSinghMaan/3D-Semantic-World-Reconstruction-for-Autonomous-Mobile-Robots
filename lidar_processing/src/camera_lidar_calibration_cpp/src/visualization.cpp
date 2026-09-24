#include "camera_lidar_calibration/visualization.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <opencv2/imgproc.hpp>

namespace camera_lidar_calibration
{

cv::Mat drawProjectedPoints(
  const cv::Mat & image,
  const std::vector<ProjectedPoint> & points,
  int radius,
  double alpha,
  bool color_by_depth)
{
  cv::Mat overlay = image.clone();

  double min_depth = 0.0;
  double max_depth = 1.0;
  bool found_depth = false;

  if (color_by_depth) {
    for (const auto & p : points) {
      if (p.valid && std::isfinite(p.depth)) {
        if (!found_depth) {
          min_depth = max_depth = p.depth;
          found_depth = true;
        } else {
          min_depth = std::min(min_depth, p.depth);
          max_depth = std::max(max_depth, p.depth);
        }
      }
    }
    if (max_depth <= min_depth) {
      max_depth = min_depth + 1.0;
    }
  }

  for (const auto & p : points) {
    if (!p.valid) {
      continue;
    }

    int u = static_cast<int>(std::lround(p.u));
    int v = static_cast<int>(std::lround(p.v));

    if (u < 0 || v < 0 ||
        u >= overlay.cols || v >= overlay.rows) {
      continue;
    }

    cv::Scalar color(0, 255, 0);

    if (color_by_depth) {
      const double normalized =
        std::clamp(
          (p.depth - min_depth) / (max_depth - min_depth),
          0.0, 1.0);

      // Blue -> cyan -> yellow -> red depth visualization.
      const int b = static_cast<int>(255.0 * (1.0 - normalized));
      const int r = static_cast<int>(255.0 * normalized);
      const int g = static_cast<int>(
        255.0 * (1.0 - std::abs(2.0 * normalized - 1.0)));
      color = cv::Scalar(b, g, r);
    }

    cv::circle(
      overlay,
      cv::Point(u, v),
      radius,
      color,
      -1,
      cv::LINE_AA);
  }

  cv::Mat result;
  cv::addWeighted(
    overlay, std::clamp(alpha, 0.0, 1.0),
    image, 1.0 - std::clamp(alpha, 0.0, 1.0),
    0.0, result);

  return result;
}

cv::Mat drawCalibrationText(
  const cv::Mat & image,
  const std::string & text)
{
  cv::Mat result = image.clone();

  cv::putText(
    result,
    text,
    cv::Point(20, 30),
    cv::FONT_HERSHEY_SIMPLEX,
    0.7,
    cv::Scalar(255, 255, 255),
    2,
    cv::LINE_AA);

  return result;
}

}  // namespace camera_lidar_calibration
