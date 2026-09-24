#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "semantic_perception/types.hpp"

namespace semantic_perception
{

cv::Mat drawDetections(
  const cv::Mat & image,
  const std::vector<Detection> & detections,
  bool draw_labels,
  int thickness);

cv::Mat drawSemanticMap(
  const cv::Mat & image,
  const cv::Mat & class_map,
  const cv::Mat & confidence_map,
  float alpha);

cv::Mat drawStatus(
  const cv::Mat & image,
  const SemanticOutput & output);

}  // namespace semantic_perception
