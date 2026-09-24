#pragma once

#include <vector>

#include "semantic_perception/types.hpp"

namespace semantic_perception
{

void filterDetections(
  std::vector<Detection> & detections,
  float minimum_confidence);

void clipDetectionsToImage(
  std::vector<Detection> & detections,
  const cv::Size & image_size);

void buildSemanticMaps(
  const std::vector<Detection> & detections,
  const cv::Size & image_size,
  cv::Mat & class_map,
  cv::Mat & confidence_map);

}  // namespace semantic_perception
