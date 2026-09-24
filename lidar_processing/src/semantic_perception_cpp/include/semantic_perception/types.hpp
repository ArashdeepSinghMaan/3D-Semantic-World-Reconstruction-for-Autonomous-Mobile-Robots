#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace semantic_perception
{

struct Detection
{
  int class_id{-1};
  std::string class_name;
  float confidence{0.0f};

  cv::Rect bounding_box;

  // Optional instance/semantic mask.
  cv::Mat mask;
};

struct SemanticPixel
{
  std::int32_t class_id{-1};
  float confidence{0.0f};
};

struct SemanticOutput
{
  std::vector<Detection> detections;

  // Per-pixel semantic result.
  // class_map: CV_32SC1
  // confidence_map: CV_32FC1
  cv::Mat class_map;
  cv::Mat confidence_map;

  // Optional color visualization.
  cv::Mat visualization;

  std::string model_name;
  std::string model_backend;
  double inference_time_ms{0.0};
  double postprocess_time_ms{0.0};
};

}  // namespace semantic_perception
