#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "semantic_perception/types.hpp"

namespace semantic_perception
{

class Inference
{
public:
  Inference() = default;

  bool load(
    const std::string & model_path,
    const std::string & backend,
    const std::string & target,
    const std::vector<std::string> & class_names,
    int input_width,
    int input_height,
    float confidence_threshold,
    float nms_threshold);

  SemanticOutput infer(const cv::Mat & image);

  bool loaded() const
  {
    return loaded_;
  }

private:
  cv::Mat letterbox(
    const cv::Mat & image,
    float & scale,
    int & pad_x,
    int & pad_y) const;

  std::vector<Detection> decodeYoloDetection(
    const std::vector<cv::Mat> & outputs,
    const cv::Size & original_size,
    float scale,
    int pad_x,
    int pad_y);

  void buildPixelMapsFromDetections(
    const std::vector<Detection> & detections,
    const cv::Size & image_size,
    cv::Mat & class_map,
    cv::Mat & confidence_map) const;

  cv::dnn::Net net_;
  bool loaded_{false};

  std::string model_path_;
  std::string backend_;
  std::string target_;

  std::vector<std::string> class_names_;

  int input_width_{640};
  int input_height_{640};

  float confidence_threshold_{0.25f};
  float nms_threshold_{0.45f};
};

}  // namespace semantic_perception
