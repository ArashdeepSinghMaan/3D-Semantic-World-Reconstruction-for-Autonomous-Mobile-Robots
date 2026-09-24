#include "semantic_perception/postprocess.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace semantic_perception
{

void filterDetections(
  std::vector<Detection> & detections,
  float minimum_confidence)
{
  detections.erase(
    std::remove_if(
      detections.begin(),
      detections.end(),
      [minimum_confidence](const Detection & d)
      {
        return d.confidence < minimum_confidence;
      }),
    detections.end());
}

void clipDetectionsToImage(
  std::vector<Detection> & detections,
  const cv::Size & image_size)
{
  for (auto & d : detections) {
    d.bounding_box &= cv::Rect(
      0, 0, image_size.width, image_size.height);
  }
}

void buildSemanticMaps(
  const std::vector<Detection> & detections,
  const cv::Size & image_size,
  cv::Mat & class_map,
  cv::Mat & confidence_map)
{
  class_map = cv::Mat(
    image_size.height,
    image_size.width,
    CV_32SC1,
    cv::Scalar(-1));

  confidence_map = cv::Mat(
    image_size.height,
    image_size.width,
    CV_32FC1,
    cv::Scalar(0.0f));

  // Detection-only fallback:
  // assign the detected class/confidence to pixels inside each box.
  //
  // When a real segmentation mask is available, use the mask instead.
  for (const auto & d : detections) {
    if (d.bounding_box.empty()) {
      continue;
    }

    for (int y = d.bounding_box.y;
         y < d.bounding_box.y + d.bounding_box.height; ++y) {
      for (int x = d.bounding_box.x;
           x < d.bounding_box.x + d.bounding_box.width; ++x) {

        float & current_conf =
          confidence_map.at<float>(y, x);

        if (d.confidence > current_conf) {
          current_conf = d.confidence;
          class_map.at<int>(y, x) = d.class_id;
        }
      }
    }
  }

  // If masks are available, overwrite box assignments with mask pixels.
  for (const auto & d : detections) {
    if (d.mask.empty() || d.bounding_box.empty()) {
      continue;
    }

    cv::Mat mask;

    if (d.mask.size() == d.bounding_box.size()) {
      mask = d.mask;
    } else {
      cv::resize(
        d.mask,
        mask,
        d.bounding_box.size(),
        0.0,
        0.0,
        cv::INTER_NEAREST);
    }

    for (int my = 0; my < mask.rows; ++my) {
      for (int mx = 0; mx < mask.cols; ++mx) {
        const float mask_value =
          mask.type() == CV_32FC1 ?
          mask.at<float>(my, mx) :
          static_cast<float>(mask.at<std::uint8_t>(my, mx)) / 255.0f;

        if (mask_value <= 0.0f) {
          continue;
        }

        const int x = d.bounding_box.x + mx;
        const int y = d.bounding_box.y + my;

        if (x < 0 || y < 0 ||
            x >= image_size.width ||
            y >= image_size.height) {
          continue;
        }

        float & current_conf =
          confidence_map.at<float>(y, x);

        if (d.confidence * mask_value > current_conf) {
          current_conf = d.confidence * mask_value;
          class_map.at<int>(y, x) = d.class_id;
        }
      }
    }
  }
}

}  // namespace semantic_perception
