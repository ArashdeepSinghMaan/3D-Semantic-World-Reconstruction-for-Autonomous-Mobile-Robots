#include "semantic_perception/visualization.hpp"

#include <algorithm>
#include <sstream>

#include <opencv2/imgproc.hpp>

namespace semantic_perception
{

cv::Scalar colorForClass(int class_id)
{
  // Deterministic generated color.
  // The semantic interface does not depend on a fixed class palette.
  const int b = (class_id * 53 + 71) % 256;
  const int g = (class_id * 97 + 113) % 256;
  const int r = (class_id * 193 + 37) % 256;
  return cv::Scalar(b, g, r);
}

cv::Mat drawDetections(
  const cv::Mat & image,
  const std::vector<Detection> & detections,
  bool draw_labels,
  int thickness)
{
  cv::Mat output = image.clone();

  for (const auto & d : detections) {
    if (d.bounding_box.empty()) {
      continue;
    }

    const cv::Scalar color =
      colorForClass(d.class_id);

    cv::rectangle(
      output,
      d.bounding_box,
      color,
      thickness,
      cv::LINE_AA);

    if (draw_labels) {
      std::ostringstream label;
      label.setf(std::ios::fixed);
      label.precision(2);
      label << d.class_name << " " << d.confidence;

      int baseline = 0;
      const cv::Size size =
        cv::getTextSize(
          label.str(),
          cv::FONT_HERSHEY_SIMPLEX,
          0.5,
          1,
          &baseline);

      const int x = std::max(0, d.bounding_box.x);
      const int y = std::max(
        size.height,
        d.bounding_box.y);

      cv::rectangle(
        output,
        cv::Rect(
          x,
          y - size.height,
          size.width + 4,
          size.height + baseline + 4),
        color,
        -1);

      cv::putText(
        output,
        label.str(),
        cv::Point(x + 2, y),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA);
    }
  }

  return output;
}

cv::Mat drawSemanticMap(
  const cv::Mat & image,
  const cv::Mat & class_map,
  const cv::Mat & confidence_map,
  float alpha)
{
  cv::Mat overlay =
    cv::Mat::zeros(
      image.size(),
      CV_8UC3);

  for (int y = 0; y < class_map.rows; ++y) {
    for (int x = 0; x < class_map.cols; ++x) {
      const int class_id =
        class_map.at<int>(y, x);

      if (class_id < 0) {
        continue;
      }

      const float confidence =
        confidence_map.at<float>(y, x);

      const cv::Scalar color =
        colorForClass(class_id);

      overlay.at<cv::Vec3b>(y, x) =
        cv::Vec3b(
          static_cast<std::uint8_t>(
            color[0] * confidence),
          static_cast<std::uint8_t>(
            color[1] * confidence),
          static_cast<std::uint8_t>(
            color[2] * confidence));
    }
  }

  cv::Mat result;
  cv::addWeighted(
    overlay,
    std::clamp(alpha, 0.0f, 1.0f),
    image,
    1.0 - std::clamp(alpha, 0.0f, 1.0f),
    0.0,
    result);

  return result;
}

cv::Mat drawStatus(
  const cv::Mat & image,
  const SemanticOutput & output)
{
  cv::Mat result = image.clone();

  std::ostringstream status;
  status.setf(std::ios::fixed);
  status.precision(1);

  status
    << "Model: " << output.model_backend
    << " | Objects: " << output.detections.size()
    << " | Inference: " << output.inference_time_ms << " ms"
    << " | Post: " << output.postprocess_time_ms << " ms";

  cv::putText(
    result,
    status.str(),
    cv::Point(15, 30),
    cv::FONT_HERSHEY_SIMPLEX,
    0.55,
    cv::Scalar(255, 255, 255),
    2,
    cv::LINE_AA);

  return result;
}

}  // namespace semantic_perception
