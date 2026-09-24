#include "semantic_perception/inference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <fstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "semantic_perception/postprocess.hpp"

namespace semantic_perception
{

bool Inference::load(
  const std::string & model_path,
  const std::string & backend,
  const std::string & target,
  const std::vector<std::string> & class_names,
  int input_width,
  int input_height,
  float confidence_threshold,
  float nms_threshold)
{
  model_path_ = model_path;
  backend_ = backend;
  target_ = target;
  class_names_ = class_names;
  input_width_ = input_width;
  input_height_ = input_height;
  confidence_threshold_ = confidence_threshold;
  nms_threshold_ = nms_threshold;

  try {
    std::ifstream model_file(model_path_, std::ios::binary);
    if (!model_file.good()) {
      loaded_ = false;
      return false;
    }

    net_ = cv::dnn::readNet(model_path_);

    if (backend_ == "cuda") {
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else if (backend_ == "cuda_fp16") {
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    } else if (backend_ == "opencv") {
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } else {
      throw std::runtime_error(
        "Unsupported backend: " + backend_);
    }

    loaded_ = true;
    return true;
  } catch (const std::exception &) {
    loaded_ = false;
    return false;
  }
}

cv::Mat Inference::letterbox(
  const cv::Mat & image,
  float & scale,
  int & pad_x,
  int & pad_y) const
{
  const float sx =
    static_cast<float>(input_width_) /
    static_cast<float>(image.cols);

  const float sy =
    static_cast<float>(input_height_) /
    static_cast<float>(image.rows);

  scale = std::min(sx, sy);

  const int resized_width =
    static_cast<int>(std::round(image.cols * scale));
  const int resized_height =
    static_cast<int>(std::round(image.rows * scale));

  cv::Mat resized;
  cv::resize(
    image,
    resized,
    cv::Size(resized_width, resized_height));

  cv::Mat output(
    input_height_,
    input_width_,
    image.type(),
    cv::Scalar(114, 114, 114));

  pad_x = (input_width_ - resized_width) / 2;
  pad_y = (input_height_ - resized_height) / 2;

  resized.copyTo(
    output(
      cv::Rect(
        pad_x,
        pad_y,
        resized_width,
        resized_height)));

  return output;
}

std::vector<Detection> Inference::decodeYoloDetection(
  const std::vector<cv::Mat> & outputs,
  const cv::Size & original_size,
  float scale,
  int pad_x,
  int pad_y)
{
  std::vector<Detection> detections;

  if (outputs.empty()) {
    return detections;
  }

  cv::Mat output = outputs[0];

  // This decoder targets a common YOLO detection ONNX layout:
  //
  // [1, 84, 8400] or [1, attributes, candidates]
  //
  // attributes = 4 + number_of_classes
  //
  // It intentionally does not claim to decode segmentation prototypes.
  if (output.dims == 3) {
    const int dimensions = output.size[1];
    const int candidates = output.size[2];

    if (dimensions < 5) {
      return detections;
    }

    cv::Mat rows(candidates, dimensions, CV_32F);

    for (int c = 0; c < candidates; ++c) {
      for (int d = 0; d < dimensions; ++d) {
        rows.at<float>(c, d) =
          output.ptr<float>(0, d)[c];
      }
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    const int number_of_classes =
      dimensions - 4;

    for (int i = 0; i < candidates; ++i) {
      float best_score = 0.0f;
      int best_class = -1;

      for (int c = 0; c < number_of_classes; ++c) {
        const float score = rows.at<float>(i, 4 + c);
        if (score > best_score) {
          best_score = score;
          best_class = c;
        }
      }

      if (best_score < confidence_threshold_) {
        continue;
      }

      const float cx = rows.at<float>(i, 0);
      const float cy = rows.at<float>(i, 1);
      const float w = rows.at<float>(i, 2);
      const float h = rows.at<float>(i, 3);

      const float x = (cx - 0.5f * w - pad_x) / scale;
      const float y = (cy - 0.5f * h - pad_y) / scale;
      const float width = w / scale;
      const float height = h / scale;

      boxes.emplace_back(
        static_cast<int>(std::round(x)),
        static_cast<int>(std::round(y)),
        static_cast<int>(std::round(width)),
        static_cast<int>(std::round(height)));

      scores.push_back(best_score);
      class_ids.push_back(best_class);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(
      boxes,
      scores,
      confidence_threshold_,
      nms_threshold_,
      keep);

    for (int idx : keep) {
      Detection d;
      d.class_id = class_ids[idx];
      d.confidence = scores[idx];
      d.bounding_box = boxes[idx];

      if (d.class_id >= 0 &&
          d.class_id < static_cast<int>(class_names_.size())) {
        d.class_name = class_names_[d.class_id];
      } else {
        d.class_name = "class_" +
          std::to_string(d.class_id);
      }

      detections.push_back(d);
    }
  }

  clipDetectionsToImage(
    detections, original_size);

  return detections;
}

void Inference::buildPixelMapsFromDetections(
  const std::vector<Detection> & detections,
  const cv::Size & image_size,
  cv::Mat & class_map,
  cv::Mat & confidence_map)
{
  buildSemanticMaps(
    detections,
    image_size,
    class_map,
    confidence_map);
}

SemanticOutput Inference::infer(const cv::Mat & image)
{
  if (!loaded_) {
    throw std::runtime_error(
      "Semantic model has not been loaded.");
  }

  if (image.empty()) {
    throw std::runtime_error(
      "Received empty image.");
  }

  SemanticOutput output;
  output.model_name = model_path_;
  output.model_backend = backend_;

  const auto inference_start =
    std::chrono::steady_clock::now();

  float scale = 1.0f;
  int pad_x = 0;
  int pad_y = 0;

  cv::Mat input =
    letterbox(image, scale, pad_x, pad_y);

  cv::Mat blob = cv::dnn::blobFromImage(
    input,
    1.0 / 255.0,
    cv::Size(input_width_, input_height_),
    cv::Scalar(),
    true,
    false,
    CV_32F);

  net_.setInput(blob);

  std::vector<cv::Mat> outputs;
  net_.forward(
    outputs,
    net_.getUnconnectedOutLayersNames());

  const auto inference_end =
    std::chrono::steady_clock::now();

  output.inference_time_ms =
    std::chrono::duration<double, std::milli>(
      inference_end - inference_start).count();

  const auto postprocess_start =
    std::chrono::steady_clock::now();

  output.detections =
    decodeYoloDetection(
      outputs,
      image.size(),
      scale,
      pad_x,
      pad_y);

  buildPixelMapsFromDetections(
    output.detections,
    image.size(),
    output.class_map,
    output.confidence_map);

  const auto postprocess_end =
    std::chrono::steady_clock::now();

  output.postprocess_time_ms =
    std::chrono::duration<double, std::milli>(
      postprocess_end - postprocess_start).count();

  return output;
}

}  // namespace semantic_perception
