#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "cv_bridge/cv_bridge.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
#include "std_msgs/msg/String.hpp"

#include "semantic_perception/inference.hpp"
#include "semantic_perception/visualization.hpp"

namespace semantic_perception
{

class SemanticPerceptionNode : public rclcpp::Node
{
public:
  SemanticPerceptionNode()
  : Node("semantic_perception")
  {
    image_topic_ = declare_parameter<std::string>(
      "topics.image", "/stereo/frame_left/image_raw");

    detection_topic_ = declare_parameter<std::string>(
      "topics.detections", "/semantic/detections");

    class_map_topic_ = declare_parameter<std::string>(
      "topics.class_map", "/semantic/class_map");

    confidence_map_topic_ = declare_parameter<std::string>(
      "topics.confidence_map", "/semantic/confidence_map");

    visualization_topic_ = declare_parameter<std::string>(
      "topics.visualization", "/semantic/visualization");

    model_path_ = declare_parameter<std::string>(
      "model.path", "");

    backend_ = declare_parameter<std::string>(
      "model.backend", "opencv");

    target_ = declare_parameter<std::string>(
      "model.target", "cpu");

    input_width_ = declare_parameter<int>(
      "model.input_width", 640);

    input_height_ = declare_parameter<int>(
      "model.input_height", 640);

    confidence_threshold_ = declare_parameter<double>(
      "model.confidence_threshold", 0.25);

    nms_threshold_ = declare_parameter<double>(
      "model.nms_threshold", 0.45);

    class_names_ = declare_parameter<std::vector<std::string>>(
      "classes.names", {});

    draw_labels_ = declare_parameter<bool>(
      "visualization.draw_labels", true);

    draw_semantic_map_ = declare_parameter<bool>(
      "visualization.draw_semantic_map", true);

    semantic_alpha_ = declare_parameter<double>(
      "visualization.semantic_alpha", 0.45);

    if (model_path_.empty()) {
      RCLCPP_WARN(
        get_logger(),
        "model.path is empty. Node will start, but inference will be disabled.");
    } else {
      if (!inference_.load(
          model_path_,
          backend_,
          target_,
          class_names_,
          input_width_,
          input_height_,
          static_cast<float>(confidence_threshold_),
          static_cast<float>(nms_threshold_))) {
        RCLCPP_ERROR(
          get_logger(),
          "Failed to load model: %s",
          model_path_.c_str());
      } else {
        RCLCPP_INFO(
          get_logger(),
          "Loaded semantic model: %s",
          model_path_.c_str());
      }
    }

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(
        &SemanticPerceptionNode::imageCallback,
        this,
        std::placeholders::_1));

    visualization_pub_ =
      create_publisher<sensor_msgs::msg::Image>(
        visualization_topic_, 10);

    detection_pub_ =
      create_publisher<std_msgs::msg::String>(
        detection_topic_, 10);

    class_map_pub_ =
      create_publisher<std_msgs::msg::Int32MultiArray>(
        class_map_topic_, 10);

    confidence_map_pub_ =
      create_publisher<std_msgs::msg::Float32MultiArray>(
        confidence_map_topic_, 10);

    RCLCPP_INFO(
      get_logger(),
      "Semantic perception node ready. Image: %s",
      image_topic_.c_str());
  }

private:
  void imageCallback(
    const sensor_msgs::msg::Image::SharedPtr msg)
  {
    if (!inference_.loaded()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Semantic model is not loaded. Set model.path.");
      return;
    }

    try {
      auto cv_ptr = cv_bridge::toCvCopy(
        msg,
        sensor_msgs::image_encodings::BGR8);

      SemanticOutput output =
        inference_.infer(cv_ptr->image);

      cv::Mat visualization =
        drawDetections(
          cv_ptr->image,
          output.detections,
          draw_labels_,
          2);

      if (draw_semantic_map_) {
        visualization =
          drawSemanticMap(
            visualization,
            output.class_map,
            output.confidence_map,
            static_cast<float>(semantic_alpha_));
      }

      visualization =
        drawStatus(
          visualization,
          output);

      auto image_msg =
        cv_bridge::CvImage(
          msg->header,
          sensor_msgs::image_encodings::BGR8,
          visualization).toImageMsg();

      visualization_pub_->publish(*image_msg);

      publishDetections(
        output.detections,
        msg->header);

      publishClassMap(
        output.class_map,
        msg->header);

      publishConfidenceMap(
        output.confidence_map,
        msg->header);

      RCLCPP_DEBUG(
        get_logger(),
        "Inference %.2f ms, postprocess %.2f ms, detections %zu",
        output.inference_time_ms,
        output.postprocess_time_ms,
        output.detections.size());

    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        get_logger(),
        "Semantic inference failed: %s",
        e.what());
    }
  }

  void publishDetections(
    const std::vector<Detection> & detections,
    const std_msgs::msg::Header &)
  {
    std_msgs::msg::String msg;
    std::ostringstream json;

    json << "{ \"detections\": [";

    for (std::size_t i = 0; i < detections.size(); ++i) {
      const auto & d = detections[i];

      if (i > 0) {
        json << ",";
      }

      json
        << "{"
        << "\"class_id\":" << d.class_id << ","
        << "\"class_name\":\"" << d.class_name << "\","
        << "\"confidence\":" << d.confidence << ","
        << "\"x\":" << d.bounding_box.x << ","
        << "\"y\":" << d.bounding_box.y << ","
        << "\"width\":" << d.bounding_box.width << ","
        << "\"height\":" << d.bounding_box.height
        << "}";
    }

    json << "] }";

    msg.data = json.str();
    detection_pub_->publish(msg);
  }

  void publishClassMap(
    const cv::Mat & class_map,
    const std_msgs::msg::Header &)
  {
    std_msgs::msg::Int32MultiArray msg;

    msg.layout.dim.resize(2);

    msg.layout.dim[0].label = "height";
    msg.layout.dim[0].size = class_map.rows;
    msg.layout.dim[0].stride =
      class_map.rows * class_map.cols;

    msg.layout.dim[1].label = "width";
    msg.layout.dim[1].size = class_map.cols;
    msg.layout.dim[1].stride = class_map.cols;

    msg.data.reserve(
      class_map.rows * class_map.cols);

    for (int y = 0; y < class_map.rows; ++y) {
      for (int x = 0; x < class_map.cols; ++x) {
        msg.data.push_back(
          class_map.at<int>(y, x));
      }
    }

    class_map_pub_->publish(msg);
  }

  void publishConfidenceMap(
    const cv::Mat & confidence_map,
    const std_msgs::msg::Header &)
  {
    std_msgs::msg::Float32MultiArray msg;

    msg.layout.dim.resize(2);

    msg.layout.dim[0].label = "height";
    msg.layout.dim[0].size = confidence_map.rows;
    msg.layout.dim[0].stride =
      confidence_map.rows * confidence_map.cols;

    msg.layout.dim[1].label = "width";
    msg.layout.dim[1].size = confidence_map.cols;
    msg.layout.dim[1].stride = confidence_map.cols;

    msg.data.reserve(
      confidence_map.rows * confidence_map.cols);

    for (int y = 0; y < confidence_map.rows; ++y) {
      for (int x = 0; x < confidence_map.cols; ++x) {
        msg.data.push_back(
          confidence_map.at<float>(y, x));
      }
    }

    confidence_map_pub_->publish(msg);
  }

  std::string image_topic_;
  std::string detection_topic_;
  std::string class_map_topic_;
  std::string confidence_map_topic_;
  std::string visualization_topic_;

  std::string model_path_;
  std::string backend_;
  std::string target_;

  int input_width_{640};
  int input_height_{640};

  double confidence_threshold_{0.25};
  double nms_threshold_{0.45};

  std::vector<std::string> class_names_;

  bool draw_labels_{true};
  bool draw_semantic_map_{true};
  double semantic_alpha_{0.45};

  Inference inference_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
    visualization_pub_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr
    detection_pub_;

  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr
    class_map_pub_;

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr
    confidence_map_pub_;
};

}  // namespace semantic_perception

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node =
    std::make_shared<
      semantic_perception::SemanticPerceptionNode>();

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
