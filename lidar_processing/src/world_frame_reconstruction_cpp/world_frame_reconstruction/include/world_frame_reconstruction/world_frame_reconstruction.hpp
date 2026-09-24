#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Transform.h>

namespace world_frame_reconstruction {

struct SemanticPoint {
  float x{0.0f};
  float y{0.0f};
  float z{0.0f};
  int32_t class_id{-1};
  float confidence{0.0f};
  double timestamp_sec{0.0};
};

class WorldFrameReconstructionNode : public rclcpp::Node {
public:
  explicit WorldFrameReconstructionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  bool readSemanticCloud(const sensor_msgs::msg::PointCloud2 & msg, std::vector<SemanticPoint> & points) const;
  bool lookupTransform(const std::string & source_frame, const rclcpp::Time & stamp, tf2::Transform & transform);
  void publishWorldCloud(const rclcpp::Time & stamp);
  sensor_msgs::msg::PointCloud2 makeCloudMessage(const std::vector<SemanticPoint> & points,
                                                  const std::string & frame,
                                                  const rclcpp::Time & stamp) const;
  void resetAccumulation();

  std::string input_topic_;
  std::string output_topic_;
  std::string accumulation_topic_;
  std::string world_frame_;
  std::string source_frame_override_;
  double tf_timeout_sec_{0.10};
  double min_confidence_{0.0};
  bool accumulate_{true};
  bool publish_accumulated_{true};
  bool keep_unknown_{true};
  int32_t unknown_class_id_{-1};
  std::size_t max_points_{1000000};
  std::size_t publish_every_n_{1};
  std::size_t frame_count_{0};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr world_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr accumulated_pub_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  std::deque<SemanticPoint> accumulated_points_;
  mutable std::mutex mutex_;
};

}  // namespace world_frame_reconstruction
