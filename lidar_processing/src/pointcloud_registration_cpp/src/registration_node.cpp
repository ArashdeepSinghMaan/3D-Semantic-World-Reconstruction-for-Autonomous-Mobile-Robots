#include <chrono>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>

#include <Eigen/Dense>

#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/string.hpp"

#include "pointcloud_registration/icp.hpp"
#include "pointcloud_registration/metrics.hpp"

namespace pointcloud_registration
{

class RegistrationNode : public rclcpp::Node
{
public:
  RegistrationNode()
  : Node("pointcloud_registration")
  {
    input_topic_ = declare_parameter<std::string>(
      "input.topic", "/lidar/points/processed");
    aligned_topic_ = declare_parameter<std::string>(
      "output.aligned_topic", "/registration/aligned");
    target_topic_ = declare_parameter<std::string>(
      "output.target_topic", "/registration/target");
    metrics_topic_ = declare_parameter<std::string>(
      "output.metrics_topic", "/registration/metrics");

    config_.method = declare_parameter<std::string>(
      "icp.method", "point_to_point");
    config_.voxel_size = declare_parameter<double>(
      "icp.voxel_size", 0.10);
    config_.max_correspondence_distance = declare_parameter<double>(
      "icp.max_correspondence_distance", 0.30);
    config_.max_iterations = declare_parameter<int>(
      "icp.max_iterations", 50);
    config_.transformation_epsilon = declare_parameter<double>(
      "icp.transformation_epsilon", 1e-6);
    config_.euclidean_fitness_epsilon = declare_parameter<double>(
      "icp.euclidean_fitness_epsilon", 1e-6);

    config_.multiscale_enabled = declare_parameter<bool>(
      "icp.multiscale.enabled", true);
    

    config_.voxel_sizes = declare_parameter<std::vector<double>>(
    "icp.multiscale.voxel_sizes",
    std::vector<double>{0.20, 0.10, 0.05});

  config_.correspondence_factors = declare_parameter<std::vector<double>>(
    "icp.multiscale.correspondence_factors",
    std::vector<double>{2.5, 2.0, 1.5});

  const auto iterations_param =
  declare_parameter<std::vector<int64_t>>(
    "icp.multiscale.iterations",
    std::vector<int64_t>{40, 30, 20});

config_.iterations.clear();
config_.iterations.reserve(iterations_param.size());

for (const auto value : iterations_param) {
  config_.iterations.push_back(static_cast<int>(value));
}


    config_.normal_k = declare_parameter<int>(
      "icp.normal_k", 30);
    config_.normal_radius_factor = declare_parameter<double>(
      "icp.normal_radius_factor", 2.5);

    const auto tx = declare_parameter<double>(
      "initial_transform.translation_xyz.x", 0.0);
    const auto ty = declare_parameter<double>(
      "initial_transform.translation_xyz.y", 0.0);
    const auto tz = declare_parameter<double>(
      "initial_transform.translation_xyz.z", 0.0);

    const auto roll = declare_parameter<double>(
      "initial_transform.rotation_rpy_rad.roll", 0.0);
    const auto pitch = declare_parameter<double>(
      "initial_transform.rotation_rpy_rad.pitch", 0.0);
    const auto yaw = declare_parameter<double>(
      "initial_transform.rotation_rpy_rad.yaw", 0.0);

    initial_transform_ = Eigen::Matrix4f::Identity();

    Eigen::AngleAxisf rx(
      static_cast<float>(roll), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf ry(
      static_cast<float>(pitch), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rz(
      static_cast<float>(yaw), Eigen::Vector3f::UnitZ());

    initial_transform_.block<3, 3>(0, 0) =
      (rz * ry * rx).toRotationMatrix();
    initial_transform_(0, 3) = static_cast<float>(tx);
    initial_transform_(1, 3) = static_cast<float>(ty);
    initial_transform_(2, 3) = static_cast<float>(tz);

    aligned_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      aligned_topic_, 10);
    target_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      target_topic_, 10);
    metrics_pub_ = create_publisher<std_msgs::msg::String>(
      metrics_topic_, 10);

    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    qos.best_effort();
    qos.durability_volatile();

    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_,
      qos,
      std::bind(
        &RegistrationNode::cloudCallback, this,
        std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "Phase 3 registration started. Input: %s, method: %s, multiscale: %s",
      input_topic_.c_str(),
      config_.method.c_str(),
      config_.multiscale_enabled ? "true" : "false");
  }

private:
  pcl::PointCloud<pcl::PointXYZ>::Ptr convertCloud(
    const sensor_msgs::msg::PointCloud2 & msg)
  {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(msg, *cloud);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
    return cloud;
  }

  static std::string matrixToString(const Eigen::Matrix4f & T)
  {
    std::ostringstream out;
    out << "[";
    for (int r = 0; r < 4; ++r) {
      if (r > 0) {
        out << "; ";
      }
      for (int c = 0; c < 4; ++c) {
        if (c > 0) {
          out << ", ";
        }
        out << T(r, c);
      }
    }
    out << "]";
    return out.str();
  }

  void cloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    auto current = convertCloud(*msg);

    if (current->size() < 20) {
      RCLCPP_WARN(
        get_logger(),
        "Skipping frame with only %zu valid points.",
        current->size());
      return;
    }

    if (!previous_) {
      previous_ = current;
      previous_header_ = msg->header;
      frame_count_ = 1;

      RCLCPP_INFO(
        get_logger(),
        "Stored first frame: %zu points.",
        current->size());
      return;
    }

    try {
      ICPResult result = icp_.run(
        previous_, current, initial_transform_, config_);

      const auto metrics =
        calculateTransformMetrics(result.transformation, initial_transform_);

      pcl::PointCloud<pcl::PointXYZ> aligned;
      pcl::transformPointCloud(
        *previous_, aligned, result.transformation);

      sensor_msgs::msg::PointCloud2 aligned_msg;
      pcl::toROSMsg(aligned, aligned_msg);
      aligned_msg.header = msg->header;
      aligned_pub_->publish(aligned_msg);

      sensor_msgs::msg::PointCloud2 target_msg;
      pcl::toROSMsg(*current, target_msg);
      target_msg.header = msg->header;
      target_pub_->publish(target_msg);

      std_msgs::msg::String metrics_msg;
      std::ostringstream json;
      json << "{"
           << "\"frame_pair\":" << frame_count_
           << ",\"converged\":" << (result.converged ? "true" : "false")
           << ",\"fitness\":" << result.fitness_score
           << ",\"inlier_rmse\":" << result.inlier_rmse
           << ",\"translation_m\":" << metrics.translation_magnitude
           << ",\"rotation_deg\":" << metrics.rotation_angle_deg
           << ",\"delta_translation_m\":" << metrics.delta_translation_m
           << ",\"delta_rotation_deg\":" << metrics.delta_rotation_deg
           << ",\"processing_time_ms\":" << result.processing_time_ms
           << ",\"source_points\":" << result.source_points
           << ",\"target_points\":" << result.target_points
           << ",\"transform\":\"" << matrixToString(result.transformation)
           << "\"}";
      metrics_msg.data = json.str();
      metrics_pub_->publish(metrics_msg);

      RCLCPP_INFO(
        get_logger(),
        "Pair %zu | converged=%s fitness=%.5f rmse=%.5f "
        "translation=%.3f m rotation=%.2f deg time=%.1f ms",
        frame_count_,
        result.converged ? "true" : "false",
        result.fitness_score,
        result.inlier_rmse,
        metrics.translation_magnitude,
        metrics.rotation_angle_deg,
        result.processing_time_ms);

    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        get_logger(),
        "ICP failed for frame pair %zu: %s",
        frame_count_, e.what());
    }

    previous_ = current;
    previous_header_ = msg->header;
    ++frame_count_;
  }

  std::string input_topic_;
  std::string aligned_topic_;
  std::string target_topic_;
  std::string metrics_topic_;

  ICPConfig config_;
  Eigen::Matrix4f initial_transform_{Eigen::Matrix4f::Identity()};

  ICP icp_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr previous_;
  std_msgs::msg::Header previous_header_;
  std::size_t frame_count_{0};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr target_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr metrics_pub_;
};

}  // namespace pointcloud_registration

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node =
    std::make_shared<pointcloud_registration::RegistrationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
