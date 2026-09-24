#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>

#include "cv_bridge/cv_bridge.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "camera_lidar_calibration/calibration.hpp"
#include "camera_lidar_calibration/projection.hpp"
#include "camera_lidar_calibration/visualization.hpp"

namespace camera_lidar_calibration
{

class CameraLidarProjectionNode : public rclcpp::Node
{
public:
  CameraLidarProjectionNode()
  : Node("camera_lidar_projection")
  {
    lidar_topic_ = declare_parameter<std::string>(
      "topics.lidar", "/os_cloud_node/points");
    image_topic_ = declare_parameter<std::string>(
      "topics.image", "/stereo/frame_left/image_raw/compressed");
    camera_info_topic_ = declare_parameter<std::string>(
      "topics.camera_info", "/stereo/frame_left/camera_info");
    overlay_topic_ = declare_parameter<std::string>(
      "topics.overlay", "/camera_lidar/overlay");
    projected_cloud_topic_ = declare_parameter<std::string>(
      "topics.projected_cloud", "/camera_lidar/projected_points");

    use_camera_info_ = declare_parameter<bool>(
      "calibration.use_camera_info", true);

    camera_.width = declare_parameter<int>(
      "calibration.image_width", 0);
    camera_.height = declare_parameter<int>(
      "calibration.image_height", 0);
    camera_.fx = declare_parameter<double>(
      "calibration.fx", 0.0);
    camera_.fy = declare_parameter<double>(
      "calibration.fy", 0.0);
    camera_.cx = declare_parameter<double>(
      "calibration.cx", 0.0);
    camera_.cy = declare_parameter<double>(
      "calibration.cy", 0.0);

    min_depth_ = declare_parameter<double>(
      "projection.min_depth", 0.10);
    apply_distortion_ = declare_parameter<bool>(
      "projection.apply_distortion", true);

    enable_occlusion_ = declare_parameter<bool>(
      "projection.occlusion.enabled", true);
    occlusion_window_ = declare_parameter<int>(
      "projection.occlusion.window_size", 1);
    occlusion_depth_tolerance_ = declare_parameter<double>(
      "projection.occlusion.depth_tolerance_m", 0.15);

    point_radius_ = declare_parameter<int>(
      "visualization.point_radius", 3);
    overlay_alpha_ = declare_parameter<double>(
      "visualization.alpha", 0.85);
    color_by_depth_ = declare_parameter<bool>(
      "visualization.color_by_depth", true);

    const double tx = declare_parameter<double>(
      "extrinsic.translation_xyz.x", 0.0);
    const double ty = declare_parameter<double>(
      "extrinsic.translation_xyz.y", 0.0);
    const double tz = declare_parameter<double>(
      "extrinsic.translation_xyz.z", 0.0);
    const double roll = declare_parameter<double>(
      "extrinsic.rotation_rpy_rad.roll", 0.0);
    const double pitch = declare_parameter<double>(
      "extrinsic.rotation_rpy_rad.pitch", 0.0);
    const double yaw = declare_parameter<double>(
      "extrinsic.rotation_rpy_rad.yaw", 0.0);

    extrinsic_.T_camera_lidar =
      makeTransform(tx, ty, tz, roll, pitch, yaw);

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(
        &CameraLidarProjectionNode::imageCallback,
        this, std::placeholders::_1));

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(
        &CameraLidarProjectionNode::cloudCallback,
        this, std::placeholders::_1));

    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(
        &CameraLidarProjectionNode::cameraInfoCallback,
        this, std::placeholders::_1));

    overlay_pub_ = create_publisher<sensor_msgs::msg::Image>(
      overlay_topic_, 10);

    projected_cloud_pub_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
        projected_cloud_topic_, 10);

    RCLCPP_INFO(
      get_logger(),
      "Camera-LiDAR projection started.");
    RCLCPP_INFO(
      get_logger(),
      "LiDAR: %s | Image: %s | CameraInfo: %s",
      lidar_topic_.c_str(),
      image_topic_.c_str(),
      camera_info_topic_.c_str());
  }

private:
  void cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    if (!use_camera_info_ || camera_info_received_) {
      return;
    }

    camera_.width = static_cast<int>(msg->width);
    camera_.height = static_cast<int>(msg->height);

    // P is the projection matrix. K is the intrinsic matrix.
    // Use P for the rectified image geometry.
    camera_.fx = msg->p[0];
    camera_.fy = msg->p[5];
    camera_.cx = msg->p[2];
    camera_.cy = msg->p[6];

    // For raw images, K + D should be used.
    // The package defaults to using K-equivalent intrinsics from P
    // for a generic projection path; users can disable distortion
    // when working with rectified images.
    camera_.distortion.assign(msg->d.begin(), msg->d.end());
    camera_.distortion_model = msg->distortion_model;

    camera_info_received_ = true;

    RCLCPP_INFO(
      get_logger(),
      "Received CameraInfo: %dx%d fx=%.3f fy=%.3f cx=%.3f cy=%.3f model=%s",
      camera_.width, camera_.height,
      camera_.fx, camera_.fy,
      camera_.cx, camera_.cy,
      camera_.distortion_model.c_str());
  }

  void imageCallback(
    const sensor_msgs::msg::Image::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(image_mutex_);
    try {
      auto cv_ptr = cv_bridge::toCvCopy(
        msg, sensor_msgs::image_encodings::BGR8);
      latest_image_ = cv_ptr->image;
      latest_image_header_ = msg->header;
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(
        get_logger(),
        "Image conversion failed: %s", e.what());
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr convertCloud(
    const sensor_msgs::msg::PointCloud2 & msg)
  {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(msg, *cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr finite =
      std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    finite->reserve(cloud->size());

    for (const auto & p : cloud->points) {
      if (std::isfinite(p.x) &&
          std::isfinite(p.y) &&
          std::isfinite(p.z)) {
        finite->push_back(p);
      }
    }

    return finite;
  }

  std::vector<ProjectedPoint> applyOcclusion(
    const std::vector<ProjectedPoint> & points,
    const cv::Mat & image) const
  {
    if (!enable_occlusion_) {
      return points;
    }

    const int width = image.cols;
    const int height = image.rows;

    // Z-buffer at pixel level. The closest LiDAR return wins.
    cv::Mat depth_buffer(
      height, width, CV_64FC1,
      cv::Scalar(std::numeric_limits<double>::infinity()));

    std::vector<ProjectedPoint> visible;
    visible.reserve(points.size());

    for (const auto & p : points) {
      if (!p.valid) {
        continue;
      }

      const int u = static_cast<int>(std::lround(p.u));
      const int v = static_cast<int>(std::lround(p.v));

      if (u < 0 || v < 0 ||
          u >= width || v >= height) {
        continue;
      }

      bool accepted = false;

      const int w = std::max(0, occlusion_window_);
      const int u0 = std::max(0, u - w);
      const int u1 = std::min(width - 1, u + w);
      const int v0 = std::max(0, v - w);
      const int v1 = std::min(height - 1, v + w);

      double & center_depth = depth_buffer.at<double>(v, u);

      if (p.depth + occlusion_depth_tolerance_ < center_depth) {
        center_depth = p.depth;
        accepted = true;
      } else if (
        std::abs(p.depth - center_depth) <=
        occlusion_depth_tolerance_) {
        accepted = true;
      }

      // Populate neighboring depth cells so dense projected LiDAR
      // does not create excessive duplicate overlays.
      if (accepted) {
        for (int yy = v0; yy <= v1; ++yy) {
          for (int xx = u0; xx <= u1; ++xx) {
            double & d = depth_buffer.at<double>(yy, xx);
            if (p.depth < d) {
              d = p.depth;
            }
          }
        }
        visible.push_back(p);
      }
    }

    return visible;
  }

  void cloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!camera_.valid()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Waiting for valid CameraInfo/calibration.");
      return;
    }

    cv::Mat image;
    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      if (latest_image_.empty()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Waiting for an image.");
        return;
      }
      image = latest_image_.clone();
    }

    auto cloud = convertCloud(*msg);

    std::vector<Eigen::Vector3d> lidar_points;
    lidar_points.reserve(cloud->size());

    for (const auto & p : cloud->points) {
      lidar_points.emplace_back(
        static_cast<double>(p.x),
        static_cast<double>(p.y),
        static_cast<double>(p.z));
    }

    try {
      auto projected = projector_.project(
        lidar_points,
        camera_,
        extrinsic_,
        min_depth_,
        apply_distortion_);

      auto visible = applyOcclusion(projected, image);

      cv::Mat overlay = drawProjectedPoints(
        image,
        visible,
        point_radius_,
        overlay_alpha_,
        color_by_depth_);

      std::ostringstream status;
      status
        << "Projected: " << visible.size()
        << " / " << cloud->size();

      overlay = drawCalibrationText(overlay, status.str());

      auto out = cv_bridge::CvImage(
        msg->header,
        sensor_msgs::image_encodings::BGR8,
        overlay).toImageMsg();

      overlay_pub_->publish(*out);

      publishProjectedCloud(
        visible,
        msg->header);

    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        get_logger(),
        "Projection failed: %s", e.what());
    }
  }

  void publishProjectedCloud(
    const std::vector<ProjectedPoint> & points,
    const std_msgs::msg::Header & header)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud =
      std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    cloud->reserve(points.size());

    for (const auto & p : points) {
      if (!p.valid) {
        continue;
      }

      pcl::PointXYZ q;
      q.x = static_cast<float>(p.camera_point.x());
      q.y = static_cast<float>(p.camera_point.y());
      q.z = static_cast<float>(p.camera_point.z());
      cloud->push_back(q);
    }

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header = header;
    projected_cloud_pub_->publish(msg);
  }

  std::string lidar_topic_;
  std::string image_topic_;
  std::string camera_info_topic_;
  std::string overlay_topic_;
  std::string projected_cloud_topic_;

  bool use_camera_info_{true};
  bool camera_info_received_{false};

  CameraCalibration camera_;
  ExtrinsicCalibration extrinsic_;
  Projector projector_;

  double min_depth_{0.1};
  bool apply_distortion_{true};

  bool enable_occlusion_{true};
  int occlusion_window_{1};
  double occlusion_depth_tolerance_{0.15};

  int point_radius_{3};
  double overlay_alpha_{0.85};
  bool color_by_depth_{true};

  cv::Mat latest_image_;
  std_msgs::msg::Header latest_image_header_;
  std::mutex image_mutex_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlay_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr projected_cloud_pub_;
};

}  // namespace camera_lidar_calibration

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node =
    std::make_shared<
      camera_lidar_calibration::CameraLidarProjectionNode>();

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
