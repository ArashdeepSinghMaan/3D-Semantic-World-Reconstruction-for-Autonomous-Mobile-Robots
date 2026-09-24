#include "semantic_3d_fusion/projection.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::PointCloud2;
using std_msgs::msg::Float32MultiArray;
using std_msgs::msg::Int32MultiArray;

namespace
{

double degToRad(double d) { return d * M_PI / 180.0; }

Eigen::Matrix3d rpyToRotation(double roll, double pitch, double yaw)
{
  Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());
  return (rz * ry * rx).toRotationMatrix();
}

struct SemanticMaps
{
  std::vector<int32_t> classes;
  std::vector<float> confidence;
  int width{0};
  int height{0};
  rclcpp::Time received_time{0, 0, RCL_ROS_TIME};
  bool valid{false};
};

}  // namespace

namespace semantic_3d_fusion
{

class SemanticFusionNode : public rclcpp::Node
{
public:
  SemanticFusionNode()
  : Node("semantic_3d_fusion"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
  {
    declareParameters();
    loadParameters();

    projector_.setExtrinsic(extrinsic_);

    camera_info_sub_ = create_subscription<CameraInfo>(
      camera_info_topic_, rclcpp::SensorDataQoS(),
      std::bind(&SemanticFusionNode::cameraInfoCallback, this, std::placeholders::_1));

    class_sub_ = create_subscription<Int32MultiArray>(
      class_map_topic_, rclcpp::SensorDataQoS(),
      std::bind(&SemanticFusionNode::classMapCallback, this, std::placeholders::_1));

    confidence_sub_ = create_subscription<Float32MultiArray>(
      confidence_map_topic_, rclcpp::SensorDataQoS(),
      std::bind(&SemanticFusionNode::confidenceMapCallback, this, std::placeholders::_1));

    lidar_sub_ = create_subscription<PointCloud2>(
      lidar_topic_, rclcpp::SensorDataQoS(),
      std::bind(&SemanticFusionNode::lidarCallback, this, std::placeholders::_1));

    semantic_cloud_pub_ = create_publisher<PointCloud2>(semantic_cloud_topic_, rclcpp::SensorDataQoS());
    unassigned_pub_ = create_publisher<PointCloud2>(unassigned_cloud_topic_, rclcpp::SensorDataQoS());

    RCLCPP_INFO(get_logger(), "Phase 6 Semantic 3D Fusion started");
    RCLCPP_INFO(get_logger(), "LiDAR: %s", lidar_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Class map: %s", class_map_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Confidence map: %s", confidence_map_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Camera info: %s", camera_info_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Output: %s", semantic_cloud_topic_.c_str());
  }

private:
  void declareParameters()
  {
    declare_parameter<std::string>("topics.lidar", "/lidar/points/processed");
    declare_parameter<std::string>("topics.class_map", "/semantic/class_map");
    declare_parameter<std::string>("topics.confidence_map", "/semantic/confidence_map");
    declare_parameter<std::string>("topics.camera_info", "/stereo/frame_left/camera_info");
    declare_parameter<std::string>("topics.semantic_cloud", "/semantic_3d/points");
    declare_parameter<std::string>("topics.unassigned_cloud", "/semantic_3d/unassigned_points");

    declare_parameter<bool>("camera.use_camera_info", true);
    declare_parameter<int>("camera.width", 0);
    declare_parameter<int>("camera.height", 0);
    declare_parameter<double>("camera.fx", 0.0);
    declare_parameter<double>("camera.fy", 0.0);
    declare_parameter<double>("camera.cx", 0.0);
    declare_parameter<double>("camera.cy", 0.0);

    declare_parameter<std::string>("frames.lidar", "");
    declare_parameter<std::string>("frames.camera", "");
    declare_parameter<bool>("extrinsic.use_tf", false);
    declare_parameter<std::string>("extrinsic.lidar_frame", "");
    declare_parameter<std::string>("extrinsic.camera_frame", "");
    declare_parameter<double>("extrinsic.translation.x", 0.0);
    declare_parameter<double>("extrinsic.translation.y", 0.0);
    declare_parameter<double>("extrinsic.translation.z", 0.0);
    declare_parameter<double>("extrinsic.rotation.roll", 0.0);
    declare_parameter<double>("extrinsic.rotation.pitch", 0.0);
    declare_parameter<double>("extrinsic.rotation.yaw", 0.0);

    declare_parameter<double>("fusion.min_depth", 0.10);
    declare_parameter<double>("fusion.max_depth", 50.0);
    declare_parameter<double>("fusion.min_confidence", 0.0);
    declare_parameter<int>("fusion.invalid_class_id", -1);
    declare_parameter<bool>("fusion.keep_unknown", true);
    declare_parameter<bool>("fusion.nearest_pixel", true);
    declare_parameter<bool>("fusion.use_bilinear", false);
    declare_parameter<bool>("fusion.require_matching_map_sizes", true);
    declare_parameter<int>("fusion.pixel_radius", 0);
  }

  void loadParameters()
  {
    lidar_topic_ = get_parameter("topics.lidar").as_string();
    class_map_topic_ = get_parameter("topics.class_map").as_string();
    confidence_map_topic_ = get_parameter("topics.confidence_map").as_string();
    camera_info_topic_ = get_parameter("topics.camera_info").as_string();
    semantic_cloud_topic_ = get_parameter("topics.semantic_cloud").as_string();
    unassigned_cloud_topic_ = get_parameter("topics.unassigned_cloud").as_string();

    use_camera_info_ = get_parameter("camera.use_camera_info").as_bool();
    camera_.width = get_parameter("camera.width").as_int();
    camera_.height = get_parameter("camera.height").as_int();
    camera_.fx = get_parameter("camera.fx").as_double();
    camera_.fy = get_parameter("camera.fy").as_double();
    camera_.cx = get_parameter("camera.cx").as_double();
    camera_.cy = get_parameter("camera.cy").as_double();

    lidar_frame_param_ = get_parameter("frames.lidar").as_string();
    camera_frame_param_ = get_parameter("frames.camera").as_string();
    use_tf_ = get_parameter("extrinsic.use_tf").as_bool();
    tf_lidar_frame_ = get_parameter("extrinsic.lidar_frame").as_string();
    tf_camera_frame_ = get_parameter("extrinsic.camera_frame").as_string();

    extrinsic_.translation.x() = get_parameter("extrinsic.translation.x").as_double();
    extrinsic_.translation.y() = get_parameter("extrinsic.translation.y").as_double();
    extrinsic_.translation.z() = get_parameter("extrinsic.translation.z").as_double();
    const double roll = get_parameter("extrinsic.rotation.roll").as_double();
    const double pitch = get_parameter("extrinsic.rotation.pitch").as_double();
    const double yaw = get_parameter("extrinsic.rotation.yaw").as_double();
    extrinsic_.rotation = rpyToRotation(roll, pitch, yaw);

    min_depth_ = get_parameter("fusion.min_depth").as_double();
    max_depth_ = get_parameter("fusion.max_depth").as_double();
    min_confidence_ = get_parameter("fusion.min_confidence").as_double();
    invalid_class_id_ = get_parameter("fusion.invalid_class_id").as_int();
    keep_unknown_ = get_parameter("fusion.keep_unknown").as_bool();
    nearest_pixel_ = get_parameter("fusion.nearest_pixel").as_bool();
    use_bilinear_ = get_parameter("fusion.use_bilinear").as_bool();
    require_matching_map_sizes_ = get_parameter("fusion.require_matching_map_sizes").as_bool();
    pixel_radius_ = get_parameter("fusion.pixel_radius").as_int();

    if (use_bilinear_) {
      RCLCPP_WARN(get_logger(), "fusion.use_bilinear is reserved; current implementation uses nearest-neighbor sampling.");
    }
    projector_.setCamera(camera_);
  }

  void cameraInfoCallback(const CameraInfo::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (use_camera_info_) {
      camera_.width = static_cast<int>(msg->width);
      camera_.height = static_cast<int>(msg->height);
      if (msg->p[0] > 0.0) {
        camera_.fx = msg->p[0];
        camera_.fy = msg->p[5];
        camera_.cx = msg->p[2];
        camera_.cy = msg->p[6];
      } else {
        camera_.fx = msg->k[0];
        camera_.fy = msg->k[4];
        camera_.cx = msg->k[2];
        camera_.cy = msg->k[5];
      }
      projector_.setCamera(camera_);
      camera_ready_ = camera_.fx > 0.0 && camera_.fy > 0.0 && camera_.width > 0 && camera_.height > 0;
    }
  }

  static bool extractArrayDimensions(const std_msgs::msg::MultiArrayLayout &layout, int &width, int &height)
  {
    if (layout.dim.size() >= 2) {
      // ROS MultiArray convention: dim[0] is the outer/row dimension, dim[1] is columns.
      height = static_cast<int>(layout.dim[0].size);
      width = static_cast<int>(layout.dim[1].size);
      return width > 0 && height > 0;
    }
    if (layout.dim.size() == 1 && layout.dim[0].size > 0) {
      // Cannot uniquely recover a 2D shape from a flat array. Keep width/height from CameraInfo if possible.
      return false;
    }
    return false;
  }

  void classMapCallback(const Int32MultiArray::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    semantic_maps_.classes = msg->data;
    int w = 0, h = 0;
    const bool dims_ok = extractArrayDimensions(msg->layout, w, h);
    if (dims_ok) {
      semantic_maps_.width = w;
      semantic_maps_.height = h;
    } else if (camera_ready_) {
      semantic_maps_.width = camera_.width;
      semantic_maps_.height = camera_.height;
    }
    semantic_maps_.received_time = now();
    semantic_maps_.valid = !semantic_maps_.classes.empty();
    checkMapConsistencyLocked();
  }

  void confidenceMapCallback(const Float32MultiArray::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    semantic_maps_.confidence = msg->data;
    int w = 0, h = 0;
    const bool dims_ok = extractArrayDimensions(msg->layout, w, h);
    if (dims_ok) {
      if (semantic_maps_.width == 0) semantic_maps_.width = w;
      if (semantic_maps_.height == 0) semantic_maps_.height = h;
    } else if (camera_ready_) {
      if (semantic_maps_.width == 0) semantic_maps_.width = camera_.width;
      if (semantic_maps_.height == 0) semantic_maps_.height = camera_.height;
    }
    semantic_maps_.received_time = now();
    checkMapConsistencyLocked();
  }

  void checkMapConsistencyLocked()
  {
    if (semantic_maps_.classes.empty() || semantic_maps_.confidence.empty()) {
      semantic_maps_.valid = false;
      return;
    }
    const size_t expected = static_cast<size_t>(semantic_maps_.width) * static_cast<size_t>(semantic_maps_.height);
    const bool size_ok = expected > 0 && semantic_maps_.classes.size() == expected && semantic_maps_.confidence.size() == expected;
    semantic_maps_.valid = size_ok;
    if (!size_ok && require_matching_map_sizes_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "Semantic maps are not ready/consistent: class=%zu confidence=%zu expected=%zu (W=%d H=%d)",
        semantic_maps_.classes.size(), semantic_maps_.confidence.size(), expected,
        semantic_maps_.width, semantic_maps_.height);
    }
  }

  bool getSemanticAtPixel(int u, int v, int &class_id, float &confidence) const
  {
    if (!semantic_maps_.valid || semantic_maps_.width <= 0 || semantic_maps_.height <= 0) return false;
    if (u < 0 || v < 0 || u >= semantic_maps_.width || v >= semantic_maps_.height) return false;
    const size_t idx = static_cast<size_t>(v) * semantic_maps_.width + static_cast<size_t>(u);
    if (idx >= semantic_maps_.classes.size() || idx >= semantic_maps_.confidence.size()) return false;
    class_id = semantic_maps_.classes[idx];
    confidence = semantic_maps_.confidence[idx];
    if (!std::isfinite(confidence)) confidence = 0.0f;
    return true;
  }

  bool sampleSemantic(double u, double v, int &class_id, float &confidence) const
  {
    if (!semantic_maps_.valid) return false;

    if (nearest_pixel_ || pixel_radius_ <= 0) {
      const int ui = static_cast<int>(std::floor(u + 0.5));
      const int vi = static_cast<int>(std::floor(v + 0.5));
      return getSemanticAtPixel(ui, vi, class_id, confidence);
    }

    bool found = false;
    float best_conf = -1.0f;
    const int uc = static_cast<int>(std::floor(u + 0.5));
    const int vc = static_cast<int>(std::floor(v + 0.5));
    for (int dy = -pixel_radius_; dy <= pixel_radius_; ++dy) {
      for (int dx = -pixel_radius_; dx <= pixel_radius_; ++dx) {
        int c = invalid_class_id_;
        float conf = 0.0f;
        if (getSemanticAtPixel(uc + dx, vc + dy, c, conf) && conf > best_conf) {
          best_conf = conf;
          class_id = c;
          confidence = conf;
          found = true;
        }
      }
    }
    return found;
  }

  bool resolveExtrinsic(const std::string &source_frame, const std::string &target_frame,
                        const rclcpp::Time &stamp, Extrinsic &out)
  {
    if (!use_tf_) {
      out = extrinsic_;
      return true;
    }

    const std::string source = tf_lidar_frame_.empty() ? source_frame : tf_lidar_frame_;
    const std::string target = tf_camera_frame_.empty() ? target_frame : tf_camera_frame_;
    if (source.empty() || target.empty()) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 3000,
        "extrinsic.use_tf=true but LiDAR/camera TF frames are not available.");
      return false;
    }

    try {
      const auto tf_msg = tf_buffer_.lookupTransform(target, source, stamp, tf2::durationFromSec(tf_timeout_sec_));
      const auto &q = tf_msg.transform.rotation;
      Eigen::Quaterniond quat(q.w, q.x, q.y, q.z);
      out.rotation = quat.normalized().toRotationMatrix();
      out.translation = Eigen::Vector3d(
        tf_msg.transform.translation.x,
        tf_msg.transform.translation.y,
        tf_msg.transform.translation.z);
      return true;
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "Could not lookup LiDAR->camera TF: %s", ex.what());
      return false;
    }
  }

  void lidarCallback(const PointCloud2::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!camera_ready_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "Waiting for valid camera intrinsics.");
      return;
    }
    if (!semantic_maps_.valid) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "Waiting for valid class and confidence maps.");
      return;
    }

    Extrinsic active_extrinsic;
    const std::string lidar_frame = msg->header.frame_id.empty() ? lidar_frame_param_ : msg->header.frame_id;
    const std::string camera_frame = camera_frame_param_;
    if (!resolveExtrinsic(lidar_frame, camera_frame, msg->header.stamp, active_extrinsic)) return;
    projector_.setExtrinsic(active_extrinsic);

    const size_t total_points = static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height);
    PointCloud2 semantic_cloud;
    PointCloud2 unassigned_cloud;
    semantic_cloud.header = msg->header;
    unassigned_cloud.header = msg->header;
    semantic_cloud.height = 1;
    unassigned_cloud.height = 1;
    semantic_cloud.is_dense = false;
    unassigned_cloud.is_dense = false;

    sensor_msgs::PointCloud2Modifier semantic_mod(semantic_cloud);
    semantic_mod.setPointCloud2Fields(
      6,
      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32,
      "class_id", 1, sensor_msgs::msg::PointField::INT32,
      "confidence", 1, sensor_msgs::msg::PointField::FLOAT32,
      "timestamp", 1, sensor_msgs::msg::PointField::FLOAT64);

    sensor_msgs::PointCloud2Modifier unassigned_mod(unassigned_cloud);
    unassigned_mod.setPointCloud2Fields(
      3,
      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32);

    semantic_mod.resize(0);
    unassigned_mod.resize(0);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

    std::vector<float> sx, sy, sz, sconf;
    std::vector<int32_t> sclass;
    std::vector<double> sts;
    std::vector<float> ux, uy, uz;
    sx.reserve(total_points);
    sy.reserve(total_points);
    sz.reserve(total_points);
    sconf.reserve(total_points);
    sclass.reserve(total_points);
    sts.reserve(total_points);
    ux.reserve(total_points / 4 + 1);
    uy.reserve(total_points / 4 + 1);
    uz.reserve(total_points / 4 + 1);

    size_t projected = 0;
    size_t assigned = 0;
    size_t unknown = 0;
    size_t low_conf = 0;
    size_t invalid = 0;

    const double stamp_sec = static_cast<double>(msg->header.stamp.sec) +
      static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    for (size_t i = 0; i < total_points; ++i, ++iter_x, ++iter_y, ++iter_z) {
      const float x = *iter_x;
      const float y = *iter_y;
      const float z = *iter_z;
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        ++invalid;
        continue;
      }

      const Eigen::Vector3d p_lidar(x, y, z);
      const ProjectionResult pr = projector_.project(p_lidar);
      if (!pr.valid || pr.depth < min_depth_ || pr.depth > max_depth_) {
        ++invalid;
        continue;
      }
      ++projected;

      int class_id = invalid_class_id_;
      float confidence = 0.0f;
      if (!sampleSemantic(pr.u, pr.v, class_id, confidence)) {
        ++unknown;
        if (keep_unknown_) {
          ux.push_back(x); uy.push_back(y); uz.push_back(z);
        }
        continue;
      }
      if (confidence < min_confidence_) {
        ++low_conf;
        if (keep_unknown_) {
          ux.push_back(x); uy.push_back(y); uz.push_back(z);
        }
        continue;
      }
      if (class_id == invalid_class_id_) {
        ++unknown;
        if (keep_unknown_) {
          ux.push_back(x); uy.push_back(y); uz.push_back(z);
        }
        continue;
      }

      sx.push_back(x);
      sy.push_back(y);
      sz.push_back(z);
      sclass.push_back(class_id);
      sconf.push_back(confidence);
      sts.push_back(stamp_sec);
      ++assigned;
    }

    semantic_mod.resize(assigned);
    sensor_msgs::PointCloud2Iterator<float> out_x(semantic_cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(semantic_cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(semantic_cloud, "z");
    sensor_msgs::PointCloud2Iterator<int32_t> out_class(semantic_cloud, "class_id");
    sensor_msgs::PointCloud2Iterator<float> out_conf(semantic_cloud, "confidence");
    sensor_msgs::PointCloud2Iterator<double> out_stamp(semantic_cloud, "timestamp");
    for (size_t i = 0; i < assigned; ++i, ++out_x, ++out_y, ++out_z, ++out_class, ++out_conf, ++out_stamp) {
      *out_x = sx[i];
      *out_y = sy[i];
      *out_z = sz[i];
      *out_class = sclass[i];
      *out_conf = sconf[i];
      *out_stamp = sts[i];
    }

    unassigned_mod.resize(ux.size());
    sensor_msgs::PointCloud2Iterator<float> ux_it(unassigned_cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> uy_it(unassigned_cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> uz_it(unassigned_cloud, "z");
    for (size_t i = 0; i < ux.size(); ++i, ++ux_it, ++uy_it, ++uz_it) {
      *ux_it = ux[i];
      *uy_it = uy[i];
      *uz_it = uz[i];
    }

    semantic_cloud.width = static_cast<uint32_t>(assigned);
    unassigned_cloud.width = static_cast<uint32_t>(ux.size());
    semantic_cloud_pub_->publish(semantic_cloud);
    if (keep_unknown_) unassigned_pub_->publish(unassigned_cloud);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
      "Fusion: input=%zu projected=%zu assigned=%zu unknown=%zu low_conf=%zu invalid=%zu",
      total_points, projected, assigned, unknown, low_conf, invalid);
  }

  std::string lidar_topic_, class_map_topic_, confidence_map_topic_, camera_info_topic_;
  std::string semantic_cloud_topic_, unassigned_cloud_topic_;
  std::string lidar_frame_param_, camera_frame_param_;
  std::string tf_lidar_frame_, tf_camera_frame_;

  bool use_camera_info_{true};
  bool use_tf_{false};
  bool camera_ready_{false};
  bool keep_unknown_{true};
  bool nearest_pixel_{true};
  bool use_bilinear_{false};
  bool require_matching_map_sizes_{true};
  int invalid_class_id_{-1};
  int pixel_radius_{0};
  double min_depth_{0.10};
  double max_depth_{50.0};
  double min_confidence_{0.0};
  double tf_timeout_sec_{0.05};

  CameraModel camera_;
  Extrinsic extrinsic_;
  Projector projector_;
  SemanticMaps semantic_maps_;

  std::mutex mutex_;
  rclcpp::Subscription<CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<Int32MultiArray>::SharedPtr class_sub_;
  rclcpp::Subscription<Float32MultiArray>::SharedPtr confidence_sub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Publisher<PointCloud2>::SharedPtr semantic_cloud_pub_;
  rclcpp::Publisher<PointCloud2>::SharedPtr unassigned_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

}  // namespace semantic_3d_fusion

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<semantic_3d_fusion::SemanticFusionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
