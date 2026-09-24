#include "world_frame_reconstruction/world_frame_reconstruction.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace world_frame_reconstruction {

WorldFrameReconstructionNode::WorldFrameReconstructionNode(const rclcpp::NodeOptions & options)
: Node("world_frame_reconstruction", options)
{
  input_topic_ = declare_parameter<std::string>("topics.input", "/semantic_3d/fused_points");
  output_topic_ = declare_parameter<std::string>("topics.world_points", "/world_reconstruction/points");
  accumulation_topic_ = declare_parameter<std::string>("topics.accumulated_points", "/world_reconstruction/accumulated");
  world_frame_ = declare_parameter<std::string>("frames.world", "map");
  source_frame_override_ = declare_parameter<std::string>("frames.source_override", "");
  tf_timeout_sec_ = declare_parameter<double>("tf.timeout_sec", 0.10);
  min_confidence_ = declare_parameter<double>("filter.min_confidence", 0.0);
  keep_unknown_ = declare_parameter<bool>("filter.keep_unknown", true);
  unknown_class_id_ = declare_parameter<int>("filter.unknown_class_id", -1);
  accumulate_ = declare_parameter<bool>("accumulation.enabled", true);
  publish_accumulated_ = declare_parameter<bool>("accumulation.publish", true);
  max_points_ = static_cast<std::size_t>(declare_parameter<int64_t>("accumulation.max_points", 1000000));
  publish_every_n_ = static_cast<std::size_t>(std::max<int64_t>(1, declare_parameter<int64_t>("output.publish_every_n_frames", 1)));

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

  world_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, rclcpp::SensorDataQoS());
  accumulated_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(accumulation_topic_, rclcpp::SensorDataQoS());
  sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    input_topic_, rclcpp::SensorDataQoS(),
    std::bind(&WorldFrameReconstructionNode::cloudCallback, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "Phase 7 World Frame Reconstruction started");
  RCLCPP_INFO(get_logger(), "Input: %s", input_topic_.c_str());
  RCLCPP_INFO(get_logger(), "World frame: %s", world_frame_.c_str());
}

bool WorldFrameReconstructionNode::readSemanticCloud(
  const sensor_msgs::msg::PointCloud2 & msg, std::vector<SemanticPoint> & points) const
{
  bool has_x=false, has_y=false, has_z=false, has_class=false, has_conf=false;
  for (const auto & f : msg.fields) {
    has_x |= f.name == "x"; has_y |= f.name == "y"; has_z |= f.name == "z";
    has_class |= f.name == "class_id"; has_conf |= f.name == "confidence";
  }
  if (!(has_x && has_y && has_z)) {
    RCLCPP_ERROR(get_logger(), "Input cloud must contain x/y/z fields.");
    return false;
  }
  if (!has_class || !has_conf) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
      "Input cloud has no semantic class_id/confidence fields; values will be treated as unknown.");
  }

  try {
    sensor_msgs::PointCloud2ConstIterator<float> ix(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iy(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iz(msg, "z");
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<int32_t>> ic;
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> iq;
    if (has_class) ic = std::make_unique<sensor_msgs::PointCloud2ConstIterator<int32_t>>(msg, "class_id");
    if (has_conf) iq = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(msg, "confidence");

    const std::size_t n = static_cast<std::size_t>(msg.width) * msg.height;
    points.reserve(n);
    for (std::size_t i=0; i<n; ++i, ++ix, ++iy, ++iz) {
      const float x=*ix, y=*iy, z=*iz;
      int32_t cls = unknown_class_id_;
      float conf = 0.0f;
      if (ic) { cls = **ic; ++(*ic); }
      if (iq) { conf = **iq; ++(*iq); }
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(conf)) continue;
      if (conf < min_confidence_) continue;
      if (!keep_unknown_ && cls == unknown_class_id_) continue;
      points.push_back({x,y,z,cls,conf,msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9});
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Failed to read semantic PointCloud2: %s", e.what());
    return false;
  }
  return true;
}

bool WorldFrameReconstructionNode::lookupTransform(
  const std::string & source_frame, const rclcpp::Time & stamp, tf2::Transform & transform)
{
  try {
    const auto tf_msg = tf_buffer_->lookupTransform(
      world_frame_, source_frame, stamp, rclcpp::Duration::from_seconds(tf_timeout_sec_));
    tf2::fromMsg(tf_msg.transform, transform);
    return true;
  } catch (const tf2::TransformException & e) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
      "TF lookup %s <- %s failed: %s", world_frame_.c_str(), source_frame.c_str(), e.what());
    return false;
  }
}

sensor_msgs::msg::PointCloud2 WorldFrameReconstructionNode::makeCloudMessage(
  const std::vector<SemanticPoint> & points, const std::string & frame, const rclcpp::Time & stamp) const
{
  sensor_msgs::msg::PointCloud2 out;
  out.header.frame_id = frame;
  out.header.stamp = stamp;
  out.height = 1;
  out.width = static_cast<uint32_t>(points.size());
  out.is_bigendian = false;
  out.is_dense = false;
  out.fields.resize(5);

  const std::string names[5] = {"x","y","z","class_id","confidence"};
  const uint8_t datatypes[5] = {sensor_msgs::msg::PointField::FLOAT32,
                                sensor_msgs::msg::PointField::FLOAT32,
                                sensor_msgs::msg::PointField::FLOAT32,
                                sensor_msgs::msg::PointField::INT32,
                                sensor_msgs::msg::PointField::FLOAT32};
  const uint32_t offsets[5] = {0,4,8,12,16};
  for (int i=0;i<5;++i) {
    out.fields[i].name=names[i]; out.fields[i].offset=offsets[i];
    out.fields[i].datatype=datatypes[i]; out.fields[i].count=1;
  }
  out.point_step=20;
  out.row_step=out.point_step*out.width;
  out.data.resize(out.row_step);
  for (std::size_t i=0;i<points.size();++i) {
    std::memcpy(out.data.data()+i*20+0, &points[i].x, 4);
    std::memcpy(out.data.data()+i*20+4, &points[i].y, 4);
    std::memcpy(out.data.data()+i*20+8, &points[i].z, 4);
    std::memcpy(out.data.data()+i*20+12, &points[i].class_id, 4);
    std::memcpy(out.data.data()+i*20+16, &points[i].confidence, 4);
  }
  return out;
}

void WorldFrameReconstructionNode::publishWorldCloud(const rclcpp::Time & stamp)
{
  std::vector<SemanticPoint> snapshot;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot.assign(accumulated_points_.begin(), accumulated_points_.end());
  }
  if (snapshot.empty()) return;
  accumulated_pub_->publish(makeCloudMessage(snapshot, world_frame_, stamp));
}

void WorldFrameReconstructionNode::resetAccumulation()
{
  std::lock_guard<std::mutex> lock(mutex_);
  accumulated_points_.clear();
}

void WorldFrameReconstructionNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  const std::string source_frame = source_frame_override_.empty() ? msg->header.frame_id : source_frame_override_;
  if (source_frame.empty()) {
    RCLCPP_ERROR(get_logger(), "Input cloud has empty frame_id and frames.source_override is empty.");
    return;
  }

  std::vector<SemanticPoint> input_points;
  if (!readSemanticCloud(*msg, input_points)) return;

  tf2::Transform T_world_source;
  if (!lookupTransform(source_frame, rclcpp::Time(msg->header.stamp), T_world_source)) return;

  std::vector<SemanticPoint> world_points;
  world_points.reserve(input_points.size());
  for (const auto & p : input_points) {
    const tf2::Vector3 pw = T_world_source * tf2::Vector3(p.x,p.y,p.z);
    if (!std::isfinite(pw.x()) || !std::isfinite(pw.y()) || !std::isfinite(pw.z())) continue;
    SemanticPoint wp = p;
    wp.x=static_cast<float>(pw.x()); wp.y=static_cast<float>(pw.y()); wp.z=static_cast<float>(pw.z());
    world_points.push_back(wp);
  }

  ++frame_count_;
  world_pub_->publish(makeCloudMessage(world_points, world_frame_, rclcpp::Time(msg->header.stamp)));

  if (accumulate_) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto & p : world_points) accumulated_points_.push_back(p);
    while (accumulated_points_.size() > max_points_) accumulated_points_.pop_front();
  }

  if (publish_accumulated_ && (frame_count_ % publish_every_n_ == 0)) {
    publishWorldCloud(rclcpp::Time(msg->header.stamp));
  }

  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
    "Frame %zu: input=%zu transformed=%zu accumulated=%zu",
    frame_count_, input_points.size(), world_points.size(), accumulated_points_.size());
}

} // namespace world_frame_reconstruction

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<world_frame_reconstruction::WorldFrameReconstructionNode>());
  rclcpp::shutdown();
  return 0;
}
