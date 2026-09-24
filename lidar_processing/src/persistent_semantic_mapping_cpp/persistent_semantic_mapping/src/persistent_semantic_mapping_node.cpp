#include "persistent_semantic_mapping/voxel_map.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/uint64.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <algorithm>
#include <chrono>
#include <memory>
#include <string>

using namespace std::chrono_literals;
using persistent_semantic_mapping::VoxelMap;
using persistent_semantic_mapping::VoxelOutput;

class PersistentSemanticMappingNode : public rclcpp::Node {
public:
  PersistentSemanticMappingNode() : Node("persistent_semantic_mapping") {
    input_topic_ = declare_parameter<std::string>("topics.input", "/world_reconstruction/points");
    map_topic_ = declare_parameter<std::string>("topics.map", "/semantic_map/points");
    marker_topic_ = declare_parameter<std::string>("topics.markers", "/semantic_map/markers");
    stats_topic_ = declare_parameter<std::string>("topics.stats", "/semantic_map/voxel_count");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    resolution_ = declare_parameter<double>("voxel.resolution", 0.10);
    min_confidence_ = declare_parameter<double>("voxel.min_confidence", 0.0);
    max_classes_ = declare_parameter<int>("voxel.max_classes_per_voxel", 16);
    publish_rate_ = declare_parameter<double>("publish.rate_hz", 2.0);
    max_publish_voxels_ = declare_parameter<int>("publish.max_voxels", 0);
    class_unknown_ = declare_parameter<int>("visualization.unknown_class", -1);
    marker_size_ = declare_parameter<double>("visualization.marker_size", 0.08);
    publish_markers_ = declare_parameter<bool>("visualization.publish_markers", false);

    if (resolution_ <= 0.0) throw std::runtime_error("voxel.resolution must be > 0");
    if (max_classes_ < 1) throw std::runtime_error("voxel.max_classes_per_voxel must be >= 1");
    if (publish_rate_ <= 0.0) throw std::runtime_error("publish.rate_hz must be > 0");

    map_ = std::make_unique<VoxelMap>(static_cast<float>(resolution_), static_cast<float>(min_confidence_), static_cast<uint32_t>(max_classes_));
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PersistentSemanticMappingNode::cloudCallback, this, std::placeholders::_1));
    map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(map_topic_, 10);
    stats_pub_ = create_publisher<std_msgs::msg::UInt64>(stats_topic_, 10);
    if (publish_markers_) marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, 10);
    timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / publish_rate_)),
      std::bind(&PersistentSemanticMappingNode::publishMap, this));
    RCLCPP_INFO(get_logger(), "Persistent semantic voxel map: %.3f m resolution, input=%s, frame=%s", resolution_, input_topic_.c_str(), map_frame_.c_str());
  }

private:
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (msg->width == 0 || msg->height == 0) return;
    try {
      sensor_msgs::PointCloud2ConstIterator<float> x(*msg, "x"), y(*msg, "y"), z(*msg, "z");
      sensor_msgs::PointCloud2ConstIterator<int32_t> cls(*msg, "class_id");
      sensor_msgs::PointCloud2ConstIterator<float> conf(*msg, "confidence");
      const double stamp = rclcpp::Time(msg->header.stamp).seconds();
      for (size_t i = 0; i < static_cast<size_t>(msg->width) * msg->height; ++i, ++x, ++y, ++z, ++cls, ++conf)
        map_->update(*x, *y, *z, *cls, *conf, stamp);
    } catch (const std::exception &e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Expected fields x,y,z,class_id,confidence: %s", e.what());
    }
  }

  void publishMap() {
    auto voxels = map_->snapshot();
    if (max_publish_voxels_ > 0 && static_cast<size_t>(max_publish_voxels_) < voxels.size()) voxels.resize(max_publish_voxels_);
    sensor_msgs::msg::PointCloud2 out;
    out.header.stamp = now(); out.header.frame_id = map_frame_;
    out.height = 1; out.width = static_cast<uint32_t>(voxels.size()); out.is_dense = true;
    sensor_msgs::PointCloud2Modifier mod(out);
    mod.setPointCloud2Fields(5, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                             "z", 1, sensor_msgs::msg::PointField::FLOAT32, "class_id", 1, sensor_msgs::msg::PointField::INT32,
                             "confidence", 1, sensor_msgs::msg::PointField::FLOAT32);
    mod.resize(voxels.size());
    sensor_msgs::PointCloud2Iterator<float> x(out, "x"), y(out, "y"), z(out, "z"), conf(out, "confidence");
    sensor_msgs::PointCloud2Iterator<int32_t> cls(out, "class_id");
    for (const auto &v : voxels) { *x++=v.x; *y++=v.y; *z++=v.z; *cls++=v.class_id; *conf++=v.confidence; }
    map_pub_->publish(out);
    std_msgs::msg::UInt64 stats; stats.data = map_->size(); stats_pub_->publish(stats);
    if (publish_markers_) publishMarkers(voxels);
  }

  void publishMarkers(const std::vector<VoxelOutput> &voxels) {
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker m;
    m.header.stamp = now(); m.header.frame_id = map_frame_; m.ns = "semantic_voxels"; m.id = 0;
    m.type = visualization_msgs::msg::Marker::CUBE_LIST; m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = marker_size_; m.scale.y = marker_size_; m.scale.z = marker_size_; m.color.a = 0.7f;
    for (const auto &v : voxels) { geometry_msgs::msg::Point p; p.x=v.x; p.y=v.y; p.z=v.z; m.points.push_back(p); }
    arr.markers.push_back(m); marker_pub_->publish(arr);
  }

  std::string input_topic_, map_topic_, marker_topic_, stats_topic_, map_frame_;
  double resolution_, min_confidence_, publish_rate_, marker_size_;
  int max_classes_, max_publish_voxels_, class_unknown_;
  bool publish_markers_;
  std::unique_ptr<VoxelMap> map_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr stats_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) { rclcpp::init(argc, argv); rclcpp::spin(std::make_shared<PersistentSemanticMappingNode>()); rclcpp::shutdown(); return 0; }
