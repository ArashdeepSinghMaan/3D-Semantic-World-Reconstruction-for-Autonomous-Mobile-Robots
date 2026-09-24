#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/uint64.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "multi_observation_semantic_fusion/semantic_fusion.hpp"
using namespace std::chrono_literals;
class Node : public rclcpp::Node {
public:
 Node():Node("multi_observation_semantic_fusion"), map_(declare_parameter("voxel.resolution",0.10),declare_parameter("semantic.num_classes",20u),declare_parameter("semantic.prior",0.001),declare_parameter("fusion.decay_seconds",0.0)) {
  in_=declare_parameter("topics.input","/semantic_map/points"); out_=declare_parameter("topics.output","/semantic_fusion/map"); minc_=declare_parameter("fusion.publish_min_confidence",0.0); max_points_=declare_parameter("output.max_points",500000u); frame_=declare_parameter("output.frame","map");
  sub_=create_subscription<sensor_msgs::msg::PointCloud2>(in_,rclcpp::SensorDataQoS(),std::bind(&Node::cb,this,std::placeholders::_1)); pub_=create_publisher<sensor_msgs::msg::PointCloud2>(out_,10); count_pub_=create_publisher<std_msgs::msg::UInt64>("/semantic_fusion/voxel_count",10); timer_=create_wall_timer(1s,std::bind(&Node::publish,this));
  RCLCPP_INFO(get_logger(),"Phase 9 multi-observation semantic fusion ready. Input: %s",in_.c_str());
 }
private:
 void cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg){try{sensor_msgs::PointCloud2ConstIterator<float> x(*msg,"x"),y(*msg,"y"),z(*msg,"z"),conf(*msg,"confidence"); sensor_msgs::PointCloud2ConstIterator<uint32_t> cls(*msg,"class_id"); for(;x!=x.end();++x,++y,++z,++conf,++cls)map_.update(*x,*y,*z,*cls,*conf,rclcpp::Time(msg->header.stamp).nanoseconds());}catch(const std::exception&e){RCLCPP_ERROR_THROTTLE(get_logger(),*get_clock(),2000,"Input cloud field error: %s",e.what());}}
 void publish(){auto vox=map_.snapshot(minc_); if(vox.size()>max_points_)vox.resize(max_points_); sensor_msgs::msg::PointCloud2 msg; msg.header.stamp=now(); msg.header.frame_id=frame_; msg.height=1;msg.width=vox.size(); msg.is_dense=true; sensor_msgs::PointCloud2Modifier mod(msg); mod.setPointCloud2Fields(5,"x",1,sensor_msgs::msg::PointField::FLOAT32,"y",1,sensor_msgs::msg::PointField::FLOAT32,"z",1,sensor_msgs::msg::PointField::FLOAT32,"class_id",1,sensor_msgs::msg::PointField::UINT32,"confidence",1,sensor_msgs::msg::PointField::FLOAT32); mod.resize(vox.size()); sensor_msgs::PointCloud2Iterator<float> x(msg,"x"),y(msg,"y"),z(msg,"z"),c(msg,"confidence"); sensor_msgs::PointCloud2Iterator<uint32_t> cls(msg,"class_id"); double r=map_.resolution(); size_t i=0; for(auto &v:vox){*x++=(v.key.x+0.5)*r;*y++=(v.key.y+0.5)*r;*z++=v.z;*cls++=v.class_id;*c++=v.confidence;i++;} pub_->publish(msg); std_msgs::msg::UInt64 n;n.data=map_.size();count_pub_->publish(n); }
 rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_; rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_; rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr count_pub_; rclcpp::TimerBase::SharedPtr timer_; SemanticFusionMap map_; std::string in_,out_,frame_; double minc_; uint32_t max_points_;
};
int main(int argc,char**argv){rclcpp::init(argc,argv);rclcpp::spin(std::make_shared<Node>());rclcpp::shutdown();}
