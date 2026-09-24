#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <std_msgs/msg/u_int64.hpp>

#include <cstdint>
#include <memory>
#include <string>

#include "multi_observation_semantic_fusion/semantic_fusion.hpp"

using namespace std::chrono_literals;

using multi_observation_semantic_fusion::SemanticFusionMap;


class Node : public rclcpp::Node
{
public:

    Node()
    : rclcpp::Node("multi_observation_semantic_fusion"),
      map_(
          declare_parameter<double>(
              "voxel.resolution",
              0.10),

          static_cast<uint32_t>(
              declare_parameter<int>(
                  "semantic.num_classes",
                  20)),

          declare_parameter<double>(
              "semantic.prior",
              0.001),

          declare_parameter<double>(
              "fusion.decay_seconds",
              0.0))

    {
        in_ = declare_parameter<std::string>(
            "topics.input",
            "/semantic_map/points");

        out_ = declare_parameter<std::string>(
            "topics.output",
            "/semantic_fusion/map");

        minc_ = declare_parameter<double>(
            "fusion.publish_min_confidence",
            0.0);

        max_points_ = static_cast<uint32_t>(
            declare_parameter<int>(
                "output.max_points",
                500000));

        frame_ = declare_parameter<std::string>(
            "output.frame",
            "map");


        sub_ =
            create_subscription<sensor_msgs::msg::PointCloud2>(
                in_,
                rclcpp::SensorDataQoS(),
                std::bind(
                    &Node::cb,
                    this,
                    std::placeholders::_1));


        pub_ =
            create_publisher<sensor_msgs::msg::PointCloud2>(
                out_,
                10);


        count_pub_ =
            create_publisher<std_msgs::msg::UInt64>(
                "/semantic_fusion/voxel_count",
                10);


        timer_ =
            create_wall_timer(
                1s,
                std::bind(
                    &Node::publish,
                    this));


        RCLCPP_INFO(
            get_logger(),
            "Phase 9 multi-observation semantic fusion ready. "
            "Input: %s",
            in_.c_str());
    }


private:

    void cb(
        const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        try
        {
            sensor_msgs::PointCloud2ConstIterator<float>
                x(*msg, "x"),
                y(*msg, "y"),
                z(*msg, "z"),
                conf(*msg, "confidence");

            sensor_msgs::PointCloud2ConstIterator<uint32_t>
                cls(*msg, "class_id");


            const std::size_t point_count =
                static_cast<std::size_t>(msg->width) *
                static_cast<std::size_t>(msg->height);


            const int64_t stamp_ns =
                rclcpp::Time(
                    msg->header.stamp).nanoseconds();


            for (std::size_t i = 0;
                 i < point_count;
                 ++i,
                 ++x,
                 ++y,
                 ++z,
                 ++conf,
                 ++cls)
            {
                map_.update(
                    *x,
                    *y,
                    *z,
                    *cls,
                    *conf,
                    stamp_ns);
            }
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Input cloud field error: %s",
                e.what());
        }
    }


    void publish()
    {
        auto vox =
            map_.snapshot(minc_);


        if (max_points_ > 0 &&
            vox.size() > max_points_)
        {
            vox.resize(max_points_);
        }


        sensor_msgs::msg::PointCloud2 msg;

        msg.header.stamp = now();
        msg.header.frame_id = frame_;

        msg.height = 1;

        msg.width =
            static_cast<uint32_t>(vox.size());

        msg.is_dense = true;


        sensor_msgs::PointCloud2Modifier modifier(msg);

        modifier.setPointCloud2Fields(
            5,

            "x",
            1,
            sensor_msgs::msg::PointField::FLOAT32,

            "y",
            1,
            sensor_msgs::msg::PointField::FLOAT32,

            "z",
            1,
            sensor_msgs::msg::PointField::FLOAT32,

            "class_id",
            1,
            sensor_msgs::msg::PointField::UINT32,

            "confidence",
            1,
            sensor_msgs::msg::PointField::FLOAT32);


        modifier.resize(vox.size());


        sensor_msgs::PointCloud2Iterator<float>
            x(msg, "x"),
            y(msg, "y"),
            z(msg, "z"),
            c(msg, "confidence");

        sensor_msgs::PointCloud2Iterator<uint32_t>
            cls(msg, "class_id");


        const double resolution =
            map_.resolution();


        for (const auto& v : vox)
        {
            *x =
                (static_cast<double>(v.key.x) + 0.5)
                * resolution;

            *y =
                (static_cast<double>(v.key.y) + 0.5)
                * resolution;

            *z = v.z;

            *cls = v.class_id;

            *c = v.confidence;


            ++x;
            ++y;
            ++z;
            ++cls;
            ++c;
        }


        pub_->publish(msg);


        std_msgs::msg::UInt64 n;

        n.data =
            static_cast<uint64_t>(map_.size());

        count_pub_->publish(n);
    }


    rclcpp::Subscription<
        sensor_msgs::msg::PointCloud2>::SharedPtr sub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

    rclcpp::Publisher<
        std_msgs::msg::UInt64>::SharedPtr count_pub_;

    rclcpp::TimerBase::SharedPtr timer_;


    SemanticFusionMap map_;


    std::string in_;
    std::string out_;
    std::string frame_;

    double minc_{0.0};

    uint32_t max_points_{0};
};


int main(
    int argc,
    char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::spin(
        std::make_shared<Node>());

    rclcpp::shutdown();

    return 0;
}