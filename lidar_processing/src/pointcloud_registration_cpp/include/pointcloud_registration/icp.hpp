#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>

namespace pointcloud_registration
{

struct ICPConfig
{
  std::string method{"point_to_point"};

  double voxel_size{0.10};
  double max_correspondence_distance{0.30};
  int max_iterations{50};
  double transformation_epsilon{1e-6};
  double euclidean_fitness_epsilon{1e-6};

  bool multiscale_enabled{true};
  std::vector<double> voxel_sizes{0.20, 0.10, 0.05};
  std::vector<double> correspondence_factors{2.5, 2.0, 1.5};
  std::vector<int> iterations{40, 30, 20};

  int normal_k{30};
  double normal_radius_factor{2.5};
};

struct ICPResult
{
  Eigen::Matrix4f transformation{Eigen::Matrix4f::Identity()};
  double fitness_score{0.0};
  double inlier_rmse{0.0};
  std::size_t source_points{0};
  std::size_t target_points{0};
  double processing_time_ms{0.0};
  bool converged{false};
};

class ICP
{
public:
  ICPResult run(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & source,
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & target,
    const Eigen::Matrix4f & initial_guess,
    const ICPConfig & config) const;

private:
  pcl::PointCloud<pcl::PointXYZ>::Ptr voxelDownsample(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & cloud,
    double voxel_size) const;

  ICPResult runPointToPoint(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & source,
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & target,
    const Eigen::Matrix4f & initial_guess,
    double max_correspondence_distance,
    int max_iterations,
    double transformation_epsilon,
    double euclidean_fitness_epsilon) const;

  ICPResult runPointToPlane(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & source,
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & target,
    const Eigen::Matrix4f & initial_guess,
    double max_correspondence_distance,
    int max_iterations,
    double transformation_epsilon,
    double euclidean_fitness_epsilon,
    int normal_k,
    double normal_radius_factor,
    double voxel_size) const;
};

}  // namespace pointcloud_registration
