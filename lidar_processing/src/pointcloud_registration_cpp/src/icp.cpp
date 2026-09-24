#include "pointcloud_registration/icp.hpp"

#include <chrono>
#include <stdexcept>

#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>

namespace pointcloud_registration
{

pcl::PointCloud<pcl::PointXYZ>::Ptr ICP::voxelDownsample(
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & cloud,
  double voxel_size) const
{
  if (voxel_size <= 0.0) {
    return std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*cloud);
  }

  auto output = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::VoxelGrid<pcl::PointXYZ> voxel;
  voxel.setInputCloud(cloud);
  voxel.setLeafSize(
    static_cast<float>(voxel_size),
    static_cast<float>(voxel_size),
    static_cast<float>(voxel_size));
  voxel.filter(*output);
  return output;
}

ICPResult ICP::runPointToPoint(
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & source,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & target,
  const Eigen::Matrix4f & initial_guess,
  double max_correspondence_distance,
  int max_iterations,
  double transformation_epsilon,
  double euclidean_fitness_epsilon) const
{
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(source);
  icp.setInputTarget(target);
  icp.setMaxCorrespondenceDistance(static_cast<float>(max_correspondence_distance));
  icp.setMaximumIterations(max_iterations);
  icp.setTransformationEpsilon(transformation_epsilon);
  icp.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon);

  pcl::PointCloud<pcl::PointXYZ> aligned;
  icp.align(aligned, initial_guess);

  ICPResult result;
  result.transformation = icp.getFinalTransformation();
  result.fitness_score = icp.getFitnessScore();
  result.source_points = source->size();
  result.target_points = target->size();
  result.converged = icp.hasConverged();

  if (icp.hasConverged() && result.fitness_score >= 0.0) {
    result.inlier_rmse = std::sqrt(result.fitness_score);
  }

  return result;
}

ICPResult ICP::runPointToPlane(
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & source,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & target,
  const Eigen::Matrix4f & initial_guess,
  double max_correspondence_distance,
  int max_iterations,
  double transformation_epsilon,
  double euclidean_fitness_epsilon,
  int normal_k,
  double normal_radius_factor,
  double voxel_size) const
{
  using PointNormal = pcl::PointNormal;

  auto source_normals = std::make_shared<pcl::PointCloud<PointNormal>>();
  auto target_normals = std::make_shared<pcl::PointCloud<PointNormal>>();

  pcl::NormalEstimation<pcl::PointXYZ, PointNormal> normal_estimator;
  auto tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();

  normal_estimator.setSearchMethod(tree);

  normal_estimator.setInputCloud(source);
  if (normal_k > 0) {
    normal_estimator.setKSearch(normal_k);
  } else {
    normal_estimator.setRadiusSearch(
      std::max(0.001, voxel_size * normal_radius_factor));
  }
  normal_estimator.compute(*source_normals);

  normal_estimator.setInputCloud(target);
  if (normal_k > 0) {
    normal_estimator.setKSearch(normal_k);
  } else {
    normal_estimator.setRadiusSearch(
      std::max(0.001, voxel_size * normal_radius_factor));
  }
  normal_estimator.compute(*target_normals);

  auto source_with_normals = std::make_shared<pcl::PointCloud<PointNormal>>();
  auto target_with_normals = std::make_shared<pcl::PointCloud<PointNormal>>();

  pcl::concatenateFields(*source, *source_normals, *source_with_normals);
  pcl::concatenateFields(*target, *target_normals, *target_with_normals);

  pcl::IterativeClosestPointWithNormals<PointNormal, PointNormal> icp;
  icp.setInputSource(source_with_normals);
  icp.setInputTarget(target_with_normals);
  icp.setMaxCorrespondenceDistance(
    static_cast<float>(max_correspondence_distance));
  icp.setMaximumIterations(max_iterations);
  icp.setTransformationEpsilon(transformation_epsilon);
  icp.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon);

  pcl::PointCloud<PointNormal> aligned;
  icp.align(aligned, initial_guess);

  ICPResult result;
  result.transformation = icp.getFinalTransformation();
  result.fitness_score = icp.getFitnessScore();
  result.source_points = source->size();
  result.target_points = target->size();
  result.converged = icp.hasConverged();

  if (icp.hasConverged() && result.fitness_score >= 0.0) {
    result.inlier_rmse = std::sqrt(result.fitness_score);
  }

  return result;
}

ICPResult ICP::run(
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & source,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & target,
  const Eigen::Matrix4f & initial_guess,
  const ICPConfig & config) const
{
  if (!source || !target || source->empty() || target->empty()) {
    throw std::runtime_error("ICP received an empty source or target cloud.");
  }

  const auto start = std::chrono::steady_clock::now();

  Eigen::Matrix4f transform = initial_guess;
  ICPResult result;

  if (config.multiscale_enabled) {
    if (config.voxel_sizes.empty() ||
        config.voxel_sizes.size() != config.correspondence_factors.size() ||
        config.voxel_sizes.size() != config.iterations.size()) {
      throw std::runtime_error(
        "Multi-scale voxel_sizes, correspondence_factors and iterations "
        "must have identical non-zero lengths.");
    }

    for (std::size_t i = 0; i < config.voxel_sizes.size(); ++i) {
      auto source_ds = voxelDownsample(source, config.voxel_sizes[i]);
      auto target_ds = voxelDownsample(target, config.voxel_sizes[i]);

      if (source_ds->size() < 10 || target_ds->size() < 10) {
        throw std::runtime_error(
          "Too few points after voxel downsampling at scale " +
          std::to_string(config.voxel_sizes[i]));
      }

      const double correspondence =
        config.voxel_sizes[i] * config.correspondence_factors[i];

      if (config.method == "point_to_point") {
        result = runPointToPoint(
          source_ds, target_ds, transform, correspondence,
          config.iterations[i], config.transformation_epsilon,
          config.euclidean_fitness_epsilon);
      } else if (config.method == "point_to_plane") {
        result = runPointToPlane(
          source_ds, target_ds, transform, correspondence,
          config.iterations[i], config.transformation_epsilon,
          config.euclidean_fitness_epsilon, config.normal_k,
          config.normal_radius_factor, config.voxel_sizes[i]);
      } else {
        throw std::runtime_error(
          "Unsupported ICP method: " + config.method);
      }

      transform = result.transformation;
    }
  } else {
    auto source_ds = voxelDownsample(source, config.voxel_size);
    auto target_ds = voxelDownsample(target, config.voxel_size);

    if (source_ds->size() < 10 || target_ds->size() < 10) {
      throw std::runtime_error("Too few points after voxel downsampling.");
    }

    if (config.method == "point_to_point") {
      result = runPointToPoint(
        source_ds, target_ds, transform,
        config.max_correspondence_distance,
        config.max_iterations,
        config.transformation_epsilon,
        config.euclidean_fitness_epsilon);
    } else if (config.method == "point_to_plane") {
      result = runPointToPlane(
        source_ds, target_ds, transform,
        config.max_correspondence_distance,
        config.max_iterations,
        config.transformation_epsilon,
        config.euclidean_fitness_epsilon,
        config.normal_k,
        config.normal_radius_factor,
        config.voxel_size);
    } else {
      throw std::runtime_error(
        "Unsupported ICP method: " + config.method);
    }
  }

  const auto end = std::chrono::steady_clock::now();
  result.processing_time_ms =
    std::chrono::duration<double, std::milli>(end - start).count();

  return result;
}

}  // namespace pointcloud_registration
