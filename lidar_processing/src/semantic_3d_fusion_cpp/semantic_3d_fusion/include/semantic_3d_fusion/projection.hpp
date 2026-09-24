#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace semantic_3d_fusion
{

struct CameraModel
{
  int width{0};
  int height{0};
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
};

struct Extrinsic
{
  Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d translation{Eigen::Vector3d::Zero()};
};

struct ProjectionResult
{
  bool valid{false};
  double u{0.0};
  double v{0.0};
  double depth{0.0};
};

class Projector
{
public:
  void setCamera(const CameraModel &camera);
  void setExtrinsic(const Extrinsic &extrinsic);

  ProjectionResult project(const Eigen::Vector3d &point_lidar) const;
  Eigen::Vector3d transformToCamera(const Eigen::Vector3d &point_lidar) const;

private:
  CameraModel camera_;
  Extrinsic extrinsic_;
};

bool insideImage(double u, double v, int width, int height);

}  // namespace semantic_3d_fusion
