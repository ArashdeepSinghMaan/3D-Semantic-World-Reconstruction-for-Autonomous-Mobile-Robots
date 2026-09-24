#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include <limits>

namespace persistent_semantic_mapping {

struct VoxelKey {
  int64_t x{0}, y{0}, z{0};
  bool operator==(const VoxelKey &o) const { return x == o.x && y == o.y && z == o.z; }
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey &k) const noexcept;
};

struct Voxel {
  float confidence_sum{0.0f};
  float confidence_max{0.0f};
  std::unordered_map<int32_t, float> class_scores;
  uint64_t observations{0};
  double last_timestamp{0.0};
  float elevation_sum{0.0f};
  float elevation_min{std::numeric_limits<float>::infinity()};
  float elevation_max{-std::numeric_limits<float>::infinity()};
};

struct VoxelOutput {
  float x, y, z;
  int32_t class_id;
  float confidence;
  uint32_t observations;
};

class VoxelMap {
public:
  explicit VoxelMap(float resolution, float min_confidence, uint32_t max_classes);

  void update(float x, float y, float z, int32_t class_id, float confidence, double stamp);
  std::vector<VoxelOutput> snapshot() const;
  std::size_t size() const;
  void clear();

  VoxelKey keyFromPoint(float x, float y, float z) const;

private:
  float resolution_;
  float min_confidence_;
  uint32_t max_classes_;
  mutable std::mutex mutex_;
  std::unordered_map<VoxelKey, Voxel> voxels_;
};

}  // namespace persistent_semantic_mapping
