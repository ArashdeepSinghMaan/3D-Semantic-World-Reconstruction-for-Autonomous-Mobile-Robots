#include "persistent_semantic_mapping/voxel_map.hpp"
#include <cmath>
#include <algorithm>

namespace persistent_semantic_mapping {

std::size_t VoxelKeyHash::operator()(const VoxelKey &k) const noexcept {
  const uint64_t hx = std::hash<int64_t>{}(k.x);
  const uint64_t hy = std::hash<int64_t>{}(k.y);
  const uint64_t hz = std::hash<int64_t>{}(k.z);
  return static_cast<std::size_t>(hx ^ (hy + 0x9e3779b97f4a7c15ULL + (hx << 6) + (hx >> 2)) ^
                                  (hz + 0x9e3779b97f4a7c15ULL + (hy << 6) + (hy >> 2)));
}

VoxelMap::VoxelMap(float resolution, float min_confidence, uint32_t max_classes)
    : resolution_(resolution), min_confidence_(min_confidence), max_classes_(max_classes) {}

VoxelKey VoxelMap::keyFromPoint(float x, float y, float z) const {
  return {static_cast<int64_t>(std::floor(x / resolution_)),
          static_cast<int64_t>(std::floor(y / resolution_)),
          static_cast<int64_t>(std::floor(z / resolution_))};
}

void VoxelMap::update(float x, float y, float z, int32_t class_id, float confidence, double stamp) {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(confidence)) return;
  if (confidence < min_confidence_) return;
  const auto key = keyFromPoint(x, y, z);
  std::lock_guard<std::mutex> lock(mutex_);
  auto &v = voxels_[key];
  v.confidence_sum += confidence;
  v.confidence_max = std::max(v.confidence_max, confidence);
  v.observations++;
  v.last_timestamp = stamp;
  v.elevation_sum += z;
  v.elevation_min = std::min(v.elevation_min, z);
  v.elevation_max = std::max(v.elevation_max, z);
  if (class_id >= 0) {
    if (v.class_scores.find(class_id) == v.class_scores.end() && v.class_scores.size() >= max_classes_) {
      auto weakest = std::min_element(v.class_scores.begin(), v.class_scores.end(),
          [](const auto &a, const auto &b) { return a.second < b.second; });
      if (weakest != v.class_scores.end() && confidence > weakest->second) v.class_scores.erase(weakest);
      else return;
    }
    v.class_scores[class_id] += confidence;
  }
}

std::vector<VoxelOutput> VoxelMap::snapshot() const {
  std::vector<VoxelOutput> out;
  std::lock_guard<std::mutex> lock(mutex_);
  out.reserve(voxels_.size());
  for (const auto &entry : voxels_) {
    const auto &k = entry.first;
    const auto &v = entry.second;
    int32_t best_class = -1;
    float best_score = 0.0f;
    for (const auto &c : v.class_scores) if (c.second > best_score) { best_score = c.second; best_class = c.first; }
    const float x = (static_cast<float>(k.x) + 0.5f) * resolution_;
    const float y = (static_cast<float>(k.y) + 0.5f) * resolution_;
    const float z = (static_cast<float>(k.z) + 0.5f) * resolution_;
    const float semantic_conf = (v.confidence_sum > 0.0f && best_score > 0.0f) ? best_score / v.confidence_sum : 0.0f;
    out.push_back({x, y, z, best_class, semantic_conf, static_cast<uint32_t>(std::min<uint64_t>(v.observations, UINT32_MAX))});
  }
  return out;
}

std::size_t VoxelMap::size() const { std::lock_guard<std::mutex> lock(mutex_); return voxels_.size(); }
void VoxelMap::clear() { std::lock_guard<std::mutex> lock(mutex_); voxels_.clear(); }

}  // namespace persistent_semantic_mapping
