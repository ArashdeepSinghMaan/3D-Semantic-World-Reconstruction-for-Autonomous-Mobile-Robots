#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
namespace multi_observation_semantic_fusion {
struct VoxelState {
  std::vector<double> class_evidence;
  uint64_t observation_count{0};
  double weight_sum{0.0};
  double mean_z{0.0};
  double m2_z{0.0};
  int64_t last_stamp_ns{0};
};
struct Key { int64_t x,y,z; bool operator==(const Key&o)const{return x==o.x&&y==o.y&&z==o.z;} };
struct KeyHash { size_t operator()(const Key&k)const noexcept { size_t h=std::hash<int64_t>{}(k.x); h^=std::hash<int64_t>{}(k.y)+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2); h^=std::hash<int64_t>{}(k.z)+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2); return h; } };
struct OutputVoxel { Key key; uint32_t class_id{0}; double confidence{0}; double total_evidence{0}; uint64_t observations{0}; double z{0}; };
class SemanticFusionMap {
public:
  SemanticFusionMap(double resolution, uint32_t classes, double prior, double decay_seconds);
  void reset();
  void update(double x,double y,double z,uint32_t cls,double confidence,int64_t stamp_ns);
  std::vector<OutputVoxel> snapshot(double min_confidence) const;
  size_t size() const;
  double resolution() const { return resolution_; }
private:
  Key keyFor(double x,double y,double z) const;
  void decay(VoxelState &v, int64_t now_ns) const;
  double resolution_; uint32_t classes_; double prior_; double decay_seconds_;
  mutable std::mutex mutex_; std::unordered_map<Key,VoxelState,KeyHash> map_;
};
}
