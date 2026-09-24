#include "multi_observation_semantic_fusion/semantic_fusion.hpp"
#include <cmath>
#include <algorithm>
namespace multi_observation_semantic_fusion {
SemanticFusionMap::SemanticFusionMap(double r,uint32_t c,double p,double d):resolution_(r),classes_(c),prior_(p),decay_seconds_(d){}
SemanticFusionMap::Key SemanticFusionMap::keyFor(double x,double y,double z) const { return {static_cast<int64_t>(std::floor(x/resolution_)),static_cast<int64_t>(std::floor(y/resolution_)),static_cast<int64_t>(std::floor(z/resolution_))}; }
void SemanticFusionMap::reset(){std::scoped_lock l(mutex_);map_.clear();}
void SemanticFusionMap::decay(VoxelState &v,int64_t now) const { if(decay_seconds_<=0||v.last_stamp_ns<=0||now<=v.last_stamp_ns)return; double f=std::exp(-(now-v.last_stamp_ns)*1e-9/decay_seconds_); for(auto &e:v.class_evidence)e*=f; v.weight_sum*=f; v.last_stamp_ns=now; }
void SemanticFusionMap::update(double x,double y,double z,uint32_t cls,double confidence,int64_t stamp){ if(cls>=classes_||!std::isfinite(confidence)||confidence<=0)return; confidence=std::clamp(confidence,0.0,1.0); std::scoped_lock l(mutex_); auto k=keyFor(x,y,z); auto &v=map_[k]; if(v.class_evidence.empty())v.class_evidence.assign(classes_,prior_); decay(v,stamp); double w=confidence; v.class_evidence[cls]+=w; v.weight_sum+=w; v.observation_count++; double delta=z-v.mean_z; v.mean_z+=delta/static_cast<double>(v.observation_count); v.m2_z+=delta*(z-v.mean_z); v.last_stamp_ns=stamp; }
std::vector<OutputVoxel> SemanticFusionMap::snapshot(double minc) const { std::scoped_lock l(mutex_); std::vector<OutputVoxel> out; out.reserve(map_.size()); for(auto &[k,v]:map_){double s=0;uint32_t best=0;double bv=-1;for(uint32_t c=0;c<classes_;++c){s+=v.class_evidence[c];if(v.class_evidence[c]>bv){bv=v.class_evidence[c];best=c;}} if(s<=0)continue;double conf=bv/s;if(conf<minc)continue;out.push_back({k,best,conf,s,v.observation_count,v.mean_z});}return out;}
size_t SemanticFusionMap::size() const {std::scoped_lock l(mutex_);return map_.size();}
}
