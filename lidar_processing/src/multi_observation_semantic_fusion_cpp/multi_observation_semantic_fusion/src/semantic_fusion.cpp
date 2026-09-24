#include "multi_observation_semantic_fusion/semantic_fusion.hpp"

#include <algorithm>
#include <cmath>

namespace multi_observation_semantic_fusion
{

SemanticFusionMap::SemanticFusionMap(
    double r,
    uint32_t c,
    double p,
    double d)
    : resolution_(r),
      classes_(c),
      prior_(p),
      decay_seconds_(d)
{
}


Key SemanticFusionMap::keyFor(
    double x,
    double y,
    double z) const
{
    return {
        static_cast<int64_t>(std::floor(x / resolution_)),
        static_cast<int64_t>(std::floor(y / resolution_)),
        static_cast<int64_t>(std::floor(z / resolution_))
    };
}


void SemanticFusionMap::reset()
{
    std::scoped_lock lock(mutex_);
    map_.clear();
}


void SemanticFusionMap::decay(
    VoxelState& v,
    int64_t now) const
{
    if (decay_seconds_ <= 0.0 ||
        v.last_stamp_ns <= 0 ||
        now <= v.last_stamp_ns)
    {
        return;
    }

    const double dt =
        static_cast<double>(now - v.last_stamp_ns) * 1e-9;

    const double factor =
        std::exp(-dt / decay_seconds_);

    for (auto& evidence : v.class_evidence)
    {
        evidence *= factor;
    }

    v.weight_sum *= factor;

    v.last_stamp_ns = now;
}


void SemanticFusionMap::update(
    double x,
    double y,
    double z,
    uint32_t cls,
    double confidence,
    int64_t stamp)
{
    if (cls >= classes_ ||
        !std::isfinite(confidence) ||
        confidence <= 0.0)
    {
        return;
    }

    confidence = std::clamp(confidence, 0.0, 1.0);

    std::scoped_lock lock(mutex_);

    const Key key = keyFor(x, y, z);

    auto& voxel = map_[key];

    // Initialize class evidence for a new voxel.
    if (voxel.class_evidence.empty())
    {
        voxel.class_evidence.assign(
            classes_,
            prior_);
    }

    // Apply temporal decay before adding the new observation.
    decay(voxel, stamp);

    const double weight = confidence;

    voxel.class_evidence[cls] += weight;
    voxel.weight_sum += weight;

    voxel.observation_count++;

    // Incremental mean and variance update for Z.
    const double delta = z - voxel.mean_z;

    voxel.mean_z +=
        delta /
        static_cast<double>(voxel.observation_count);

    voxel.m2_z +=
        delta * (z - voxel.mean_z);

    voxel.last_stamp_ns = stamp;
}


std::vector<OutputVoxel>
SemanticFusionMap::snapshot(
    double minc) const
{
    std::scoped_lock lock(mutex_);

    std::vector<OutputVoxel> output;

    output.reserve(map_.size());

    for (const auto& [key, voxel] : map_)
    {
        double total_evidence = 0.0;

        uint32_t best_class = 0;
        double best_evidence = -1.0;

        for (uint32_t c = 0; c < classes_; ++c)
        {
            const double evidence =
                voxel.class_evidence[c];

            total_evidence += evidence;

            if (evidence > best_evidence)
            {
                best_evidence = evidence;
                best_class = c;
            }
        }

        if (total_evidence <= 0.0)
        {
            continue;
        }

        const double confidence =
            best_evidence / total_evidence;

        if (confidence < minc)
        {
            continue;
        }

        OutputVoxel out;

        out.key = key;
        out.class_id = best_class;
        out.confidence = confidence;
        out.total_evidence = total_evidence;
        out.observations = voxel.observation_count;
        out.z = voxel.mean_z;

        output.push_back(out);
    }

    return output;
}


std::size_t SemanticFusionMap::size() const
{
    std::scoped_lock lock(mutex_);

    return map_.size();
}

}  // namespace multi_observation_semantic_fusion