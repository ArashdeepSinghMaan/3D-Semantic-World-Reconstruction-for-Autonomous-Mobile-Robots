#include "persistent_semantic_mapping/voxel_map.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace persistent_semantic_mapping
{

VoxelMap::VoxelMap(
    float resolution,
    float min_confidence,
    uint32_t max_classes)
    : resolution_(resolution),
      min_confidence_(min_confidence),
      max_classes_(max_classes)
{
}


VoxelKey VoxelMap::keyFromPoint(
    float x,
    float y,
    float z) const
{
    return {
        static_cast<int64_t>(
            std::floor(x / resolution_)),

        static_cast<int64_t>(
            std::floor(y / resolution_)),

        static_cast<int64_t>(
            std::floor(z / resolution_))
    };
}


void VoxelMap::update(
    float x,
    float y,
    float z,
    int32_t class_id,
    float confidence,
    double stamp)
{
    if (!std::isfinite(x) ||
        !std::isfinite(y) ||
        !std::isfinite(z) ||
        !std::isfinite(confidence))
    {
        return;
    }

    if (confidence < min_confidence_)
    {
        return;
    }

    const VoxelKey key = keyFromPoint(x, y, z);

    std::lock_guard<std::mutex> lock(mutex_);

    auto& voxel = voxels_[key];

    voxel.confidence_sum += confidence;

    voxel.confidence_max =
        std::max(voxel.confidence_max, confidence);

    voxel.observations++;

    voxel.last_timestamp = stamp;

    voxel.elevation_sum += z;

    voxel.elevation_min =
        std::min(voxel.elevation_min, z);

    voxel.elevation_max =
        std::max(voxel.elevation_max, z);


    if (class_id >= 0)
    {
        const auto existing =
            voxel.class_scores.find(class_id);

        if (existing == voxel.class_scores.end() &&
            voxel.class_scores.size() >= max_classes_)
        {
            auto weakest =
                std::min_element(
                    voxel.class_scores.begin(),
                    voxel.class_scores.end(),
                    [](const auto& a, const auto& b)
                    {
                        return a.second < b.second;
                    });

            if (weakest != voxel.class_scores.end() &&
                confidence > weakest->second)
            {
                voxel.class_scores.erase(weakest);
            }
            else
            {
                return;
            }
        }

        voxel.class_scores[class_id] += confidence;
    }
}


std::vector<VoxelOutput>
VoxelMap::snapshot() const
{
    std::vector<VoxelOutput> output;

    std::lock_guard<std::mutex> lock(mutex_);

    output.reserve(voxels_.size());

    for (const auto& entry : voxels_)
    {
        const auto& key = entry.first;
        const auto& voxel = entry.second;

        int32_t best_class = -1;
        float best_score = 0.0f;


        for (const auto& class_entry : voxel.class_scores)
        {
            if (class_entry.second > best_score)
            {
                best_score = class_entry.second;
                best_class = class_entry.first;
            }
        }


        const float x =
            (static_cast<float>(key.x) + 0.5f)
            * resolution_;

        const float y =
            (static_cast<float>(key.y) + 0.5f)
            * resolution_;

        const float z =
            (static_cast<float>(key.z) + 0.5f)
            * resolution_;


        const float semantic_confidence =
            (voxel.confidence_sum > 0.0f &&
             best_score > 0.0f)
                ? best_score / voxel.confidence_sum
                : 0.0f;


        const uint32_t observations =
            static_cast<uint32_t>(
                std::min<uint64_t>(
                    voxel.observations,
                    static_cast<uint64_t>(
                        std::numeric_limits<uint32_t>::max()
                    )
                )
            );


        output.push_back(
            {
                x,
                y,
                z,
                best_class,
                semantic_confidence,
                observations
            }
        );
    }

    return output;
}


std::size_t VoxelMap::size() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    return voxels_.size();
}


void VoxelMap::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);

    voxels_.clear();
}

}  // namespace persistent_semantic_mapping