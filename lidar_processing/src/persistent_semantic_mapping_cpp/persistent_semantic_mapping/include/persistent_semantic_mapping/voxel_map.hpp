#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <limits>

namespace persistent_semantic_mapping
{

struct VoxelKey
{
    int64_t x;
    int64_t y;
    int64_t z;

    bool operator==(const VoxelKey& other) const
    {
        return x == other.x &&
               y == other.y &&
               z == other.z;
    }
};


struct VoxelKeyHash
{
    std::size_t operator()(const VoxelKey& key) const noexcept
    {
        std::size_t h1 = std::hash<int64_t>{}(key.x);
        std::size_t h2 = std::hash<int64_t>{}(key.y);
        std::size_t h3 = std::hash<int64_t>{}(key.z);

        std::size_t seed = h1;

        seed ^= h2
                + 0x9e3779b97f4a7c15ULL
                + (seed << 6)
                + (seed >> 2);

        seed ^= h3
                + 0x9e3779b97f4a7c15ULL
                + (seed << 6)
                + (seed >> 2);

        return seed;
    }
};


struct Voxel
{
    float confidence_sum{0.0f};
    float confidence_max{0.0f};

    std::unordered_map<int32_t, float> class_scores;

    uint64_t observations{0};
    double last_timestamp{0.0};

    float elevation_sum{0.0f};

    float elevation_min{
        std::numeric_limits<float>::infinity()
    };

    float elevation_max{
        -std::numeric_limits<float>::infinity()
    };
};


struct VoxelOutput
{
    float x;
    float y;
    float z;

    int32_t class_id;

    float confidence;

    uint32_t observations;
};


class VoxelMap
{
public:

    explicit VoxelMap(
        float resolution,
        float min_confidence,
        uint32_t max_classes);

    void update(
        float x,
        float y,
        float z,
        int32_t class_id,
        float confidence,
        double stamp);

    std::vector<VoxelOutput> snapshot() const;

    std::size_t size() const;

    void clear();

    VoxelKey keyFromPoint(
        float x,
        float y,
        float z) const;


private:

    float resolution_;
    float min_confidence_;
    uint32_t max_classes_;

    mutable std::mutex mutex_;

    std::unordered_map<
        VoxelKey,
        Voxel,
        VoxelKeyHash
    > voxels_;
};

}  // namespace persistent_semantic_mapping
