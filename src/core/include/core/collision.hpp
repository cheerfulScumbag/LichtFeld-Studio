/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <span>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

namespace lfs::core {

    struct LFS_CORE_API CollisionSettings {
        bool cluster = false;
        // Scale-invariant reconstruction cleanup; seeded clustering uses source
        // connectivity rather than a scene-dependent opacity threshold.
        bool adaptive_filter = true;
        bool auto_resolution = false; // UI defaults to Auto; explicit API resolutions remain exact.
        float max_gaussian_area = 0;  // Three-sigma footprint, scene units squared; zero selects a relative outlier fence.
        float cluster_resolution = 1.0f;
        float voxel_size = 0.05f;
        float opacity = 0.1f;
        glm::vec3 seed{0};
        // Reconstruction coordinates are not necessarily metres. Explicit calibration,
        // never inferred from arbitrary camera/scene extents.
        float units_per_meter = 1.0f;
        void validate() const;
        friend bool operator==(const CollisionSettings&, const CollisionSettings&) = default;
    };

    // axes maps a unit sphere to the Gaussian's rotated/scaled one-sigma ellipsoid.
    struct CollisionGaussian {
        // Bound source snapshots independently of sparse occupancy/output budgets.
        static constexpr size_t max_count = 32'000'000;
        glm::vec3 center;
        glm::mat3 axes;
        float opacity;
    };

    namespace collision_detail {
        struct Leaf {
            glm::ivec3 min;
            int size;
            uint32_t index;
            bool solid;
            glm::ivec3 max() const { return min + size; }
        };
        // Validated tree; p is relative to the supplied subtree root.
        inline Leaf lookupLeaf(std::span<const uint32_t> nodes, glm::ivec3 p, int edge, uint32_t index = 0) {
            glm::ivec3 min(0);
            while (nodes[index] > 1) {
                edge /= 2;
                const auto bit = glm::ivec3(glm::greaterThanEqual(p, min + edge));
                index = nodes[index] - 2 + bit.x + 2 * bit.y + 4 * bit.z;
                min += bit * edge;
            }
            return {min, edge, index, nodes[index] != 0};
        }
    } // namespace collision_detail

    // Progress is a fraction in [0, 1] plus the current operation.
    using CollisionProgress = std::function<void(float, std::string)>;
    inline CollisionProgress collisionProgressRange(CollisionProgress callback, float start, float end) {
        if (!callback)
            return {};
        return [callback = std::move(callback), start, end](float value, std::string stage) {
            callback(start + (end - start) * value, std::move(stage));
        };
    }

    struct LFS_CORE_API CollisionVolume {
        static constexpr size_t max_cells = 16'000'000;
        static constexpr size_t max_tree_words = 512'000'001;
        static_assert(max_tree_words + 2 < (uint32_t(1) << 31),
                      "GPU occupancy reserves the high bit for brick pointers");
        CollisionSettings settings;
        glm::vec3 origin{0};
        glm::ivec3 dimensions{0};
        std::vector<uint8_t> solid;
        // Sparse payload: 0/1 uniform leaves, otherwise child-base + 2.
        std::vector<uint32_t> tree;
        int tree_size = 0;
        std::string warning;
        friend bool operator==(const CollisionVolume&, const CollisionVolume&) = default;

        [[nodiscard]] size_t index(glm::ivec3 p) const;
        [[nodiscard]] bool contains(glm::ivec3 p) const;
        [[nodiscard]] bool occupied(glm::ivec3 p) const;
        [[nodiscard]] glm::ivec3 cell(glm::vec3 p) const;
        void validate() const;
        [[nodiscard]] std::vector<uint32_t> octreeOccupancy() const;
        [[nodiscard]] uint64_t occupiedCount() const;
        void forEachSolidBox(const std::function<void(glm::ivec3, glm::ivec3)>& fn) const;
    };

    using CollisionVoxelizer = CollisionVolume (*)(std::span<const CollisionGaussian>, CollisionSettings,
                                                   std::stop_token, CollisionProgress);
    [[nodiscard]] LFS_CORE_API CollisionVolume voxelizeCollisionGpu(
        std::span<const CollisionGaussian> gaussians, CollisionSettings settings,
        std::stop_token stop = {}, CollisionProgress progress = {});

    [[nodiscard]] LFS_CORE_API CollisionVolume generateCollision(
        std::span<const CollisionGaussian> gaussians, CollisionSettings settings,
        std::stop_token stop = {}, CollisionProgress progress = {}, CollisionVoxelizer voxelizer = nullptr);
} // namespace lfs::core
