/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "collision_tree.hpp"
#include "core/collision.hpp"

namespace lfs::core::collision_detail {
    float recommendCollisionVoxelSize(std::span<const CollisionGaussian> source, float units_per_meter, std::stop_token stop = {});
    struct AdaptiveSelection {
        std::vector<uint32_t> indices;
        std::vector<uint32_t> oversized_indices;
        double area_limit = 0;
        std::string note;
    };
    AdaptiveSelection selectAdaptiveSource(std::span<const CollisionGaussian> source,
                                           const CollisionSettings& settings, std::stop_token stop = {},
                                           CollisionProgress progress = {});
    struct PreparedGaussian {
        glm::vec3 center;
        glm::mat3 inverse;
        glm::vec3 extent;
        float opacity;
        glm::ivec3 min, max;
    };
    struct PreparedCollision {
        CollisionVolume volume;
        std::vector<PreparedGaussian> gaussians;
    };
    PreparedCollision prepareCollision(std::span<const CollisionGaussian> source, CollisionSettings settings,
                                       std::stop_token stop);

    CollisionVolume voxelizeSparse(std::span<const CollisionGaussian> source, CollisionSettings settings,
                                   std::stop_token stop, CollisionProgress progress = {}, unsigned parallelism = 0);
    inline Tree volumeTree(const CollisionVolume& volume, std::stop_token stop = {}) {
        Tree tree(volume.dimensions, stop);
        if (!volume.tree.empty()) {
            tree.nodes = volume.tree;
        } else {
            volume.forEachSolidBox([&](auto lo, auto hi) { tree.setBox(lo, hi); });
            tree.compact();
        }
        return tree;
    }
} // namespace lfs::core::collision_detail
