/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/collision.hpp"
#include "core/tensor.hpp"

namespace lfs::core {
    // Raw geometry in log-scale/logit-opacity domains, without SH or scene state.
    // CPU buffers must remain unchanged until extraction returns.
    struct CollisionGeometry {
        Tensor means, scaling, rotation, opacity, deleted;
        glm::mat4 world_transform{1};
        size_t size = 0;
    };
    [[nodiscard]] LFS_CORE_API std::vector<CollisionGaussian> extractCollisionSource(
        std::span<const CollisionGeometry> sources, const glm::mat4& world_to_grid = glm::mat4(1),
        std::stop_token stop = {}, unsigned parallelism = 8);
} // namespace lfs::core
