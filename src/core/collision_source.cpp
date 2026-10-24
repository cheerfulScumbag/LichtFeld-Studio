/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision_source.hpp"
#include "core/mesh_data.hpp"
#include <algorithm>
#include <cmath>
#include <glm/gtc/quaternion.hpp>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

namespace lfs::core {
    std::vector<CollisionGaussian> extractCollisionSource(
        std::span<const CollisionGeometry> snapshots, const glm::mat4& world_to_grid,
        std::stop_token stop, unsigned parallelism) {
        auto check = [&] {
            if (stop.stop_requested())
                throw std::runtime_error("Collision generation cancelled");
        };
        check();
        struct Batch {
            size_t snapshot, begin, end, kept = 0, output = 0;
        };
        std::vector<Batch> batches;
        size_t source_count = 0;
        for (size_t n = 0; n < snapshots.size(); ++n) {
            const auto& source = snapshots[n];
            if (source.size > CollisionGaussian::max_count - source_count)
                throw std::runtime_error("Collision source exceeds 32 million Gaussians");
            source_count += source.size;
            if (!source.size)
                continue;
            auto valid = [&](const Tensor& tensor, size_t count, DataType type) {
                return tensor.is_valid() && tensor.device() == Device::CPU && tensor.dtype() == type &&
                       tensor.is_contiguous() && tensor.numel() == count;
            };
            if (!valid(source.means, source.size * 3, DataType::Float32) ||
                !valid(source.scaling, source.size * 3, DataType::Float32) ||
                !valid(source.rotation, source.size * 4, DataType::Float32) ||
                !valid(source.opacity, source.size, DataType::Float32) ||
                (source.deleted.is_valid() && !valid(source.deleted, source.size, DataType::Bool)))
                throw std::invalid_argument("Collision source requires contiguous owned CPU geometry snapshots");
            for (size_t begin = 0; begin < source.size; begin += 4096)
                batches.push_back({n, begin, std::min(begin + 4096, source.size)});
        }
        tbb::task_arena arena(std::clamp(parallelism, 1u, 8u));
        auto each_batch = [&](auto&& fn) {
            arena.execute([&] {
                tbb::parallel_for(tbb::blocked_range<size_t>(0, batches.size()), [&](const auto& range) {
                    for (size_t b = range.begin(); b < range.end(); ++b) {
                        check();
                        fn(batches[b]);
                    }
                });
            });
        };
        each_batch([&](Batch& batch) {
            const auto& source = snapshots[batch.snapshot];
            batch.kept = batch.end - batch.begin;
            if (source.deleted.is_valid()) {
                const auto* deleted = source.deleted.ptr<bool>();
                batch.kept -= std::count(deleted + batch.begin, deleted + batch.end, true);
            }
        });
        size_t count = 0;
        for (auto& batch : batches) {
            batch.output = count;
            count += batch.kept;
        }
        // Exactly one output allocation. Prefix offsets give workers disjoint
        // spans without atomics, per-thread full copies or a compaction copy.
        std::vector<CollisionGaussian> result(count);
        each_batch([&](const Batch& batch) {
            const auto& source = snapshots[batch.snapshot];
            const auto* means = source.means.ptr<float>();
            const auto* scaling = source.scaling.ptr<float>();
            const auto* rotation = source.rotation.ptr<float>();
            const auto* opacity = source.opacity.ptr<float>();
            const auto* deleted = source.deleted.is_valid() ? source.deleted.ptr<bool>() : nullptr;
            const auto transform = world_to_grid * source.world_transform;
            size_t output = batch.output;
            for (size_t i = batch.begin; i < batch.end; ++i) {
                if ((i & 1023) == 0)
                    check();
                if (deleted && deleted[i])
                    continue;
                auto q = glm::normalize(glm::quat(rotation[i * 4], rotation[i * 4 + 1], rotation[i * 4 + 2], rotation[i * 4 + 3]));
                auto axes = glm::mat3_cast(q);
                for (int axis = 0; axis < 3; ++axis)
                    axes[axis] *= std::exp(scaling[i * 3 + axis]);
                result[output++] = {glm::vec3(transform * glm::vec4(means[i * 3], means[i * 3 + 1], means[i * 3 + 2], 1)),
                                    glm::mat3(transform) * axes, 1.0f / (1.0f + std::exp(-opacity[i]))};
            }
        });
        check();
        return result;
    }
    std::shared_ptr<MeshData> makeCollisionMesh(CollisionMesh data, std::stop_token stop) {
        if (data.vertices.empty() || data.indices.empty())
            throw std::runtime_error("Collision generation produced an empty surface");
        auto mesh = std::make_shared<MeshData>();
        auto vertices = data.vertices.size() / 3, faces = data.indices.size() / 3;
        mesh->vertices = Tensor::from_vector(data.vertices, {vertices, 3}, Device::CPU);
        mesh->indices = Tensor::from_vector(data.indices, {faces, 3}, Device::CPU);
        // A simple parallel-safe indexed normal pass avoids OpenMesh topology
        // construction and its large adjacency/conversion allocations.
        mesh->normals = Tensor::zeros({vertices, 3}, Device::CPU);
        auto* normals = mesh->normals.ptr<float>();
        for (size_t f = 0; f < data.indices.size(); f += 3) {
            if ((f & 4095) == 0 && stop.stop_requested())
                throw std::runtime_error("Collision surface cancelled");
            const auto a = data.indices[f], b = data.indices[f + 1], c = data.indices[f + 2];
            auto point = [&](int i) { return glm::vec3(data.vertices[i * 3], data.vertices[i * 3 + 1], data.vertices[i * 3 + 2]); };
            auto n = glm::cross(point(b) - point(a), point(c) - point(a));
            for (int i : {a, b, c})
                for (int k = 0; k < 3; ++k)
                    normals[i * 3 + k] += n[k];
        }
        for (size_t i = 0; i < vertices; ++i) {
            if ((i & 4095) == 0 && stop.stop_requested())
                throw std::runtime_error("Collision surface cancelled");
            glm::vec3 n(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
            if (auto length = glm::length(n); length > 0)
                n /= length;
            for (int k = 0; k < 3; ++k)
                normals[i * 3 + k] = n[k];
        }
        return mesh;
    }
} // namespace lfs::core
