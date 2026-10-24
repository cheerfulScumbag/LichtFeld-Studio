/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "collision_sparse.hpp"
#include <atomic>
#include <cmath>
#include <format>
#include <memory>
#include <mutex>
#include <numeric>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <thread>

namespace lfs::core::collision_detail {
    namespace {
        void check(std::stop_token stop) {
            if (stop.stop_requested())
                throw std::runtime_error("Collision generation cancelled");
        }
        glm::vec3 extent(const CollisionGaussian& g) {
            glm::vec3 result;
            for (int a = 0; a < 3; ++a)
                result[a] = 3 * glm::length(glm::vec3(g.axes[0][a], g.axes[1][a], g.axes[2][a]));
            return result;
        }
    } // namespace

    PreparedCollision prepareCollision(std::span<const CollisionGaussian> source, CollisionSettings settings,
                                       std::stop_token stop) {
        const double cutoff = -std::log1p(-double(settings.opacity));
        struct Bounds {
            glm::vec3 min, max;
            float opacity;
        };
        std::vector<Bounds> bounds;
        bounds.reserve(source.size());
        glm::vec3 lo(std::numeric_limits<float>::max()), hi(-lo);
        for (const auto& g : source) {
            check(stop);
            auto finite = [](glm::vec3 p) { return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z); };
            if (!finite(g.center) || !finite(g.axes[0]) || !finite(g.axes[1]) || !finite(g.axes[2]) ||
                !std::isfinite(g.opacity) || g.opacity < 0 || g.opacity > 1)
                throw std::invalid_argument("Non-finite or invalid Gaussian in collision source");
            if (g.opacity == 0 || std::abs(glm::determinant(g.axes)) < 1e-20f)
                continue;
            auto e = extent(g) + settings.voxel_size * 0.5f;
            if (!finite(e) || !finite(g.center - e) || !finite(g.center + e))
                throw std::invalid_argument("Gaussian collision support exceeds finite coordinate range");
            bounds.push_back({g.center - e, g.center + e, g.opacity});
            lo = glm::min(lo, bounds.back().min);
            hi = glm::max(hi, bounds.back().max);
        }
        // This pass only tightens the world bounds; it never filters the
        // Gaussians evaluated by the voxelizer. A sparse grid does not allocate
        // its empty AABB, so overlap proofs buy nothing when all supports already
        // fit the coordinate limit. Dense aerial scans otherwise spend hundreds
        // of millions of BVH visits proving optional bounds before CUDA starts.
        const auto support_origin = glm::floor(lo / settings.voxel_size) * settings.voxel_size - 2 * settings.voxel_size;
        const auto support_dimensions = glm::ceil((glm::dvec3(hi) - glm::dvec3(support_origin)) / double(settings.voxel_size)) + 3.0;
        const bool supports_fit = glm::all(glm::greaterThanEqual(support_dimensions, glm::dvec3(1))) &&
                                  glm::all(glm::lessThanEqual(support_dimensions, glm::dvec3(1 << 21)));
        if (!supports_fit) {
            lo = glm::vec3(std::numeric_limits<float>::max());
            hi = -lo;
            struct BvhNode {
                glm::vec3 min, max;
                size_t begin, end;
                uint32_t left = 0, right = 0;
            };
            std::vector<uint32_t> order(bounds.size());
            std::iota(order.begin(), order.end(), 0);
            std::vector<BvhNode> bvh;
            std::function<uint32_t(size_t, size_t)> buildBounds = [&](size_t begin, size_t end) {
                check(stop);
                const auto index = uint32_t(bvh.size());
                glm::vec3 min(std::numeric_limits<float>::max()), max(-min);
                for (size_t i = begin; i < end; ++i) {
                    min = glm::min(min, bounds[order[i]].min);
                    max = glm::max(max, bounds[order[i]].max);
                }
                bvh.push_back({min, max, begin, end});
                if (end - begin > 16) {
                    auto delta = glm::dvec3(max) - glm::dvec3(min);
                    int axis = delta.x >= delta.y && delta.x >= delta.z ? 0 : delta.y >= delta.z ? 1
                                                                                                 : 2;
                    auto middle = begin + (end - begin) / 2;
                    std::nth_element(order.begin() + begin, order.begin() + middle, order.begin() + end,
                                     [&](auto a, auto b) {
                                         return double(bounds[a].min[axis]) + bounds[a].max[axis] <
                                                double(bounds[b].min[axis]) + bounds[b].max[axis];
                                     });
                    auto left = buildBounds(begin, middle), right = buildBounds(middle, end);
                    bvh[index].left = left;
                    bvh[index].right = right;
                }
                return index;
            };
            buildBounds(0, order.size());
            // Each support proof is independent. Keep the final reduction ordered,
            // but avoid a million serial BVH queries and std::function allocations.
            std::vector<uint8_t> eligible(bounds.size());
            std::atomic<uint64_t> bound_work{0};
            tbb::task_group_context context(tbb::task_group_context::isolated);
            const auto count = bounds.size() >= 4096 ? std::min(8u, std::max(1u, std::thread::hardware_concurrency())) : 1u;
            tbb::task_arena arena(count);
            arena.execute([&] {
                tbb::parallel_for(size_t(0), (bounds.size() + 255) / 256, [&](size_t batch) {
                    uint64_t local_work = 0;
                    for (size_t row = batch * 256; row < std::min((batch + 1) * 256, bounds.size()); ++row) {
                        check(stop);
                    const auto& bound = bounds[row];
                    double upper = 0;
                    auto accumulate = [&](auto&& self, uint32_t i) -> void {
                        if (upper >= cutoff || context.is_group_execution_cancelled())
                            return;
                        if ((++local_work & 4095) == 0) {
                            check(stop);
                            if (bound_work.fetch_add(4096, std::memory_order_relaxed) + 4096 > 500'000'000)
                                throw std::runtime_error("Sparse opacity-bound work budget exceeded");
                        }
                        const auto& node = bvh[i];
                        if (glm::any(glm::greaterThan(bound.min, node.max)) || glm::any(glm::lessThan(bound.max, node.min)))
                            return;
                        if (node.left) {
                            self(self, node.left);
                            self(self, node.right);
                        } else {
                            for (size_t j = node.begin; j < node.end && upper < cutoff; ++j) {
                                const auto& other = bounds[order[j]];
                                if (glm::all(glm::lessThanEqual(bound.min, other.max)) && glm::all(glm::greaterThanEqual(bound.max, other.min)))
                                    upper += other.opacity + 8 * std::numeric_limits<float>::epsilon();
                            }
                        }
                    };
                    if (bound.opacity < cutoff)
                        accumulate(accumulate, 0);
                    eligible[row] = bound.opacity >= cutoff || upper >= cutoff;
                    }
                    if (bound_work.fetch_add(local_work & 4095, std::memory_order_relaxed) + (local_work & 4095) > 500'000'000)
                        throw std::runtime_error("Sparse opacity-bound work budget exceeded"); }, context);
            });
            for (size_t i = 0; i < bounds.size(); ++i)
                if (eligible[i]) {
                    lo = glm::min(lo, bounds[i].min);
                    hi = glm::max(hi, bounds[i].max);
                }
        }
        if (glm::any(glm::greaterThan(lo, hi)))
            throw std::runtime_error("No occupied voxels; lower opacity or voxel size");
        CollisionVolume volume;
        volume.settings = settings;
        const float size = settings.voxel_size;
        volume.origin = glm::floor(lo / size) * size - 2 * size;
        auto dims = glm::ceil((glm::dvec3(hi) - glm::dvec3(volume.origin)) / double(size)) + 3.0;
        if (glm::any(glm::greaterThan(dims, glm::dvec3(1 << 21))) ||
            glm::any(glm::lessThan(dims, glm::dvec3(1))) ||
            !std::isfinite(dims.x) || !std::isfinite(dims.y) || !std::isfinite(dims.z))
            throw std::runtime_error("Sparse collision coordinate range exceeded (maximum 2^21 cells per axis)");
        volume.dimensions = glm::ivec3(dims);
        std::vector<PreparedGaussian> prepared;
        prepared.reserve(source.size());
        size_t bound_index = 0;
        const auto world_max = glm::dvec3(volume.origin) + glm::dvec3(volume.dimensions) * double(size);
        for (const auto& g : source) {
            if (g.opacity == 0 || std::abs(glm::determinant(g.axes)) < 1e-20f)
                continue;
            const auto& bound = bounds[bound_index++];
            // Weak isolated supports can lie arbitrarily far outside the effective
            // grid. Clip before converting coordinates, retaining their contribution
            // wherever they overlap the grid without overflowing cell indices.
            const auto padding = (glm::abs(glm::dvec3(bound.min)) + glm::abs(glm::dvec3(bound.max)) +
                                  glm::abs(glm::dvec3(volume.origin)) + 1.0) *
                                 (8 * double(std::numeric_limits<float>::epsilon()));
            const auto clipped_min = glm::max(glm::dvec3(bound.min) - padding, glm::dvec3(volume.origin));
            const auto clipped_max = glm::min(glm::dvec3(bound.max) + padding, world_max);
            if (glm::any(glm::greaterThan(clipped_min, clipped_max)))
                continue;
            const auto cell_min = glm::clamp(glm::ivec3(glm::floor((clipped_min - glm::dvec3(volume.origin)) / double(size))),
                                             glm::ivec3(0), volume.dimensions);
            const auto cell_max = glm::clamp(glm::ivec3(glm::floor((clipped_max - glm::dvec3(volume.origin)) / double(size))) + 1,
                                             glm::ivec3(0), volume.dimensions);
            prepared.push_back({g.center, glm::inverse(g.axes), extent(g), g.opacity, cell_min, cell_max});
        }
        return {std::move(volume), std::move(prepared)};
    }

    CollisionVolume voxelizeSparse(std::span<const CollisionGaussian> source, CollisionSettings settings,
                                   std::stop_token stop, CollisionProgress progress, unsigned parallelism) {
        if (progress)
            progress(0, "Preparing Gaussian bounds on CPU");
        const double cutoff = -std::log1p(-double(settings.opacity));
        auto input = prepareCollision(source, settings, stop);
        auto volume = std::move(input.volume);
        const auto& prepared = input.gaussians;
        const float size = settings.voxel_size;
        Tree tree(volume.dimensions, stop);
        std::vector<uint32_t> all(prepared.size());
        std::iota(all.begin(), all.end(), 0);

        const auto available = parallelism ? parallelism : std::max(1u, std::thread::hardware_concurrency());
        const unsigned workers = source.size() >= 4096 && tree.size > 8 ? std::min(available, 8u) : 1;
        std::atomic<size_t> allocated{workers > 1 ? 9u : 1u};
        tbb::task_group_context context(tbb::task_group_context::isolated);
        std::atomic<uint64_t> shared_work{0};
        auto buildBranch = [&](int octant) {
            Tree tree(volume.dimensions, stop);
            uint64_t work = 0;
            auto children = [&](uint32_t index, auto&& fn) {
                if (allocated.fetch_add(8, std::memory_order_relaxed) + 8 > Tree::max_nodes)
                    throw std::runtime_error("Sparse collision node memory budget exceeded");
                const auto base = uint32_t(tree.nodes.size());
                tree.nodes.resize(tree.nodes.size() + 8);
                tree.nodes[index] = base + 2;
                for (int child = 0; child < 8; ++child)
                    fn(base + child, child);
                const auto first = tree.nodes[base];
                if (first <= 1 && std::all_of(tree.nodes.begin() + base, tree.nodes.begin() + base + 8,
                                              [first](auto value) { return value == first; })) {
                    tree.nodes[index] = first;
                    allocated.fetch_sub(tree.nodes.size() - base, std::memory_order_relaxed);
                    tree.nodes.resize(base);
                }
            };
            std::function<void(uint32_t, glm::ivec3, int, const std::vector<uint32_t>&)> build;
            build = [&](uint32_t node, glm::ivec3 lo, int n, const std::vector<uint32_t>& candidates) {
                check(stop);
                if (context.is_group_execution_cancelled())
                    throw std::runtime_error("Collision generation cancelled after worker failure");
                auto hi = glm::min(lo + n, volume.dimensions);
                if (glm::any(glm::greaterThanEqual(lo, hi)))
                    return;
                const auto middle = glm::dvec3(volume.origin) +
                                    (glm::dvec3(lo) + glm::dvec3(hi)) * (double(size) / 2);
                const auto half = glm::dvec3(hi - lo) * (double(size) / 2);
                // Bound float evaluation error as well as the whole voxel-volume box.
                const auto error = (glm::abs(middle) + glm::abs(glm::dvec3(volume.origin)) + 1.0) *
                                   (8 * double(std::numeric_limits<float>::epsilon()));
                std::vector<uint32_t> relevant;
                relevant.reserve(candidates.size());
                double upper = 0, lower = 0;
                for (auto index : candidates) {
                    if ((++work & 4095) == 0) {
                        check(stop);
                        if (shared_work.fetch_add(4096, std::memory_order_relaxed) > 4'000'000'000ULL)
                            throw std::runtime_error("Sparse Gaussian active-work budget exceeded");
                    }
                    if (work > 4'000'000'000ULL)
                        throw std::runtime_error("Sparse Gaussian active-work budget exceeded");
                    const auto& g = prepared[index];
                    if (glm::any(glm::greaterThanEqual(lo, g.max)) || glm::any(glm::lessThanEqual(hi, g.min)))
                        continue;
                    auto inv = glm::dmat3(g.inverse);
                    auto q = inv * (middle - glm::dvec3(g.center));
                    auto h = half + error + glm::abs(glm::dvec3(g.center)) * (8 * double(std::numeric_limits<float>::epsilon()));
                    auto qr = glm::abs(inv[0]) * h.x + glm::abs(inv[1]) * h.y + glm::abs(inv[2]) * h.z;
                    auto near = glm::max(glm::abs(q) - qr, glm::dvec3(0));
                    double min_distance = glm::dot(near, near);
                    relevant.push_back(index);
                    upper += g.opacity * std::exp(-0.5 * min_distance);
                    auto far = glm::abs(q) + qr;
                    double max_distance = glm::dot(far, far);
                    if (max_distance <= 9)
                        lower += g.opacity * std::exp(-0.5 * max_distance);
                }
                const double rounding = relevant.size() * (8 * double(std::numeric_limits<float>::epsilon()));
                if (relevant.empty() || upper + rounding < cutoff)
                    return;
                if (lower > 0 && lower - rounding >= cutoff &&
                    glm::all(glm::equal(hi, lo + n))) {
                    tree.nodes[node] = 1;
                    return;
                }
                if (n > 8) {
                    children(node, [&](uint32_t index, int child) {
                        build(index, lo + glm::ivec3(child & 1, (child >> 1) & 1, child >> 2) * (n / 2),
                              n / 2, relevant);
                    });
                    return;
                }
                std::array<float, 512> density{};
                const auto total = uint32_t((hi.x - lo.x) * (hi.y - lo.y) * (hi.z - lo.z));
                auto remaining = total;
                for (auto index : relevant) {
                    const auto& g = prepared[index];
                    if (remaining == 0)
                        break;
                    const auto from = glm::max(lo, g.min), to = glm::min(hi, g.max);
                    for (int z = from.z; z < to.z; ++z)
                        for (int y = from.y; y < to.y; ++y)
                            for (int x = from.x; x < to.x; ++x) {
                                if ((++work & 4095) == 0) {
                                    check(stop);
                                    if (shared_work.fetch_add(4096, std::memory_order_relaxed) > 4'000'000'000ULL)
                                        throw std::runtime_error("Sparse Gaussian active-work budget exceeded");
                                }
                                auto& value = density[(z - lo.z) * 64 + (y - lo.y) * 8 + x - lo.x];
                                if (value > 0 && double(value) >= cutoff)
                                    continue;
                                const auto center = volume.origin + (glm::vec3(x, y, z) + 0.5f) * size;
                                if (glm::any(glm::greaterThan(glm::abs(center - g.center), g.extent + size * 0.5f)))
                                    continue;
                                const auto closest = glm::clamp(g.center, center - size * 0.5f, center + size * 0.5f);
                                const auto q = g.inverse * (closest - g.center);
                                value += g.opacity * std::exp(-0.5f * glm::dot(q, q));
                                if (value > 0 && double(value) >= cutoff)
                                    --remaining;
                            }
                }
                if (remaining == total)
                    return;
                if (remaining == 0 && glm::all(glm::equal(hi, lo + n))) {
                    tree.nodes[node] = 1;
                    return;
                }
                std::function<void(uint32_t, glm::ivec3, int)> emit = [&](uint32_t index, glm::ivec3 p, int edge) {
                    if (edge == 1) {
                        if (glm::any(glm::greaterThanEqual(p, hi)))
                            return;
                        const auto q = p - lo;
                        const auto value = density[q.z * 64 + q.y * 8 + q.x];
                        tree.nodes[index] = value > 0 && double(value) >= cutoff;
                        return;
                    }
                    children(index, [&](uint32_t child_index, int child) {
                        emit(child_index, p + glm::ivec3(child & 1, (child >> 1) & 1, child >> 2) * (edge / 2),
                             edge / 2);
                    });
                };
                emit(node, lo, n);
            };
            if (octant < 0)
                build(0, {}, tree.size, all);
            else
                build(0, glm::ivec3(octant & 1, (octant >> 1) & 1, octant >> 2) * (tree.size / 2), tree.size / 2, all);
            return std::move(tree.nodes);
        };
        if (workers == 1) {
            tree.nodes = buildBranch(-1);
        } else {
            std::array<std::vector<uint32_t>, 8> branches;
            unsigned completed = 0;
            std::mutex progress_mutex;
            tbb::task_arena arena(workers);
            arena.execute([&] {
                tbb::parallel_for(0u, 8u, [&](unsigned octant) {
                    branches[octant] = buildBranch(int(octant));
                    if (progress) {
                        std::lock_guard lock(progress_mutex);
                        progress(.1f + .9f * ++completed / 8, "Evaluating sparse collision regions on CPU");
                    } }, context);
            });
            tree.nodes.assign(9, 0);
            tree.nodes[0] = 3;
            tree.nodes.reserve(allocated.load());
            for (unsigned octant = 0; octant < 8; ++octant) {
                check(stop);
                auto& branch = branches[octant];
                const uint32_t shift = uint32_t(tree.nodes.size()) - 1;
                tree.nodes[octant + 1] = branch[0] <= 1 ? branch[0] : branch[0] + shift;
                for (size_t i = 1; i < branch.size(); ++i)
                    tree.nodes.push_back(branch[i] <= 1 ? branch[i] : branch[i] + shift);
                branch.clear();
                branch.shrink_to_fit();
            }
        }
        volume.tree_size = tree.size;
        volume.tree = std::move(tree.nodes);
        if (progress)
            progress(1, "CPU collision cells ready");
        check(stop);
        return volume;
    }

} // namespace lfs::core::collision_detail
