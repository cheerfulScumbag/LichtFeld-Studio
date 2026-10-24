/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include "collision_sparse.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <unordered_map>

namespace lfs::core {
    namespace {
        struct GridPointHash {
            size_t operator()(const std::array<int32_t, 3>& p) const noexcept {
                size_t h = std::hash<int32_t>{}(p[0]);
                for (int a = 1; a < 3; ++a)
                    h ^= std::hash<int32_t>{}(p[a]) + size_t(0x9e3779b9) + (h << 6) + (h >> 2);
                return h;
            }
        };
        void checkpoint(std::stop_token stop) {
            if (stop.stop_requested())
                throw std::runtime_error("Collision generation cancelled");
        }
        bool finite(glm::vec3 p) {
            return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        }
        void positive(float v, const char* name, bool zero = false) {
            if (!std::isfinite(v) || (zero ? v < 0 : v <= 0))
                throw std::invalid_argument(std::string(name) + " must be finite and positive");
        }
        void probability(float v, const char* name) {
            positive(v, name, true);
            if (v > 1)
                throw std::invalid_argument(std::string(name) + " must be in [0, 1]");
        }
        size_t cellCount(glm::ivec3 d) {
            if (glm::any(glm::lessThanEqual(d, glm::ivec3(0))))
                throw std::invalid_argument("Invalid collision grid dimensions");
            uint64_t n = 1;
            for (int a = 0; a < 3; ++a) {
                if (static_cast<uint64_t>(d[a]) > CollisionVolume::max_cells / n)
                    throw std::runtime_error("Collision grid exceeds 16 million cells; increase voxel size or isolate the scene");
                n *= d[a];
            }
            return n;
        }
    } // namespace

    void CollisionSettings::validate() const {
        positive(voxel_size, "Voxel size");
        positive(units_per_meter, "Scene units per metre");
        positive(cluster_resolution, "Cluster resolution");
        positive(max_gaussian_area, "Maximum Gaussian area", true);
        probability(opacity, "Voxel opacity");
        if (!finite(seed))
            throw std::invalid_argument("Seed must be finite");
    }
    size_t CollisionVolume::index(glm::ivec3 p) const {
        return (size_t(p.z) * dimensions.y + p.y) * dimensions.x + p.x;
    }
    bool CollisionVolume::contains(glm::ivec3 p) const {
        return glm::all(glm::greaterThanEqual(p, glm::ivec3(0))) && glm::all(glm::lessThan(p, dimensions));
    }
    bool CollisionVolume::occupied(glm::ivec3 p) const {
        if (!contains(p))
            return false;
        if (tree.empty())
            return solid[index(p)] != 0;
        return collision_detail::lookupLeaf(tree, p, tree_size).solid;
    }
    glm::ivec3 CollisionVolume::cell(glm::vec3 p) const {
        const auto q = glm::floor((p - origin) / settings.voxel_size);
        if (!finite(q) || glm::any(glm::greaterThan(glm::abs(q), glm::vec3(1e9f))))
            throw std::invalid_argument("Collision query outside supported coordinate range");
        return glm::ivec3(q);
    }
    void CollisionVolume::validate() const {
        settings.validate();
        if (!finite(origin))
            throw std::invalid_argument("Invalid collision volume payload");
        if (!tree.empty()) {
            collision_detail::Tree expected(dimensions);
            if (!solid.empty() || tree_size != expected.size || tree.size() > collision_detail::Tree::max_nodes)
                throw std::invalid_argument("Invalid sparse collision volume payload");
            if ((tree.size() - 1) % 8)
                throw std::invalid_argument("Invalid sparse collision sibling blocks");
            // Each canonical eight-child block has exactly one earlier parent.
            // Forward references rule out cycles, and block levels check depth.
            // One byte per block replaces one byte per node plus recursive calls.
            std::vector<uint8_t> levels((tree.size() - 1) / 8);
            size_t branches = 0;
            const auto root_level = std::bit_width(unsigned(tree_size));
            for (size_t i = 0; i < tree.size(); ++i) {
                const auto level = i ? levels[(i - 1) / 8] : root_level;
                if (!level)
                    throw std::invalid_argument("Unreachable sparse collision nodes");
                if (tree[i] <= 1)
                    continue;
                const auto base = tree[i] - 2;
                if (level <= 1 || base <= i || (base - 1) % 8 || uint64_t(base) + 8 > tree.size())
                    throw std::invalid_argument("Invalid sparse collision branch");
                auto& child_level = levels[(base - 1) / 8];
                if (child_level)
                    throw std::invalid_argument("Invalid sparse collision child reference");
                child_level = uint8_t(level - 1);
                ++branches;
            }
            if (branches != levels.size())
                throw std::invalid_argument("Unreachable sparse collision nodes");
            return;
        }
        if (solid.size() != cellCount(dimensions))
            throw std::invalid_argument("Invalid collision volume payload");
        if (std::any_of(solid.begin(), solid.end(), [](auto v) { return v > 1; }))
            throw std::invalid_argument("Invalid collision occupancy value");
    }
    namespace collision_detail {
        float recommendCollisionVoxelSize(std::span<const CollisionGaussian> source, float units, std::stop_token stop) {
            positive(units, "Scene units per metre");
            std::vector<double> areas;
            areas.reserve(source.size());
            for (size_t i = 0; i < source.size(); ++i) {
                if ((i & 4095) == 0)
                    checkpoint(stop);
                const auto& g = source[i];
                if (g.opacity <= 0)
                    continue;
                const auto area = std::max({glm::length(glm::cross(glm::dvec3(g.axes[0]), glm::dvec3(g.axes[1]))),
                                            glm::length(glm::cross(glm::dvec3(g.axes[0]), glm::dvec3(g.axes[2]))),
                                            glm::length(glm::cross(glm::dvec3(g.axes[1]), glm::dvec3(g.axes[2])))});
                if (area > 0 && std::isfinite(area))
                    areas.push_back(area);
            }
            if (areas.empty())
                return .05f * units;
            const auto middle = areas.begin() + areas.size() / 2;
            std::nth_element(areas.begin(), middle, areas.end());
            // Robust observed surface footprint, not empty scene AABB or a metre
            // inference. Aim for 32M surface samples of two-sigma ellipses. This
            // is an estimate, never a guarantee or a hidden retry/coarsening.
            const auto edge = std::max(.05 * double(units), std::sqrt(*middle * areas.size() * 4 * std::numbers::pi / 32'000'000));
            return float(std::ceil(edge / units * 100 - 1e-6) / 100 * units);
        }

        AdaptiveSelection selectAdaptiveSource(std::span<const CollisionGaussian> source,
                                               const CollisionSettings& settings, std::stop_token stop,
                                               CollisionProgress progress) {
            checkpoint(stop);
            AdaptiveSelection result;
            result.indices.reserve(source.size());
            std::vector<float> thickness;
            thickness.reserve(source.size());
            std::vector<float> volume;
            volume.reserve(source.size());
            std::vector<float> area_samples;
            area_samples.reserve(source.size());
            auto measure = [](const CollisionGaussian& g, double* projected_area = nullptr) {
                // Minimum parallelepiped altitude: rotation invariant, and thin
                // sheared/anisotropic sheets stay thin. A long axis alone is not
                // evidence of a volumetric reconstruction outlier.
                const double area = std::max({glm::length(glm::cross(glm::dvec3(g.axes[0]), glm::dvec3(g.axes[1]))),
                                              glm::length(glm::cross(glm::dvec3(g.axes[0]), glm::dvec3(g.axes[2]))),
                                              glm::length(glm::cross(glm::dvec3(g.axes[1]), glm::dvec3(g.axes[2])))});
                if (projected_area)
                    *projected_area = area;
                return area > 0 ? float(std::abs(glm::determinant(glm::dmat3(g.axes))) / area) : 0.f;
            };
            for (size_t i = 0; i < source.size(); ++i) {
                if ((i & 4095) == 0)
                    checkpoint(stop);
                const auto& g = source[i];
                if (!finite(g.center) || !finite(g.axes[0]) || !finite(g.axes[1]) || !finite(g.axes[2]) ||
                    !std::isfinite(g.opacity) || g.opacity < 0 || g.opacity > 1)
                    throw std::invalid_argument("Invalid Gaussian in collision source");
                double area = 0;
                const auto t = measure(g, &area);
                if (t > 0 && g.opacity > 0) {
                    area_samples.push_back(float(std::log(area)));
                    thickness.push_back(std::log(t));
                    volume.push_back(float(std::log(std::abs(glm::determinant(glm::dmat3(g.axes)))) / 3.));
                }
            }
            // Far-out fences in log thickness and equivalent-volume radius,
            // with an eightfold minimum spread. Exceptionally thin sheets are
            // exempt from volume rejection. Small fixtures bypass these volume
            // statistics; a separate area fence below also handles thin splats.
            float fence = std::numeric_limits<float>::infinity();
            float volume_fence = fence, thin_guard = fence, area_fence = fence;
            if (thickness.size() >= 128) {
                auto quantile = [&](size_t i) {
                    std::nth_element(thickness.begin(), thickness.begin() + i, thickness.end());
                    return thickness[i];
                };
                const auto lower = quantile(thickness.size() / 4);
                const auto upper = quantile(3 * thickness.size() / 4);
                fence = upper + std::max(3 * (upper - lower), float(std::log(8.0)));
                thin_guard = lower - float(std::log(4.));
                std::nth_element(volume.begin(), volume.begin() + volume.size() / 4, volume.end());
                const auto volume_lower = volume[volume.size() / 4];
                std::nth_element(volume.begin(), volume.begin() + 3 * volume.size() / 4, volume.end());
                const auto volume_upper = volume[3 * volume.size() / 4];
                volume_fence = volume_upper + std::max(3 * (volume_upper - volume_lower), float(std::log(8.)));
            }
            if (area_samples.size() >= 16) {
                std::nth_element(area_samples.begin(), area_samples.begin() + area_samples.size() / 4, area_samples.end());
                const auto area_lower = area_samples[area_samples.size() / 4];
                std::nth_element(area_samples.begin(), area_samples.begin() + 3 * area_samples.size() / 4, area_samples.end());
                const auto area_upper = area_samples[3 * area_samples.size() / 4];
                area_fence = area_upper + std::max(3 * (area_upper - area_lower), float(std::log(64.)));
            }
            result.area_limit = settings.max_gaussian_area > 0 ? settings.max_gaussian_area : std::exp(double(area_fence)) * 9 * std::numbers::pi;
            size_t volume_outliers = 0;
            for (size_t i = 0; i < source.size(); ++i) {
                if ((i & 4095) == 0)
                    checkpoint(stop);
                double area = 0;
                const auto t = measure(source[i], &area);
                const auto log_t = t > 0 ? std::log(t) : -std::numeric_limits<float>::infinity();
                const auto log_v = t > 0 ? std::log(std::abs(glm::determinant(glm::dmat3(source[i].axes)))) / 3. : -std::numeric_limits<double>::infinity();
                if (area * 9 * std::numbers::pi > result.area_limit) {
                    result.oversized_indices.push_back(uint32_t(i));
                    continue;
                }
                if (log_t <= fence && !(log_v > volume_fence && log_t > thin_guard))
                    result.indices.push_back(uint32_t(i));
                else
                    ++volume_outliers;
            }
            result.note = std::format("Surface cleanup kept {} / {} splats: {} oversized-area and {} volume outliers excluded.",
                                      result.indices.size(), source.size(), result.oversized_indices.size(), volume_outliers);
            if (std::isfinite(result.area_limit))
                result.note += std::format(" Area limit: {:.4g} square scene units ({}).", result.area_limit, settings.max_gaussian_area > 0 ? "custom" : "automatic");
            if (progress)
                progress(settings.cluster ? .3f : 1.f, "Source cleanup: " + result.note);
            if (!settings.cluster || result.indices.empty()) {
                if (result.indices.size() == source.size())
                    result.note.clear();
                return result;
            }

            // Connectivity comes from observed Gaussian centres, not accumulated
            // alpha. A sparse sheet need not reach 99.9% opacity to be structural.
            // The graph bridges diagonal cell neighbours but does not dilate the
            // final collision grid or connect via a giant Gaussian's support box.
            using Key = std::array<int32_t, 3>;
            struct Cell {
                Key p;
                uint32_t parent, count;
            };
            std::unordered_map<Key, uint32_t, GridPointHash> lookup;
            lookup.reserve(std::min(result.indices.size(), size_t(1'000'000)));
            std::vector<Cell> cells;
            std::vector<uint32_t> memberships;
            memberships.reserve(result.indices.size());
            const auto size = double(settings.cluster_resolution);
            auto point = [&](glm::vec3 p) {
                const auto v = glm::floor(glm::dvec3(p) / size);
                // Leave room for neighbour offsets; no signed overflow or silent
                // coordinate clipping for remote components.
                for (int a = 0; a < 3; ++a)
                    if (v[a] < double(INT32_MIN) + 1 || v[a] > double(INT32_MAX) - 1)
                        throw std::runtime_error("Cluster coordinates exceed integer range; increase cluster resolution");
                return Key{int32_t(v.x), int32_t(v.y), int32_t(v.z)};
            };
            for (auto i : result.indices) {
                if ((memberships.size() & 4095) == 0)
                    checkpoint(stop);
                if (source[i].opacity == 0 || std::abs(glm::determinant(source[i].axes)) < 1e-20f) {
                    memberships.push_back(UINT32_MAX);
                    continue;
                }
                const auto p = point(source[i].center);
                const auto [it, added] = lookup.try_emplace(p, uint32_t(cells.size()));
                if (added)
                    cells.push_back({p, it->second, 0});
                ++cells[it->second].count;
                memberships.push_back(it->second);
            }
            if (cells.empty()) {
                result.indices.clear();
                result.note += " No contributing source cells were found.";
                return result;
            }
            auto root = [&](uint32_t i) {
                while (cells[i].parent != i) {
                    cells[i].parent = cells[cells[i].parent].parent;
                    i = cells[i].parent;
                }
                return i;
            };
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if ((i & 4095) == 0) {
                    checkpoint(stop);
                    if (progress)
                        progress(.3f + .6f * float(i) / cells.size(), "Connecting observed source cells on CPU");
                }
                const auto p = cells[i].p;
                for (int z = -1; z <= 1; ++z)
                    for (int y = -1; y <= 1; ++y)
                        for (int x = -1; x <= 1; ++x) {
                            // Only one half of the 26 undirected neighbours.
                            if (z < 0 || (z == 0 && y < 0) || (z == 0 && y == 0 && x <= 0))
                                continue;
                            const auto it = lookup.find({p[0] + x, p[1] + y, p[2] + z});
                            if (it == lookup.end())
                                continue;
                            auto a = root(i), b = root(it->second);
                            if (a == b)
                                continue;
                            if (cells[a].count < cells[b].count)
                                std::swap(a, b);
                            cells[b].parent = a;
                            cells[a].count += cells[b].count;
                        }
            }
            // Respect an explicit seed inside observed geometry. Otherwise prefer
            // substantial nearby support over one tiny floater nearest the camera.
            const auto seed = glm::dvec3(settings.seed) / size;
            double nearest = std::numeric_limits<double>::infinity();
            uint32_t selected = 0;
            const auto exact = lookup.find(point(settings.seed));
            if (exact != lookup.end())
                selected = root(exact->second);
            else {
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if ((i & 4095) == 0)
                        checkpoint(stop);
                    const auto p = cells[i].p;
                    const auto d = glm::max(glm::max(glm::dvec3(p[0], p[1], p[2]) - seed,
                                                     seed - glm::dvec3(p[0] + 1., p[1] + 1., p[2] + 1.)),
                                            glm::dvec3(0));
                    const auto distance = glm::dot(d, d);
                    if (distance < nearest || (distance == nearest && cells[root(i)].count > cells[root(selected)].count)) {
                        nearest = distance;
                        selected = root(i);
                    }
                }
                const auto radius2 = std::pow(std::sqrt(nearest) + 2., 2);
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if ((i & 4095) == 0)
                        checkpoint(stop);
                    const auto p = cells[i].p;
                    const auto d = glm::max(glm::max(glm::dvec3(p[0], p[1], p[2]) - seed,
                                                     seed - glm::dvec3(p[0] + 1., p[1] + 1., p[2] + 1.)),
                                            glm::dvec3(0));
                    if (glm::dot(d, d) <= radius2 && cells[root(i)].count > cells[selected].count)
                        selected = root(i);
                }
            }
            size_t kept = 0;
            for (size_t i = 0; i < memberships.size(); ++i) {
                if ((i & 4095) == 0)
                    checkpoint(stop);
                if (memberships[i] != UINT32_MAX && root(memberships[i]) == selected)
                    result.indices[kept++] = result.indices[i];
            }
            result.indices.resize(kept);
            result.note += std::format(" Connected source region kept {} / {} splats at {:.4g} scene-unit spacing; final voxel size is unchanged.", kept, source.size(), size);
            if (progress)
                progress(1, "Source filtering complete: " + result.note);
            return result;
        }

    } // namespace collision_detail

    CollisionVolume generateCollision(std::span<const CollisionGaussian> gaussians, CollisionSettings s,
                                      std::stop_token stop, CollisionProgress progress, CollisionVoxelizer backend) {
        s.validate();
        if (gaussians.empty())
            throw std::invalid_argument("No source Gaussians");
        if (gaussians.size() > CollisionGaussian::max_count)
            throw std::runtime_error("Collision source exceeds 32 million Gaussians");
        auto report = [&](float value, std::string text) { checkpoint(stop); if (progress) progress(value, std::move(text)); checkpoint(stop); };
        const float filtered_end = (s.cluster || s.adaptive_filter) ? .2f : 0.f;
        auto voxelize = [&](std::span<const CollisionGaussian> source, CollisionSettings options, CollisionProgress report) {
            return backend ? backend(source, options, stop, std::move(report)) : collision_detail::voxelizeSparse(source, options, stop, std::move(report));
        };
        std::vector<CollisionGaussian> filtered;
        std::string cluster_note;
        if (s.adaptive_filter || s.cluster) {
            report(0, "Finding reconstruction outliers on CPU");
            auto selection = collision_detail::selectAdaptiveSource(gaussians, s, stop, collisionProgressRange(progress, 0, filtered_end));
            cluster_note = std::move(selection.note);
            if (selection.indices.size() != gaussians.size()) {
                filtered.reserve(selection.indices.size());
                for (auto i : selection.indices)
                    filtered.push_back(gaussians[i]);
                gaussians = filtered;
            }
            if (gaussians.empty())
                throw std::runtime_error("Source cleanup excluded all splats; increase maximum splat area or choose Keep all splats");
        }
        if (s.auto_resolution) {
            s.voxel_size = collision_detail::recommendCollisionVoxelSize(gaussians, s.units_per_meter, stop);
            const auto note = std::format("Auto cell size: {:.4g} scene units.", s.voxel_size);
            if (!cluster_note.empty())
                cluster_note += " ";
            cluster_note += note;
            if (progress)
                progress(filtered_end, note);
        }
        report(filtered_end, "Preparing full-resolution collision cells");
        auto grid = voxelize(gaussians, s, collisionProgressRange(progress, filtered_end, 1.f));
        grid.validate();
        const bool occupied = grid.tree.empty() ? std::find(grid.solid.begin(), grid.solid.end(), 1) != grid.solid.end()
                                                : std::find(grid.tree.begin(), grid.tree.end(), 1u) != grid.tree.end();
        if (!occupied)
            throw std::runtime_error("No occupied voxels; lower opacity or voxel size");
        checkpoint(stop);
        if (!cluster_note.empty())
            grid.warning = cluster_note + (grid.warning.empty() ? "" : " " + grid.warning);
        return grid;
    }
    std::vector<uint32_t> CollisionVolume::octreeOccupancy() const {
        validate();
        return collision_detail::volumeTree(*this).nodes;
    }
    void CollisionVolume::forEachSolidBox(const std::function<void(glm::ivec3, glm::ivec3)>& fn) const {
        if (!tree.empty()) {
            std::function<void(uint32_t, glm::ivec3, int)> visit = [&](uint32_t i, glm::ivec3 lo, int n) {
                if (glm::any(glm::greaterThanEqual(lo, dimensions)) || tree[i] == 0)
                    return;
                if (tree[i] == 1) {
                    fn(lo, glm::min(lo + n, dimensions));
                    return;
                }
                for (int a = 0; a < 8; ++a)
                    visit(tree[i] - 2 + a, lo + glm::ivec3(a & 1, (a >> 1) & 1, a >> 2) * (n / 2), n / 2);
            };
            visit(0, {}, tree_size);
        } else {
            for (int z = 0; z < dimensions.z; ++z)
                for (int y = 0; y < dimensions.y; ++y)
                    for (int x = 0; x < dimensions.x;) {
                        if (!occupied({x, y, z})) {
                            ++x;
                            continue;
                        }
                        int start = x++;
                        while (x < dimensions.x && occupied({x, y, z}))
                            ++x;
                        fn({start, y, z}, {x, y + 1, z + 1});
                    }
        }
    }
    uint64_t CollisionVolume::occupiedCount() const {
        uint64_t count = 0;
        forEachSolidBox([&](auto lo, auto hi) {
            auto d = hi - lo;
            count += uint64_t(d.x) * d.y * d.z;
        });
        return count;
    }
} // namespace lfs::core
