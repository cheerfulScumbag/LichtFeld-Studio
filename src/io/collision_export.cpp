/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "io/collision_export.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "core/uuid.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace lfs::io {
    namespace {
        using Json = nlohmann::json;
        constexpr uint32_t max_nodes = 1u << 24;
        void check(std::stop_token stop) {
            if (stop.stop_requested())
                throw std::runtime_error("Collision export cancelled");
        }
        glm::mat4 playcanvas(const glm::mat4& world) {
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 4; ++r)
                    if (!std::isfinite(world[c][r]))
                        throw std::runtime_error("Collision transform contains non-finite values");
            glm::mat4 flip(1);
            flip[0][0] = flip[1][1] = -1;
            return flip * world;
        }
        Json xyz(glm::vec3 p) { return Json::array({p.x, p.y, p.z}); }
        void word(std::ostream& out, uint32_t v) {
            const std::array<char, 4> bytes{char(v), char(v >> 8), char(v >> 16), char(v >> 24)};
            out.write(bytes.data(), bytes.size());
        }
        void words(std::ostream& out, std::span<const uint32_t> values, std::stop_token stop) {
            for (size_t i = 0; i < values.size(); i += 16384) {
                check(stop);
                const auto count = std::min(size_t(16384), values.size() - i);
                if constexpr (std::endian::native == std::endian::little)
                    out.write(reinterpret_cast<const char*>(values.data() + i), count * 4);
                else
                    for (size_t k = i; k < i + count; ++k)
                        word(out, values[k]);
            }
        }
        std::ofstream output(const std::filesystem::path& path) {
            std::ofstream out;
            out.exceptions(std::ios::badbit | std::ios::failbit);
            out.open(core::native_fs::native_path(path), std::ios::binary | std::ios::trunc);
            return out;
        }
        // Writers prepare sibling temporary files, so cancellation never exposes
        // a partially written payload. The metadata is published last.
        struct Temporary {
            std::filesystem::path path;
            explicit Temporary(const std::filesystem::path& target) : path(target) { path += "." + core::generate_uuid_v4().to_string() + ".tmp"; }
            ~Temporary() {
                std::error_code ec;
                core::native_fs::remove(path, ec);
            }
        };
        void publish(std::initializer_list<std::pair<Temporary*, std::filesystem::path>> files) {
            struct Entry {
                Temporary* temp;
                std::filesystem::path target, backup;
                bool backed_up = false, installed = false;
            };
            std::vector<Entry> entries;
            for (const auto& [temp, target] : files) {
                if (core::native_fs::exists(target) && !core::native_fs::is_regular_file(target))
                    throw std::runtime_error("Collision export destination is not a regular file: " + core::path_to_utf8(target));
                auto backup = temp->path;
                backup += ".backup";
                entries.push_back({temp, target, backup});
            }
            try {
                for (auto& e : entries)
                    if (core::native_fs::exists(e.target)) {
                        core::native_fs::rename(e.target, e.backup);
                        e.backed_up = true;
                    }
                for (auto& e : entries) {
                    core::native_fs::rename(e.temp->path, e.target);
                    e.installed = true;
                }
            } catch (...) {
                const auto failure = std::current_exception();
                std::error_code ec;
                for (auto& e : entries) {
                    if (e.installed)
                        core::native_fs::remove(e.target, ec);
                    if (e.backed_up) {
                        core::native_fs::rename(e.backup, e.target, ec);
                        if (ec)
                            throw std::runtime_error("Collision export failed; previous file retained at " + core::path_to_utf8(e.backup) + ": " + ec.message());
                    }
                }
                std::rethrow_exception(failure);
            }
            for (auto& e : entries)
                if (e.backed_up) {
                    std::error_code ec;
                    core::native_fs::remove(e.backup, ec);
                }
        }
    } // namespace

    void exportCollisionVoxels(const core::CollisionVolume& v, const glm::mat4& data_world,
                               const std::filesystem::path& path, std::stop_token stop, core::CollisionProgress progress) {
        check(stop);
        v.validate();
        if (progress)
            progress(0, "Preparing SuperSplat export");
        if (!core::path_to_utf8(path.filename()).ends_with(".voxel.json"))
            throw std::runtime_error("SuperSplat collision metadata must end in .voxel.json");
        auto binary = path;
        binary.replace_extension(".bin");
        const auto transform = playcanvas(data_world);
        const float scale = glm::length(glm::vec3(transform[0]));
        std::array<int, 3> axis{}, sign{};
        unsigned used = 0;
        for (int src = 0; src < 3; ++src) {
            int dst = 0;
            for (int k = 1; k < 3; ++k)
                if (std::abs(transform[src][k]) > std::abs(transform[src][dst]))
                    dst = k;
            if (!(scale > 0) || (used & (1u << dst)) || std::abs(std::abs(transform[src][dst]) - scale) > scale * 1e-5f)
                throw std::runtime_error("SuperSplat voxels require an axis-aligned, uniformly scaled grid. Align the scene and regenerate collision.");
            for (int k = 0; k < 3; ++k)
                if (k != dst && std::abs(transform[src][k]) > scale * 1e-5f)
                    throw std::runtime_error("SuperSplat voxels cannot store a rotated grid. Align the scene and regenerate collision.");
            axis[src] = dst;
            sign[src] = transform[src][dst] > 0 ? 1 : -1;
            used |= 1u << dst;
        }
        if (transform[0][3] != 0 || transform[1][3] != 0 || transform[2][3] != 0 || transform[3][3] != 1)
            throw std::runtime_error("Collision export requires an affine transform");
        int edge = 8, depth = 1;
        while (edge < std::max({v.dimensions.x, v.dimensions.y, v.dimensions.z})) {
            edge *= 2;
            ++depth;
        }
        const auto owned = v.tree.empty() ? v.octreeOccupancy() : std::vector<uint32_t>{};
        const auto& source = v.tree.empty() ? owned : v.tree;
        auto remap = [&](glm::ivec3 p, int n) {
            glm::ivec3 q;
            for (int s = 0; s < 3; ++s)
                q[axis[s]] = sign[s] > 0 ? p[s] : n - 1 - p[s];
            return q;
        };
        std::vector<uint32_t> nodes(1), leaves;
        struct Pending {
            uint32_t source, dest;
            int edge;
            glm::ivec3 min;
        };
        std::vector<Pending> level{{0, 0, edge, {0, 0, 0}}}, next;
        uint32_t interiors = 0, mixed = 0, levels = 0;
        const bool small = std::max({v.dimensions.x, v.dimensions.y, v.dimensions.z}) <= 4;
        if (!v.occupiedCount())
            level.clear(), nodes.clear();
        while (!level.empty()) {
            if (progress)
                progress(.7f * levels++ / (depth + 1), "Packing SuperSplat octree: " + std::to_string(nodes.size()) + " nodes");
            next.clear();
            for (const auto& p : level) {
                check(stop);
                const auto value = small ? 2u : source[p.source];
                if (value == 1) {
                    nodes[p.dest] = 0xff000000u;
                    continue;
                }
                if (p.edge == 4) {
                    uint64_t mask = 0;
                    for (int z = 0; z < 4; ++z)
                        for (int y = 0; y < 4; ++y)
                            for (int x = 0; x < 4; ++x) {
                                const glm::ivec3 cell(x, y, z);
                                if (small ? v.occupied(p.min + cell) : core::collision_detail::lookupLeaf(source, cell, 4, p.source).solid) {
                                    auto q = remap(cell, 4);
                                    mask |= uint64_t(1) << (q.x + 4 * q.y + 16 * q.z);
                                }
                            }
                    if (mask == ~uint64_t(0)) {
                        nodes[p.dest] = 0xff000000u;
                        continue;
                    }
                    if (mixed >= max_nodes)
                        throw std::runtime_error("SuperSplat mixed-leaf limit exceeded; increase collision cell size and regenerate");
                    nodes[p.dest] = mixed++;
                    leaves.push_back(uint32_t(mask));
                    leaves.push_back(uint32_t(mask >> 32));
                    continue;
                }
                std::array<uint32_t, 8> children{};
                std::array<glm::ivec3, 8> minima{};
                uint32_t mask = 0;
                for (int oct = 0; oct < 8; ++oct) {
                    auto bits = glm::ivec3(oct & 1, (oct >> 1) & 1, oct >> 2);
                    auto q = remap(bits, 2);
                    const int dst = q.x + 2 * q.y + 4 * q.z;
                    const auto child = small ? 0u : value - 2 + oct;
                    auto min = p.min + bits * (p.edge / 2);
                    bool present = !small && source[child] != 0;
                    if (small)
                        for (int z = 0; z < 4; ++z)
                            for (int y = 0; y < 4; ++y)
                                for (int x = 0; x < 4; ++x)
                                    present |= v.occupied(min + glm::ivec3(x, y, z));
                    if (present) {
                        mask |= 1u << dst;
                        children[dst] = child;
                        minima[dst] = min;
                    }
                }
                const auto base = uint32_t(nodes.size());
                if (nodes.size() + std::popcount(mask) > max_nodes || base >= max_nodes)
                    throw std::runtime_error("SuperSplat 24-bit octree limit exceeded; increase collision cell size and regenerate");
                nodes[p.dest] = (mask << 24) | base;
                ++interiors;
                for (int oct = 0; oct < 8; ++oct)
                    if (mask & (1u << oct)) {
                        next.push_back({children[oct], uint32_t(nodes.size()), p.edge / 2, minima[oct]});
                        nodes.push_back(0);
                    }
            }
            level.swap(next);
        }
        glm::vec3 lo(std::numeric_limits<float>::max()), hi(-lo), scene_lo(lo), scene_hi(-lo);
        for (int oct = 0; oct < 8; ++oct) {
            const glm::vec3 bits(oct & 1, (oct >> 1) & 1, oct >> 2);
            const auto p = glm::vec3(transform * glm::vec4(v.origin + bits * float(edge) * v.settings.voxel_size, 1));
            lo = glm::min(lo, p);
            hi = glm::max(hi, p);
            const auto s = glm::vec3(transform * glm::vec4(v.origin + bits * glm::vec3(v.dimensions) * v.settings.voxel_size, 1));
            scene_lo = glm::min(scene_lo, s);
            scene_hi = glm::max(scene_hi, s);
        }
        Json meta{{"version", "1.1"}, {"asset", {{"generator", "LichtFeld Studio"}}}, {"gridBounds", {{"min", xyz(lo)}, {"max", xyz(hi)}}}, {"sceneBounds", {{"min", xyz(scene_lo)}, {"max", xyz(scene_hi)}}}, {"voxelResolution", scale * v.settings.voxel_size}, {"leafSize", 4}, {"treeDepth", depth}, {"numInteriorNodes", interiors}, {"numMixedLeaves", mixed}, {"nodeCount", nodes.size()}, {"leafDataCount", leaves.size()}};
        Temporary bin_tmp(binary), json_tmp(path);
        if (progress)
            progress(.8f, "Writing SuperSplat voxel pair");
        {
            auto out = output(bin_tmp.path);
            words(out, nodes, stop);
            words(out, leaves, stop);
            out.close();
            auto header = output(json_tmp.path);
            header << meta.dump(2) << '\n';
            header.close();
        }
        check(stop);
        publish({{&bin_tmp, binary}, {&json_tmp, path}});
        if (progress)
            progress(1, "SuperSplat export complete");
    }

} // namespace lfs::io
