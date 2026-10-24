/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/collision.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <vector>

namespace lfs::core::collision_detail {
    // 0/1 are uniform empty/solid; other words point to eight contiguous children.
    class Tree {
        std::vector<uint32_t> free_blocks_;
        uint64_t work_ = 0;
        std::stop_token stop_;

        void tick() {
            if ((++work_ & 4095) == 0 && stop_.stop_requested())
                throw std::runtime_error("Collision generation cancelled");
            if (work_ > 4'000'000'000ULL)
                throw std::runtime_error("Sparse collision operation work budget exceeded");
        }
        static bool overlap(glm::ivec3 lo, glm::ivec3 hi, glm::ivec3 min, int n) {
            return glm::all(glm::lessThan(lo, min + n)) && glm::all(glm::greaterThan(hi, min));
        }
        uint32_t split(uint32_t index) {
            const auto value = nodes[index];
            if (value > 1)
                return value - 2;
            uint32_t base;
            if (free_blocks_.empty()) {
                if (nodes.size() + 8 > max_nodes)
                    throw std::runtime_error("Sparse collision node memory budget exceeded");
                base = uint32_t(nodes.size());
                nodes.resize(nodes.size() + 8);
            } else {
                base = free_blocks_.back();
                free_blocks_.pop_back();
            }
            std::fill_n(nodes.begin() + base, 8, value);
            nodes[index] = base + 2;
            return base;
        }
        void release(uint32_t index) {
            const auto value = nodes[index];
            if (value <= 1)
                return;
            const auto base = value - 2;
            for (int i = 0; i < 8; ++i)
                release(base + i);
            free_blocks_.push_back(base);
        }
        void set(uint32_t index, glm::ivec3 min, int n, glm::ivec3 lo, glm::ivec3 hi, bool value) {
            tick();
            if (!overlap(lo, hi, min, n) || nodes[index] == uint32_t(value))
                return;
            if (glm::all(glm::lessThanEqual(lo, min)) && glm::all(glm::greaterThanEqual(hi, min + n))) {
                release(index);
                nodes[index] = value;
                return;
            }
            const auto base = split(index);
            const int half = n / 2;
            for (int i = 0; i < 8; ++i)
                set(base + i, min + glm::ivec3(i & 1, (i >> 1) & 1, i >> 2) * half,
                    half, lo, hi, value);
            const auto first = nodes[base];
            if (first <= 1 && std::all_of(nodes.begin() + base, nodes.begin() + base + 8,
                                          [first](auto v) { return v == first; })) {
                nodes[index] = first;
                free_blocks_.push_back(base);
            }
        }

    public:
        static constexpr size_t max_nodes = CollisionVolume::max_tree_words;
        int size = 1;
        glm::ivec3 dimensions{1};
        std::vector<uint32_t> nodes{0};
        explicit Tree(glm::ivec3 dims, std::stop_token stop = {})
            : stop_(stop), dimensions(dims) {
            for (int axis = 0; axis < 3; ++axis)
                if (dims[axis] <= 0 || dims[axis] > (1 << 21))
                    throw std::invalid_argument("Sparse collision axis exceeds supported integer precision");
            while (size < std::max({dims.x, dims.y, dims.z}))
                size *= 2;
        }
        Leaf leaf(glm::ivec3 p) const { return lookupLeaf(nodes, p, size); }
        void setBox(glm::ivec3 lo, glm::ivec3 hi, bool value = true) {
            lo = glm::max(lo, glm::ivec3(0));
            hi = glm::min(hi, dimensions);
            if (glm::all(glm::lessThan(lo, hi)))
                set(0, {}, size, lo, hi, value);
        }
        void compact() {
            auto old = std::move(nodes);
            nodes.assign(1, 0);
            std::function<void(uint32_t, uint32_t)> copy = [&](uint32_t src, uint32_t dst) {
                if (old[src] <= 1) {
                    nodes[dst] = old[src];
                    return;
                }
                auto base = uint32_t(nodes.size());
                nodes.resize(nodes.size() + 8);
                nodes[dst] = base + 2;
                for (int i = 0; i < 8; ++i)
                    copy(old[src] - 2 + i, base + i);
            };
            copy(0, 0);
            free_blocks_.clear();
        }
    };
} // namespace lfs::core::collision_detail
