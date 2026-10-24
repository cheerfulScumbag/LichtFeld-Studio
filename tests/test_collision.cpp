/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../src/core/collision_sparse.hpp"
#include "core/collision.hpp"
#include <algorithm>
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <map>
#include <random>

using namespace lfs::core;
TEST(CollisionProgress, NestedRangesPreserveFractionsAndStageLabels) {
    float value = -1;
    std::string stage;
    auto report = [&](float p, std::string text) { value = p; stage = std::move(text); };
    const auto nested = collisionProgressRange(collisionProgressRange(report, .2f, .8f), .25f, .75f);
    nested(0, "Preparing");
    EXPECT_FLOAT_EQ(value, .35f);
    nested(1, "Ready");
    EXPECT_FLOAT_EQ(value, .65f);
    EXPECT_EQ(stage, "Ready");
    EXPECT_FALSE(collisionProgressRange({}, 0, 1));
}
namespace {
    CollisionVolume grid(glm::ivec3 dimensions = {12, 12, 12}, float size = 0.1f) {
        CollisionVolume g;
        g.dimensions = dimensions;
        g.settings.voxel_size = size;
        g.solid.resize(size_t(dimensions.x) * dimensions.y * dimensions.z);
        return g;
    }
    void box(CollisionVolume& g, glm::ivec3 lo, glm::ivec3 hi, bool shell = false) {
        for (int z = lo.z; z <= hi.z; ++z)
            for (int y = lo.y; y <= hi.y; ++y)
                for (int x = lo.x; x <= hi.x; ++x)
                    if (!shell || x == lo.x || y == lo.y || z == lo.z || x == hi.x || y == hi.y || z == hi.z)
                        g.solid[g.index({x, y, z})] = 1;
    }
    size_t count(const CollisionVolume& g) { return g.occupiedCount(); }
    CollisionVolume sparse(CollisionVolume g) {
        g.tree = g.octreeOccupancy();
        g.tree_size = 1;
        while (g.tree_size < std::max({g.dimensions.x, g.dimensions.y, g.dimensions.z}))
            g.tree_size *= 2;
        g.solid.clear();
        g.validate();
        return g;
    }
    void sameOccupancy(const CollisionVolume& a, const CollisionVolume& b) {
        ASSERT_EQ(a.dimensions, b.dimensions);
        ASSERT_EQ(a.occupiedCount(), b.occupiedCount());
        for (int z = 0; z < a.dimensions.z; ++z)
            for (int y = 0; y < a.dimensions.y; ++y)
                for (int x = 0; x < a.dimensions.x; ++x)
                    ASSERT_EQ(a.occupied({x, y, z}), b.occupied({x, y, z})) << x << "," << y << "," << z;
    }
    auto recordProgress(std::vector<float>& values) {
        return [&values](float value, const std::string& stage) {
            EXPECT_GE(value, values.empty() ? 0 : values.back());
            EXPECT_LE(value, 1);
            EXPECT_FALSE(stage.empty());
            values.push_back(value);
        };
    }
} // namespace
TEST(CollisionProgress, GenerationReportsMonotonicFractionsAcrossFilterModes) {
    std::vector<CollisionGaussian> source;
    for (int i = 0; i < 96; ++i)
        source.push_back({{float(i % 8) * .04f, 0, float(i / 8) * .04f}, glm::mat3(.04f), .8f});
    source.push_back({{5, 0, 0}, glm::mat3(.05f), .7f});
    for (unsigned mode = 0; mode < 3; ++mode)
        for (bool automatic : {false, true}) {
            CollisionSettings s;
            s.adaptive_filter = mode == 1 || mode == 2;
            s.cluster = mode >= 2;
            s.cluster_resolution = .2f;
            s.voxel_size = .1f;
            s.auto_resolution = automatic;
            s.seed = {.1f, .1f, .1f};
            std::vector<float> values;
            const auto result = generateCollision(source, s, {}, recordProgress(values));
            ASSERT_FALSE(values.empty());
            EXPECT_FLOAT_EQ(values.front(), 0);
            EXPECT_FLOAT_EQ(values.back(), 1);
            EXPECT_GT(result.occupiedCount(), 0u);
        }
}
TEST(CollisionVoxelization, RotationScaleAndCombinedOpacity) {
    CollisionSettings s;
    s.voxel_size = 0.05f;
    s.opacity = 0.5f;
    CollisionGaussian a{{0, 0, 0}, glm::mat3(0.2f), 0.4f};
    EXPECT_THROW((void)generateCollision(std::span(&a, 1), s), std::runtime_error);
    // Density adds: two 0.4 contributions yield 1-exp(-0.8), above 0.5.
    std::array both{a, a};
    auto combined = generateCollision(both, s);
    EXPECT_GT(count(combined), 0u);
    a.opacity = 1;
    a.axes = glm::mat3(1);
    a.axes[0] *= 0.35f;
    a.axes[1] *= 0.03f;
    a.axes[2] *= 0.03f;
    s.opacity = 0.1f;
    auto horizontal = generateCollision(std::span(&a, 1), s);
    EXPECT_TRUE(horizontal.occupied(horizontal.cell({0.4f, 0, 0})));
    EXPECT_FALSE(horizontal.occupied(horizontal.cell({0, 0.4f, 0})));
    a.axes = glm::mat3(glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(0, 0, 1))) * a.axes;
    auto vertical = generateCollision(std::span(&a, 1), s);
    EXPECT_FALSE(vertical.occupied(vertical.cell({0.4f, 0, 0})));
    EXPECT_TRUE(vertical.occupied(vertical.cell({0, 0.4f, 0})));
}
TEST(CollisionPipeline, CancellationIsExplicit) {
    const CollisionGaussian g{{0, 0, 0}, glm::mat3(.1f), 1};
    std::stop_source source;
    source.request_stop();
    EXPECT_THROW((void)generateCollision({&g, 1}, {}, source.get_token()), std::runtime_error);
}
TEST(CollisionSettings, RejectsInvalidDomainsAndGridDimensions) {
    CollisionSettings s;
    for (float value : {-1.f, 0.f, std::numeric_limits<float>::infinity()}) {
        s.voxel_size = value;
        EXPECT_THROW(s.validate(), std::invalid_argument);
    }
    auto v = grid();
    v.dimensions = {16'000'000, 2, 1};
    EXPECT_THROW(v.validate(), std::runtime_error);
    v = grid();
    v.solid[0] = 2;
    EXPECT_THROW(v.validate(), std::invalid_argument);
}
TEST(CollisionSparse, DistantGaussianComponentsDoNotAllocateEmptyWorldBounds) {
    CollisionSettings s;
    s.voxel_size = 0.05f;
    const std::array source{CollisionGaussian{{0, 0, 0}, glm::mat3(0.1f), 1},
                            CollisionGaussian{{40'000, 40'000, 40'000}, glm::mat3(0.1f), 1}};
    auto g = generateCollision(source, s);
    EXPECT_TRUE(g.solid.empty());
    EXPECT_GT(uint64_t(g.dimensions.x) * g.dimensions.y * g.dimensions.z, 1'000'000'000'000ULL);
    EXPECT_LT(g.tree.size(), 5'000u);
    EXPECT_TRUE(g.occupied(g.cell(source.front().center)));
    EXPECT_TRUE(g.occupied(g.cell(source.back().center)));
    EXPECT_FALSE(g.occupied(g.cell({20'000, 20'000, 20'000})));
}
TEST(CollisionSparse, AccumulatedOpacityMatchesDirectCellEvaluation) {
    CollisionSettings s;
    s.voxel_size = 0.1f;
    s.opacity = 0.3f;
    std::mt19937 rng(19);
    std::uniform_real_distribution<float> position(-1, 1), scale(0.02f, 0.5f), alpha(0.005f, 0.8f);
    std::vector<CollisionGaussian> source;
    for (int i = 0; i < 50; ++i) {
        auto axes = glm::mat3(glm::rotate(glm::mat4(1), position(rng) * 3, glm::normalize(glm::vec3(1, 2, 3))));
        for (int a = 0; a < 3; ++a)
            axes[a] *= scale(rng);
        source.push_back({{position(rng), position(rng), position(rng)}, axes, alpha(rng)});
    }
    auto g = generateCollision(source, s);
    for (int z = 0; z < g.dimensions.z; ++z)
        for (int y = 0; y < g.dimensions.y; ++y)
            for (int x = 0; x < g.dimensions.x; ++x) {
                glm::ivec3 p(x, y, z);
                float a = 0;
                for (const auto& gaussian : source) {
                    const auto center = g.origin + (glm::vec3(p) + 0.5f) * s.voxel_size;
                    glm::vec3 extent;
                    for (int axis = 0; axis < 3; ++axis)
                        extent[axis] = 3 * glm::length(glm::vec3(gaussian.axes[0][axis], gaussian.axes[1][axis], gaussian.axes[2][axis]));
                    if (glm::any(glm::greaterThan(glm::abs(center - gaussian.center), extent + s.voxel_size * 0.5f)))
                        continue;
                    const auto closest = glm::clamp(gaussian.center, center - s.voxel_size * 0.5f, center + s.voxel_size * 0.5f);
                    const auto q = glm::inverse(gaussian.axes) * (closest - gaussian.center);
                    a += gaussian.opacity * std::exp(-0.5f * glm::dot(q, q));
                }
                ASSERT_EQ(g.occupied(p), a > 0 && -std::expm1(-a) >= s.opacity) << x << "," << y << "," << z;
            }
}
TEST(CollisionSparse, WeakIsolatedAndDegenerateFarSupportsDoNotExpandBounds) {
    CollisionSettings s;
    const std::array source{
        CollisionGaussian{{0, 0, 0}, glm::mat3(0.1f), 1},
        CollisionGaussian{{1e9f, 1e9f, 1e9f}, glm::mat3(1), 0.01f},
        CollisionGaussian{{-1e9f, -1e9f, -1e9f}, glm::mat3(0), 1}};
    const auto full = generateCollision(source, s);
    const auto foreground = generateCollision(std::span(source.data(), 1), s);
    sameOccupancy(full, foreground);
    EXPECT_EQ(full.origin, foreground.origin);
    // The proof must not discard sub-threshold supports which combine.
    const std::array combined{
        CollisionGaussian{{0, 0, 0}, glm::mat3(0.1f), 0.06f},
        CollisionGaussian{{0, 0, 0}, glm::mat3(0.1f), 0.06f}};
    EXPECT_GT(generateCollision(combined, s).occupiedCount(), 0u);
}
TEST(CollisionSparse, ConservativeBoundsPreserveCombinedWeakSupportsAndEmptyRegions) {
    CollisionSettings s;
    s.voxel_size = .1f;
    s.opacity = .1f;
    const std::array source{
        CollisionGaussian{{0, 0, 0}, glm::mat3(.1f), .06f},
        CollisionGaussian{{0, 0, 0}, glm::mat3(.1f), .06f},
        CollisionGaussian{{10000, 0, 0}, glm::mat3(.1f), .01f}};
    const auto volume = generateCollision(source, s);
    EXPECT_TRUE(volume.occupied(volume.cell({0, 0, 0})));
    EXPECT_FALSE(volume.occupied(volume.cell({10000, 0, 0})));
    EXPECT_FALSE(volume.occupied(volume.cell({5000, 0, 0})));
    EXPECT_LT(volume.tree.size(), 10000u);
    const auto foreground = generateCollision(std::span(source.data(), 2), s);
    EXPECT_EQ(volume.occupiedCount(), foreground.occupiedCount());
}
TEST(CollisionSparse, PointLookupPreservesPartialDimensionsAndSubtreeCoordinates) {
    auto dense = grid({9, 7, 11});
    for (size_t i = 0; i < dense.solid.size(); ++i)
        dense.solid[i] = i % 3 == 0;
    const auto v = sparse(dense);
    collision_detail::Tree tree(v.dimensions);
    tree.nodes = v.tree;
    for (int z = 0; z < dense.dimensions.z; ++z)
        for (int y = 0; y < dense.dimensions.y; ++y)
            for (int x = 0; x < dense.dimensions.x; ++x) {
                const glm::ivec3 p(x, y, z), bit = p / (tree.size / 2), offset = bit * (tree.size / 2);
                const auto leaf = tree.leaf(p);
                const auto child = tree.nodes[0] - 2 + bit.x + 2 * bit.y + 4 * bit.z;
                const auto local = collision_detail::lookupLeaf(tree.nodes, p - offset, tree.size / 2, child);
                EXPECT_EQ(leaf.solid, dense.occupied(p));
                EXPECT_EQ(v.occupied(p), dense.occupied(p));
                EXPECT_EQ(leaf.min, local.min + offset);
                EXPECT_EQ(leaf.size, local.size);
                EXPECT_EQ(leaf.index, local.index);
                EXPECT_EQ(leaf.solid, local.solid);
            }
    EXPECT_FALSE(v.occupied({9, 0, 0}));
    EXPECT_FALSE(v.occupied({0, -1, 0}));
}
TEST(CollisionVoxelization, ParallelBricksMatchSerialIncludingPartialBoundaryCells) {
    std::vector<CollisionGaussian> source;
    for (int i = 0; i < 4096; ++i)
        source.push_back({{float(i % 16) * 0.12f, float((i / 16) % 16) * 0.12f, float(i / 256) * 0.12f}, glm::mat3(0.035f), 0.8f});
    CollisionSettings settings;
    settings.voxel_size = 0.07f;
    const auto serial = collision_detail::voxelizeSparse(source, settings, {}, {}, 1);
    std::vector<float> values;
    const auto parallel = collision_detail::voxelizeSparse(source, settings, {}, recordProgress(values), 4);
    ASSERT_FALSE(values.empty());
    EXPECT_FLOAT_EQ(values.front(), 0);
    EXPECT_FLOAT_EQ(values.back(), 1);
    ASSERT_EQ(serial.origin, parallel.origin);
    sameOccupancy(serial, parallel);
    try {
        (void)collision_detail::voxelizeSparse(source, settings, {}, [](float value, auto) { if (value > 0) throw std::runtime_error("worker callback failure"); }, 4);
        FAIL() << "Expected worker callback failure";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "worker callback failure");
    }
    std::stop_source stop;
    EXPECT_THROW((void)collision_detail::voxelizeSparse(source, settings, stop.get_token(), [&](float value, auto) { if (value > .1f && value < 1) stop.request_stop(); }, 4), std::runtime_error);
    EXPECT_TRUE(stop.stop_requested());
    sameOccupancy(serial, collision_detail::voxelizeSparse(source, settings, {}, {}, 4));
}
TEST(CollisionVoxelization, ParallelBoundsKeepOverlappingWeakSupportAndRejectDistantOutliers) {
    CollisionSettings settings;
    settings.voxel_size = .1f;
    std::vector<CollisionGaussian> source(4096, {{0, 0, 0}, glm::mat3(.02f), .001f});
    const auto reference = collision_detail::prepareCollision(source, settings, {});
    source.push_back({{1e8f, 0, 0}, glm::mat3(.02f), .001f});
    const auto result = collision_detail::prepareCollision(source, settings, {});
    EXPECT_EQ(result.volume.origin, reference.volume.origin);
    EXPECT_EQ(result.volume.dimensions, reference.volume.dimensions);
    EXPECT_EQ(result.gaussians.size(), reference.gaussians.size());
}
TEST(CollisionAdaptive, RemovesExceptionalVolumeAndAreaWhilePreservingRemoteSurfaces) {
    std::vector<CollisionGaussian> source;
    for (int i = 0; i < 256; ++i)
        source.push_back({{i * .1f, 0, 0}, glm::mat3(.1f), .5f});
    source.push_back({{0, 0, 0}, glm::mat3(100), .9f});
    source.push_back({{0, 0, 0}, glm::mat3(100, 0, 0, 0, .001f, 0, 0, 0, 100), .9f});
    source.push_back({{100000, 0, 0}, glm::mat3(.1f), .5f});
    CollisionSettings s;
    const auto a = collision_detail::selectAdaptiveSource(source, s);
    EXPECT_EQ(a.indices.size(), source.size() - 2);
    EXPECT_EQ(a.oversized_indices, (std::vector<uint32_t>{256, 257}));
    EXPECT_EQ(a.indices.back(), 258u);
    for (auto& g : source) {
        g.center *= .001f;
        g.axes *= .001f;
    }
    s.units_per_meter *= .001f;
    s.cluster_resolution *= .001f;
    EXPECT_EQ(collision_detail::selectAdaptiveSource(source, s).indices, a.indices);
}
TEST(CollisionAdaptive, SparseDiagonalObservationsConnectWithoutChangingCollisionOccupancy) {
    std::vector<CollisionGaussian> source;
    for (int i = 0; i < 20; ++i)
        source.push_back({{i + .5f, i + .5f, 0}, glm::mat3(.01f), .2f});
    source.push_back({{100, 0, 0}, glm::mat3(.01f), .2f});
    CollisionSettings s;
    s.cluster = true;
    s.seed = source.front().center;
    auto selection = collision_detail::selectAdaptiveSource(source, s);
    ASSERT_EQ(selection.indices.size(), 20u);
    auto volume = generateCollision(source, s);
    for (int i = 0; i < 20; ++i)
        EXPECT_TRUE(volume.occupied(volume.cell(source[i].center)));
    EXPECT_FALSE(volume.occupied(volume.cell({1, 1, 0})));
    EXPECT_FALSE(volume.occupied(volume.cell(source.back().center)));
}
TEST(CollisionAdaptive, EmptySeedPrefersNearbySupportButExplicitSmallRegionIsRespected) {
    std::vector<CollisionGaussian> source;
    source.push_back({{.2f, .2f, .2f}, glm::mat3(.01f), .5f});
    for (int i = 0; i < 100; ++i)
        source.push_back({{2.1f + i * .01f, .2f, .2f}, glm::mat3(.01f), .5f});
    CollisionSettings s;
    s.cluster = true;
    s.seed = {-.1f, .2f, .2f};
    EXPECT_EQ(collision_detail::selectAdaptiveSource(source, s).indices.size(), 100u);
    s.seed = source.front().center;
    EXPECT_EQ(collision_detail::selectAdaptiveSource(source, s).indices, std::vector<uint32_t>{0});
    std::stop_source stop;
    stop.request_stop();
    EXPECT_THROW((void)collision_detail::selectAdaptiveSource(source, s, stop.get_token()), std::runtime_error);
    source[0].center.x = std::numeric_limits<float>::infinity();
    EXPECT_THROW((void)collision_detail::selectAdaptiveSource(source, s), std::invalid_argument);
}
TEST(CollisionAdaptive, AutomaticResolutionScalesWithObservedFootprintAndPreservesCustomValues) {
    std::vector<CollisionGaussian> source(10000, {{0, 0, 0}, glm::mat3(1), .5f});
    const auto edge = collision_detail::recommendCollisionVoxelSize(source, 1);
    EXPECT_FLOAT_EQ(edge, .07f);
    for (auto& g : source)
        g.axes *= .1f;
    EXPECT_NEAR(collision_detail::recommendCollisionVoxelSize(source, .1f), edge * .1f, 1e-6);
    // Empty world distance has no influence on the recommendation.
    source.back().center = {1e8f, -1e8f, 1e8f};
    EXPECT_NEAR(collision_detail::recommendCollisionVoxelSize(source, .1f), edge * .1f, 1e-6);
    CollisionSettings s;
    s.voxel_size = .03f;
    const CollisionGaussian g{{0, 0, 0}, glm::mat3(.01f), .9f};
    EXPECT_FLOAT_EQ(generateCollision(std::span(&g, 1), s).settings.voxel_size, .03f);
    s.auto_resolution = true;
    EXPECT_FLOAT_EQ(generateCollision(std::span(&g, 1), s).settings.voxel_size, .05f);
}
TEST(CollisionSparse, CanonicalValidationRejectsAliasesUnreachableBlocksAndExcessDepth) {
    CollisionVolume v;
    v.dimensions = {4, 4, 4};
    v.tree_size = 4;
    v.tree.resize(25, 0);
    v.tree[0] = 3;
    v.tree[1] = 11;
    v.tree[2] = 19;
    EXPECT_NO_THROW(v.validate());
    auto bad = v;
    bad.tree[2] = 11;
    EXPECT_THROW(bad.validate(), std::invalid_argument);
    bad = v;
    bad.tree[2] = 0;
    EXPECT_THROW(bad.validate(), std::invalid_argument);
    bad = v;
    bad.tree[2] = 18;
    EXPECT_THROW(bad.validate(), std::invalid_argument);
    bad = v;
    bad.dimensions = {2, 2, 2};
    bad.tree_size = 2;
    EXPECT_THROW(bad.validate(), std::invalid_argument);
    bad = v;
    bad.tree[0] = 2;
    EXPECT_THROW(bad.validate(), std::invalid_argument);
}
TEST(CollisionAdaptive, TransparentAndDegenerateSplatsCannotBridgeSourceRegions) {
    std::vector<CollisionGaussian> source{{{.1f, .1f, .1f}, glm::mat3(.1f), .9f},
                                          {{1.1f, .1f, .1f}, glm::mat3(.1f), 0},
                                          {{2.1f, .1f, .1f}, glm::mat3(.1f), .9f},
                                          {{1.1f, .1f, .1f}, glm::mat3(0), .9f}};
    CollisionSettings s;
    s.cluster = true;
    s.seed = source[0].center;
    EXPECT_EQ(collision_detail::selectAdaptiveSource(source, s).indices, std::vector<uint32_t>{0});
}
TEST(CollisionAdaptive, AreaLimitFindsThinBroadSplatsWithoutMovingOrDeletingSource) {
    std::vector<CollisionGaussian> source(32, {{0, 0, 0}, glm::mat3(.1f), .5f});
    source.push_back({{4, 2, 8}, glm::mat3(10, 0, 0, 0, .000001f, 0, 0, 0, 10), .01f});
    CollisionSettings s;
    const auto selected = collision_detail::selectAdaptiveSource(source, s);
    ASSERT_EQ(selected.oversized_indices, std::vector<uint32_t>{32});
    EXPECT_EQ(selected.indices.size(), 32);
    EXPECT_TRUE(std::isfinite(selected.area_limit));
    EXPECT_LT(selected.area_limit, 3000);
    EXPECT_EQ(source.back().center, glm::vec3(4, 2, 8));
    EXPECT_EQ(source.size(), 33);
    const auto rotation = glm::mat3(glm::rotate(glm::mat4(1), .73f, glm::normalize(glm::vec3(1, 2, 3))));
    for (auto& g : source) {
        g.center = rotation * g.center;
        g.axes = rotation * g.axes;
    }
    const auto rotated = collision_detail::selectAdaptiveSource(source, s);
    EXPECT_EQ(rotated.oversized_indices, selected.oversized_indices);
    EXPECT_NEAR(rotated.area_limit, selected.area_limit, 1e-3);
    for (auto& g : source) {
        g.center *= .01f;
        g.axes *= .01f;
    }
    s.units_per_meter = .01f;
    const auto scaled = collision_detail::selectAdaptiveSource(source, s);
    EXPECT_EQ(scaled.oversized_indices, selected.oversized_indices);
    EXPECT_NEAR(scaled.area_limit, selected.area_limit * .0001, 1e-6);
}
TEST(CollisionAdaptive, CustomAreaCapWorksOnSmallSourcesAndKeepAllBypassesIt) {
    std::array source{CollisionGaussian{{0, 0, 0}, glm::mat3(.01f), .9f},
                      CollisionGaussian{{2, 0, 0}, glm::mat3(.2f, 0, 0, 0, .0001f, 0, 0, 0, .2f), .9f}};
    CollisionSettings s;
    s.max_gaussian_area = .1f;
    const auto selected = collision_detail::selectAdaptiveSource(source, s);
    EXPECT_EQ(selected.oversized_indices, std::vector<uint32_t>{1});
    auto cleaned = generateCollision(source, s);
    EXPECT_TRUE(cleaned.occupied(cleaned.cell(source[0].center)));
    EXPECT_FALSE(cleaned.occupied(cleaned.cell(source[1].center)));
    s.adaptive_filter = false;
    auto all = generateCollision(source, s);
    EXPECT_TRUE(all.occupied(all.cell(source[1].center)));
    s.adaptive_filter = true;
    s.max_gaussian_area = .000001f;
    EXPECT_THROW((void)generateCollision(source, s), std::runtime_error);
    s.max_gaussian_area = -1;
    EXPECT_THROW(s.validate(), std::invalid_argument);
}
