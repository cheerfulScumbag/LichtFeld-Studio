/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/voxelizer.hpp"
#include "core/argument_parser.hpp"
#include "core/collision_source.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "core/uuid.hpp"
#include "io/exporter.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace {
    using namespace lfs::core;
    class VoxelizeCli : public testing::Test {
    protected:
        const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("lfs-voxelize-" + generate_uuid_v4().to_string());
        void SetUp() override {
            std::filesystem::create_directories(dir);
            std::ofstream(dir / "input.ply") << "placeholder";
        }
        void TearDown() override { std::filesystem::remove_all(dir); }
        auto parse(std::vector<std::string> values) {
            values.insert(values.begin(), {"LichtFeld-Studio", "voxelize", path_to_utf8(dir / "input.ply")});
            std::vector<const char*> args;
            for (const auto& value : values)
                args.push_back(value.c_str());
            return args::parse_args(int(args.size()), args.data());
        }
        args::VoxelizeMode options() {
            return {.input_path = dir / "input.ply", .output_path = dir / "output.voxel.json", .overwrite = true};
        }
    };
    TEST_F(VoxelizeCli, DefaultsAndExplicitControlsFollowTheExistingCommandPattern) {
        auto result = parse({"-o", path_to_utf8(dir / "out.voxel.json")});
        ASSERT_TRUE(result) << result.error();
        const auto& mode = std::get<args::VoxelizeMode>(*result);
        EXPECT_FLOAT_EQ(mode.settings.voxel_size, .05f);
        EXPECT_FLOAT_EQ(mode.settings.opacity, .1f);
        EXPECT_TRUE(mode.settings.adaptive_filter);
        EXPECT_FALSE(mode.settings.cluster);
        EXPECT_FALSE(mode.settings.auto_resolution);
        EXPECT_FALSE(mode.overwrite);
        result = parse({path_to_utf8(dir / "out.voxel.json"), "--filter", "connected", "--seed-pos", "-1,2.5,3", "--cluster-resolution", ".4", "--max-gaussian-area", "25", "--voxel-opacity", ".3", "--auto", "--overwrite"});
        ASSERT_TRUE(result) << result.error();
        const auto& custom = std::get<args::VoxelizeMode>(*result);
        EXPECT_TRUE(custom.settings.cluster);
        EXPECT_TRUE(custom.settings.auto_resolution);
        EXPECT_EQ(custom.settings.seed, glm::vec3(-1, 2.5, 3));
        EXPECT_FLOAT_EQ(custom.settings.cluster_resolution, .4f);
        EXPECT_FLOAT_EQ(custom.settings.max_gaussian_area, 25);
        EXPECT_FLOAT_EQ(custom.settings.opacity, .3f);
        EXPECT_TRUE(custom.overwrite);
        result = parse({"-o", path_to_utf8(dir / "out.voxel.json"), "--filter", "none", "--voxel-size", ".02"});
        ASSERT_TRUE(result) << result.error();
        EXPECT_FALSE(std::get<args::VoxelizeMode>(*result).settings.adaptive_filter);
        EXPECT_FLOAT_EQ(std::get<args::VoxelizeMode>(*result).settings.voxel_size, .02f);
    }
    TEST_F(VoxelizeCli, InvalidNumbersPathsSeedsAndConflictingOptionsFailBeforeGpuWork) {
        for (const std::vector<std::string> flags : {
                 std::vector<std::string>{"--voxel-size", "0"},
                 {"--voxel-size", "nan"},
                 {"--voxel-size", "inf"},
                 {"--voxel-opacity", "1.1"},
                 {"--voxel-opacity", "-1"},
                 {"--max-gaussian-area", "-1"},
                 {"--filter", "strict"},
                 {"--auto", "--voxel-size", ".1"},
                 {"--seed-pos", "0,0,0"},
                 {"--filter", "none", "--max-gaussian-area", "1"},
                 {"--filter", "connected", "--seed-pos", "1,2"},
                 {"--filter", "connected", "--seed-pos", "1,2,3,4"},
                 {"--filter", "connected", "--seed-pos", "1,2,nan"},
                 {"--filter", "connected", "--cluster-resolution", "0"}}) {
            auto values = flags;
            values.insert(values.begin(), {"-o", path_to_utf8(dir / "out.voxel.json")});
            EXPECT_FALSE(parse(std::move(values))) << testing::PrintToString(flags);
        }
        EXPECT_FALSE(parse({"-o", "out.glb"}));
        EXPECT_FALSE(parse({}));
        EXPECT_FALSE(parse({"out.voxel.json", "-o", "other.voxel.json"}));
        EXPECT_TRUE(parse({"--help"}));
    }
    TEST_F(VoxelizeCli, RejectsProjectsAndPreservesUnicodePaths) {
        const auto project = path_to_utf8(dir / "input.licht");
        std::ofstream(dir / "input.licht") << "not opened";
        const auto output = path_to_utf8(dir / utf8_to_path("出力_é.voxel.json"));
        const char* values[] = {"LichtFeld-Studio", "voxelize", project.c_str(), "-o", output.c_str()};
        const auto rejected = args::parse_args(5, values);
        ASSERT_FALSE(rejected);
        EXPECT_NE(rejected.error().find(".licht"), std::string::npos);
        auto accepted = parse({"-o", output});
        ASSERT_TRUE(accepted) << accepted.error();
        EXPECT_EQ(path_to_utf8(std::get<args::VoxelizeMode>(*accepted).output_path), output);
    }
    TEST_F(VoxelizeCli, RefusesToReplaceInputIncludingHardLinks) {
        auto mode = options();
        std::filesystem::create_hard_link(mode.input_path, mode.output_path);
        EXPECT_EQ(lfs::app::run_voxelizer(mode), 1);
        EXPECT_EQ(std::filesystem::file_size(mode.input_path), 11);
        std::filesystem::remove(mode.output_path);
        auto binary = mode.output_path;
        binary.replace_extension(".bin");
        std::filesystem::create_hard_link(mode.input_path, binary);
        EXPECT_EQ(lfs::app::run_voxelizer(mode), 1);
        EXPECT_FALSE(std::filesystem::exists(mode.output_path));
        EXPECT_EQ(std::filesystem::file_size(mode.input_path), 11);
    }
    TEST_F(VoxelizeCli, DeclinedOverwriteKeepsBothCompanions) {
        auto mode = options();
        mode.overwrite = false;
        auto binary = mode.output_path;
        binary.replace_extension(".bin");
        std::ofstream(mode.output_path) << "prior json";
        std::ofstream(binary) << "prior bin";
        std::istringstream input("n\n");
        auto* previous = std::cin.rdbuf(input.rdbuf());
        EXPECT_EQ(lfs::app::run_voxelizer(mode), 0);
        std::cin.rdbuf(previous);
        std::cin.clear();
        std::string text;
        std::getline(std::ifstream(mode.output_path), text);
        EXPECT_EQ(text, "prior json");
        std::getline(std::ifstream(binary), text);
        EXPECT_EQ(text, "prior bin");
    }
    class VoxelizeGpuCli : public VoxelizeCli {
        void SetUp() override {
            VoxelizeCli::SetUp();
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
                GTEST_SKIP() << "CUDA device unavailable";
        }
    };
    TEST_F(VoxelizeGpuCli, LoadsPlyAndSpzAndWritesBothCompanionsWithoutChangingTheSource) {
        auto means = Tensor::zeros({16, 3}, Device::CPU);
        auto sh0 = Tensor::zeros({16, 1, 3}, Device::CPU);
        auto scales = Tensor::full({16, 3}, std::log(.04f), Device::CPU);
        auto rotations = Tensor::zeros({16, 4}, Device::CPU);
        auto opacity = Tensor::full({16, 1}, 2.f, Device::CPU);
        for (int i = 0; i < 16; ++i) {
            means.ptr<float>()[i * 3] = (i % 4) * .1f;
            means.ptr<float>()[i * 3 + 2] = (i / 4) * .1f;
            rotations.ptr<float>()[i * 4] = 1;
        }
        SplatData model(0, means, sh0, Tensor{}, scales, rotations, opacity, 1);
        auto mode = options();
        mode.output_path = dir / utf8_to_path("出力_é.voxel.json");
        ASSERT_TRUE(lfs::io::save_ply(model, {.output_path = mode.input_path}));
        const auto source_bytes = std::filesystem::file_size(mode.input_path);
        const auto source_time = std::filesystem::last_write_time(mode.input_path);
        ASSERT_EQ(lfs::app::run_voxelizer(mode), 0);
        nlohmann::json meta;
        std::ifstream(mode.output_path) >> meta;
        EXPECT_EQ(meta["version"], "1.1");
        EXPECT_GT(meta["nodeCount"].get<size_t>(), 0);
        EXPECT_FLOAT_EQ(meta["voxelResolution"], .05f);
        EXPECT_EQ(std::filesystem::file_size(mode.input_path), source_bytes);
        EXPECT_EQ(std::filesystem::last_write_time(mode.input_path), source_time);
        auto spz = dir / "input.spz";
        ASSERT_TRUE(lfs::io::save_spz(model, {.output_path = spz}));
        mode.input_path = spz;
        mode.settings.auto_resolution = true;
        ASSERT_EQ(lfs::app::run_voxelizer(mode), 0);
        std::ifstream(mode.output_path) >> meta;
        EXPECT_GT(meta["nodeCount"].get<size_t>(), 0);
        auto binary = mode.output_path;
        binary.replace_extension(".bin");
        EXPECT_EQ(std::filesystem::file_size(binary), 4 * (meta["nodeCount"].get<size_t>() + meta["leafDataCount"].get<size_t>()));
    }
} // namespace

using namespace lfs::core;
TEST(CollisionSource, ParallelExtractionPreservesTransformsDeletionAndStableOrder) {
    std::vector<CollisionGeometry> snapshots;
    for (size_t n : {size_t(8201), size_t(17)}) {
        CollisionGeometry source;
        source.size = n;
        source.means = Tensor::zeros({n, 3}, Device::CPU);
        source.scaling = Tensor::full({n, 3}, std::log(.2f), Device::CPU);
        source.rotation = Tensor::zeros({n, 4}, Device::CPU);
        source.opacity = Tensor::zeros({n, 1}, Device::CPU);
        source.deleted = Tensor::zeros({n}, Device::CPU, DataType::Bool);
        for (size_t i = 0; i < n; ++i) {
            source.means.ptr<float>()[i * 3] = float(i);
            source.rotation.ptr<float>()[i * 4] = 2; // Deliberately not normalized.
            source.deleted.ptr<bool>()[i] = i % 7 == 0;
        }
        source.world_transform = glm::translate(glm::mat4(1), {3, 4, 5});
        snapshots.push_back(std::move(source));
    }
    const auto frame = glm::scale(glm::mat4(1), {1, -1, -1});
    const auto serial = extractCollisionSource(snapshots, frame, {}, 1);
    const auto parallel = extractCollisionSource(snapshots, frame, {}, 8);
    size_t index = 0;
    for (const auto& source : snapshots)
        for (size_t i = 0; i < source.size; ++i) {
            if (i % 7 == 0)
                continue;
            ASSERT_LT(index, parallel.size());
            EXPECT_EQ(parallel[index].center, glm::vec3(float(i) + 3, -4, -5));
            EXPECT_FLOAT_EQ(parallel[index].opacity, .5f);
            for (int axis = 0; axis < 3; ++axis) {
                EXPECT_EQ(serial[index].axes[axis], parallel[index].axes[axis]);
                EXPECT_FLOAT_EQ(parallel[index].axes[axis][axis], axis == 0 ? .2f : -.2f);
            }
            ++index;
        }
    EXPECT_EQ(index, parallel.size());
    EXPECT_EQ(serial.size(), parallel.size());
    std::stop_source stop;
    stop.request_stop();
    EXPECT_THROW((void)extractCollisionSource(snapshots, frame, stop.get_token()), std::runtime_error);
    snapshots[0].scaling = Tensor{};
    EXPECT_THROW((void)extractCollisionSource(snapshots, frame), std::invalid_argument);
}
