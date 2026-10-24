/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/voxelizer.hpp"
#include "app/converter_overwrite.hpp"
#include "app/headless_run_coordinator.hpp"
#include "app/terminal_progress_bar.hpp"
#include "core/argument_parser.hpp"
#include "core/collision_source.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "io/collision_export.hpp"
#include "io/loader.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <print>

namespace lfs::app {
    int run_voxelizer(const core::args::VoxelizeMode& options) {
        using namespace core;
        HeadlessRunCoordinator run;
        const auto stop = run.stop_token();
        std::string stage = "Checking output paths";
        try {
            options.settings.validate();
            if (!path_to_utf8(options.output_path.filename()).ends_with(".voxel.json"))
                throw std::invalid_argument("Output must end in .voxel.json");
            auto binary = options.output_path;
            binary.replace_extension(".bin");
            bool overwrite_all = options.overwrite;
            for (const auto& target : {options.output_path, binary}) {
                std::error_code error;
                if (target == options.input_path || native_fs::equivalent(target, options.input_path, error))
                    throw std::invalid_argument("Output must not replace the input splat file");
                if (native_fs::exists(target)) {
                    if (!native_fs::is_regular_file(target))
                        throw std::invalid_argument("Output is not a regular file: " + path_to_utf8(target));
                    if (!overwrite_all) {
                        const auto choice = ask_overwrite(target, std::cin, std::cout);
                        if (choice == OverwriteChoice::NO) {
                            std::println("Skipped existing voxel pair. Use --overwrite to replace both files.");
                            return 0;
                        }
                        overwrite_all = choice == OverwriteChoice::ALL;
                    }
                }
            }
            TerminalProgressBar bar("Voxelize");
            std::mutex progress_mutex;
            float completed = 0;
            auto report = [&](float value, std::string text) {
                if (stop.stop_requested() && value < 1)
                    throw std::runtime_error("Voxelization cancelled");
                std::lock_guard lock(progress_mutex);
                stage = std::move(text);
                completed = std::max(completed, std::clamp(value, 0.f, 1.f));
                bar.report(completed, stage);
            };
            report(0, "Loading splat data");
            if (io::Loader::isDatasetPath(options.input_path))
                throw std::invalid_argument("Input is a dataset; train and export the splats first");
            auto loaded = io::Loader::create()->load(options.input_path, {.progress = [&](float value, const std::string& text) { report(.15f * std::clamp(value / 100, 0.f, 1.f), text); },
                                                                          .cancel_requested = [&] { return stop.stop_requested(); }});
            if (!loaded)
                throw std::runtime_error(loaded.error().format());
            auto* model = std::get_if<std::shared_ptr<SplatData>>(&loaded->data);
            if (!model || !*model || loaded->point_cloud_source)
                throw std::invalid_argument("Input is not a Gaussian splat model");
            if ((*model)->lod_tree && (*model)->lod_tree->has_tree()) {
                report(.15f, "Extracting original Gaussians from LOD data");
                auto leaves = extract_lod_leaves(**model);
                if (!leaves)
                    throw std::runtime_error(std::string(leaves.error().detail()));
                *model = std::make_shared<SplatData>(std::move(*leaves));
            }
            if ((*model)->size() > CollisionGaussian::max_count)
                throw std::runtime_error("Collision source exceeds 32 million Gaussians");
            report(.2f, "Reading positions, scales, rotations and opacity");
            std::vector<CollisionGaussian> source;
            {
                auto host = [](const Tensor& tensor) {
                    return tensor.is_valid() ? tensor.to(Device::CPU).contiguous() : Tensor{};
                };
                const CollisionGeometry geometry{
                    host((*model)->means_raw()), host((*model)->scaling_raw()), host((*model)->rotation_raw()),
                    host((*model)->opacity_raw()), host((*model)->deleted()), glm::mat4(1), (*model)->size()};
                source = extractCollisionSource({&geometry, 1}, glm::mat4(1), stop);
            }
            // Do not retain appearance/LOD/tensor storage during voxelization.
            model->reset();
            report(.25f, "Generating sparse collision occupancy");
            const auto start = std::chrono::steady_clock::now();
            auto volume = generateCollision(source, options.settings, stop, collisionProgressRange(report, .25f, .85f), voxelizeCollisionGpu);
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const auto source_count = source.size();
            source.clear();
            source.shrink_to_fit();
            native_fs::create_directories(options.output_path.parent_path());
            // Match the rotation SuperSplat applies when importing the source PLY.
            io::exportCollisionVoxels(volume, glm::mat4(1), options.output_path, stop, collisionProgressRange(report, .85f, 1));
            bar.complete();
            std::println("Generated {} occupied voxels from {} Gaussians in {:.3f}s (cell size {:.6g} scene units).",
                         volume.occupiedCount(), source_count, seconds, volume.settings.voxel_size);
            if (!volume.warning.empty())
                std::println("{}", volume.warning);
            std::println("Saved {}\n      {}", path_to_utf8(options.output_path), path_to_utf8(binary));
            return 0;
        } catch (const std::exception& error) {
            if (run.interrupted()) {
                std::println(stderr, "Voxelization cancelled during {}. No incomplete voxel pair was published.", stage);
                return run.interrupted_exit_code();
            }
            LOG_ERROR("Voxelization failed during {}: {}", stage, error.what());
            std::println(stderr, "Voxelization failed during {}: {}", stage, error.what());
            return 1;
        }
    }
} // namespace lfs::app
