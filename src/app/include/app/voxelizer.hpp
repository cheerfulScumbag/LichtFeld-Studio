/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
namespace lfs::core::args {
    struct VoxelizeMode;
}
namespace lfs::app {
    int run_voxelizer(const core::args::VoxelizeMode& options);
}
