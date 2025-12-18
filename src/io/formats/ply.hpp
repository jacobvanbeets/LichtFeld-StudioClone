/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Re-export public API
#include "core/point_cloud.hpp"
#include "io/exporter.hpp"

namespace lfs::io {

    // Check if PLY contains Gaussian splat properties (opacity, scaling, rotation)
    bool is_gaussian_splat_ply(const std::filesystem::path& filepath);

    // Load PLY as Gaussian splat (with opacity, scaling, rotation, SH)
    std::expected<SplatData, std::string> load_ply(const std::filesystem::path& filepath);

    // Load PLY as simple point cloud (xyz + optional colors)
    std::expected<lfs::core::PointCloud, std::string> load_ply_point_cloud(const std::filesystem::path& filepath);

    // Alias for backward compatibility
    using SaveProgressCallback = ExportProgressCallback;

} // namespace lfs::io
