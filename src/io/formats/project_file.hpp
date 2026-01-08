/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include <expected>
#include <filesystem>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::vis {
    class Scene;
}

namespace lfs::io {

    /**
     * @brief LichtFeld Studio Project File (.lfsp)
     * 
     * A self-contained project format that bundles:
     * - Scene graph hierarchy (nodes, transforms, visibility)
     * - All PLY/splat files embedded
     * - Crop box settings
     * - Camera settings
     * - Metadata (version, creation date)
     * 
     * Format: ZIP archive containing:
     *   - project.json: Scene state and metadata
     *   - models/model_0.ply: Embedded PLY files
     *   - models/model_1.ply: ...
     */

    constexpr const char* PROJECT_FILE_VERSION = "1.0";
    constexpr const char* PROJECT_FILE_EXTENSION = ".lfsp";

    struct ProjectFileMetadata {
        std::string version = PROJECT_FILE_VERSION;
        std::string created_with;  // Application version
        std::string creation_date; // ISO 8601 timestamp
    };

    /**
     * @brief Save the current scene as a self-contained project file
     * 
     * @param path Output path for .lfsp file
     * @param scene Scene to save
     * @return Success or error message
     */
    std::expected<void, std::string> save_project_file(
        const std::filesystem::path& path,
        const lfs::vis::Scene& scene);

    /**
     * @brief Load a project file into the scene
     * 
     * @param path Path to .lfsp file
     * @param scene Scene to load into (will be cleared first)
     * @return Success or error message
     */
    std::expected<void, std::string> load_project_file(
        const std::filesystem::path& path,
        lfs::vis::Scene& scene);

} // namespace lfs::io
