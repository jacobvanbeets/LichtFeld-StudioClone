/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "checkpoint_loader.hpp"
#include "core_new/logger.hpp"
#include "core_new/splat_data.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>

namespace lfs::io {

    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    std::expected<LoadResult, std::string> CheckpointLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("Checkpoint Loading");
        auto start_time = std::chrono::high_resolution_clock::now();

        if (options.progress) {
            options.progress(0.0f, "Loading checkpoint file...");
        }

        // Validate file exists
        if (!std::filesystem::exists(path)) {
            return std::unexpected(std::format("Checkpoint file does not exist: {}", path.string()));
        }

        if (!std::filesystem::is_regular_file(path)) {
            return std::unexpected(std::format("Path is not a regular file: {}", path.string()));
        }

        // Validation only mode
        if (options.validate_only) {
            LOG_DEBUG("Validation only mode for checkpoint: {}", path.string());

            auto header_result = loadHeader(path);
            if (!header_result) {
                return std::unexpected(header_result.error());
            }

            if (options.progress) {
                options.progress(100.0f, "Checkpoint validation complete");
            }

            LoadResult result;
            result.data = std::shared_ptr<SplatData>{};
            result.scene_center = Tensor::zeros({3}, Device::CPU);
            result.loader_used = name();
            result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time);
            result.warnings = {};

            return result;
        }

        // Load SplatData from checkpoint
        if (options.progress) {
            options.progress(50.0f, "Parsing checkpoint data...");
        }

        LOG_INFO("Loading checkpoint file: {}", path.string());
        auto splat_result = lfs::training::load_checkpoint_splat_data(path);
        if (!splat_result) {
            LOG_ERROR("Failed to load checkpoint: {}", splat_result.error());
            return std::unexpected(splat_result.error());
        }

        if (options.progress) {
            options.progress(100.0f, "Checkpoint loading complete");
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Get iteration from header for info
        const auto header = loadHeader(path);
        const int iteration = header ? header->iteration : 0;

        auto splat_ptr = std::make_shared<SplatData>(std::move(*splat_result));
        const auto num_gaussians = splat_ptr->size();

        LoadResult result{
            .data = std::move(splat_ptr),
            .scene_center = Tensor::zeros({3}, Device::CPU),
            .loader_used = name(),
            .load_time = load_time,
            .warnings = {std::format("Checkpoint from iteration {}", iteration)}};

        LOG_INFO("Checkpoint loaded successfully in {}ms ({} Gaussians from iteration {})",
                 load_time.count(), num_gaussians, iteration);

        return result;
    }

    bool CheckpointLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path) || std::filesystem::is_directory(path)) {
            return false;
        }

        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext != ".resume") {
            return false;
        }

        // Verify magic number
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return false;
        }

        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        return magic == lfs::training::CHECKPOINT_MAGIC;
    }

    std::string CheckpointLoader::name() const {
        return "Checkpoint";
    }

    std::vector<std::string> CheckpointLoader::supportedExtensions() const {
        return {".resume", ".RESUME"};
    }

    int CheckpointLoader::priority() const {
        return 15;  // Higher priority than PLY since it contains more info
    }

    std::expected<lfs::training::CheckpointHeader, std::string> CheckpointLoader::loadHeader(
        const std::filesystem::path& path) {
        return lfs::training::load_checkpoint_header(path);
    }

    std::expected<int, std::string> CheckpointLoader::getIteration(
        const std::filesystem::path& path) {
        auto header = loadHeader(path);
        if (!header) {
            return std::unexpected(header.error());
        }
        return header->iteration;
    }

} // namespace lfs::io
