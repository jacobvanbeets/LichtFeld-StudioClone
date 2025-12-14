/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/splat_data.hpp"
#include <expected>
#include <filesystem>
#include <string>

namespace lfs::io {

    using lfs::core::SplatData;

    /**
     * @brief Check if a PLY file is in compressed format
     * @param filepath Path to the PLY file
     * @return true if the file is a compressed PLY
     */
    bool is_compressed_ply(const std::filesystem::path& filepath);

    /**
     * @brief Load compressed PLY file and return SplatData
     *
     * Compressed PLY format uses chunk-based quantization:
     * - 256 splats per chunk with min/max bounds
     * - Position: 11-10-11 bit packed (uint32)
     * - Rotation: 10-10-10-2 bit packed (uint32)
     * - Scale: 11-10-11 bit packed (uint32)
     * - Color+Opacity: 8-8-8-8 bit packed (uint32)
     * - Optional SH: uint8 per coefficient
     *
     * @param filepath Path to the compressed PLY file
     * @return SplatData on success, error string on failure
     */
    std::expected<SplatData, std::string>
    load_compressed_ply(const std::filesystem::path& filepath);

    /**
     * @brief Options for compressed PLY export
     */
    struct CompressedPlyWriteOptions {
        std::filesystem::path output_path;
        bool include_sh = true;  // Include higher-order SH coefficients
    };

    /**
     * @brief Write a compressed PLY file
     *
     * Uses chunk-based compression with Morton ordering for spatial coherence.
     * Each chunk contains 256 splats with shared min/max bounds.
     *
     * @param splat_data The splat data to export
     * @param options Export options including output path
     * @return Success or error string
     */
    std::expected<void, std::string>
    write_compressed_ply(const SplatData& splat_data,
                         const CompressedPlyWriteOptions& options);

} // namespace lfs::io
