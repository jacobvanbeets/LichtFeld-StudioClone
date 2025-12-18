/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/sogs.hpp"
#include "io/exporter.hpp"

namespace lfs::core {

    std::expected<void, std::string> write_sog(
        const SplatData& splat_data,
        const SogWriteOptions& options) {

        // Forward to the io module implementation
        lfs::io::SogSaveOptions io_options{
            .output_path = options.output_path,
            .kmeans_iterations = options.iterations,
            .use_gpu = options.use_gpu,
            .progress_callback = options.progress_callback};

        auto result = lfs::io::save_sog(splat_data, io_options);
        if (!result) {
            return std::unexpected(result.error().format());
        }
        return {};
    }

} // namespace lfs::core
