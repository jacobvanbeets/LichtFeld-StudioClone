/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <functional>
#include <string>

namespace lfs::io {

    enum class ExtractionMode {
        FPS,      // Extract at specific FPS
        INTERVAL  // Extract every N frames
    };

    enum class ImageFormat {
        PNG,
        JPG
    };

    class VideoFrameExtractor {
    public:
        VideoFrameExtractor();
        ~VideoFrameExtractor();

        struct Params {
            std::filesystem::path video_path;
            std::filesystem::path output_dir;
            ExtractionMode mode = ExtractionMode::FPS;
            double fps = 1.0;
            int frame_interval = 1;
            ImageFormat format = ImageFormat::PNG;
            int jpg_quality = 95;
            std::function<void(int, int)> progress_callback;  // (current, total)
        };

        bool extract(const Params& params, std::string& error);

    private:
        class Impl;
        Impl* impl_;
    };

} // namespace lfs::io
