/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

namespace lfs::io::video {

    enum class VideoResolution : uint8_t {
        HD_720P,   // 1280x720
        FHD_1080P, // 1920x1080
        UHD_4K     // 3840x2160
    };

    [[nodiscard]] inline constexpr int getWidth(const VideoResolution res) {
        switch (res) {
            case VideoResolution::HD_720P: return 1280;
            case VideoResolution::FHD_1080P: return 1920;
            case VideoResolution::UHD_4K: return 3840;
        }
        return 1920;
    }

    [[nodiscard]] inline constexpr int getHeight(const VideoResolution res) {
        switch (res) {
            case VideoResolution::HD_720P: return 720;
            case VideoResolution::FHD_1080P: return 1080;
            case VideoResolution::UHD_4K: return 2160;
        }
        return 1080;
    }

    [[nodiscard]] inline constexpr const char* getResolutionName(const VideoResolution res) {
        switch (res) {
            case VideoResolution::HD_720P: return "720p (1280x720)";
            case VideoResolution::FHD_1080P: return "1080p (1920x1080)";
            case VideoResolution::UHD_4K: return "4K (3840x2160)";
        }
        return "1080p";
    }

    struct VideoExportOptions {
        VideoResolution resolution = VideoResolution::FHD_1080P;
        int framerate = 30;
        int crf = 18; // Constant Rate Factor (15-28, lower = better quality)
    };

} // namespace lfs::io::video
