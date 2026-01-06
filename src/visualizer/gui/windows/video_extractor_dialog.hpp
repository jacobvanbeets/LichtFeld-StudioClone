/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/video_frame_extractor.hpp"
#include <atomic>
#include <filesystem>
#include <functional>
#include <string>

namespace lfs::gui {

    struct VideoExtractionParams {
        std::filesystem::path video_path;
        std::filesystem::path output_dir;
        io::ExtractionMode mode = io::ExtractionMode::FPS;
        double fps = 1.0;
        int frame_interval = 1;
        io::ImageFormat format = io::ImageFormat::PNG;
        int jpg_quality = 95;
    };

    class VideoExtractorDialog {
    public:
        VideoExtractorDialog();

        void render(bool* p_open);
        void setOnStartExtraction(std::function<void(const VideoExtractionParams&)> callback);
        
        void updateProgress(int current, int total);
        void setExtractionComplete();
        void setExtractionError(const std::string& error);

    private:
        void renderFileSelection();
        void renderExtractionSettings();
        void renderFormatSettings();

        std::filesystem::path video_path_;
        std::filesystem::path output_dir_;
        
        int mode_selection_ = 0;  // 0 = FPS, 1 = INTERVAL
        float fps_ = 1.0f;
        int frame_interval_ = 1;
        
        int format_selection_ = 0;  // 0 = PNG, 1 = JPG
        int jpg_quality_ = 95;

        std::function<void(const VideoExtractionParams&)> on_start_extraction_;
        
        // Progress tracking
        std::atomic<bool> extracting_{false};
        std::atomic<int> current_frame_{0};
        std::atomic<int> total_frames_{0};
        std::string error_message_;
        bool show_completion_message_ = false;
    };

} // namespace lfs::gui
