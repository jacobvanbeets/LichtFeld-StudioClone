/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_extractor_dialog.hpp"
#include "gui/utils/windows_utils.hpp"

#include <imgui.h>
#include <array>

using lfs::vis::gui::OpenVideoFileDialog;
using lfs::vis::gui::SelectFolderDialog;

namespace lfs::gui {

    VideoExtractorDialog::VideoExtractorDialog() = default;

    void VideoExtractorDialog::setOnStartExtraction(std::function<void(const VideoExtractionParams&)> callback) {
        on_start_extraction_ = std::move(callback);
    }
    
    void VideoExtractorDialog::updateProgress(int current, int total) {
        current_frame_.store(current);
        total_frames_.store(total);
    }
    
    void VideoExtractorDialog::setExtractionComplete() {
        extracting_.store(false);
        error_message_.clear();
        show_completion_message_ = true;
    }
    
    void VideoExtractorDialog::setExtractionError(const std::string& error) {
        extracting_.store(false);
        error_message_ = error;
    }

    void VideoExtractorDialog::renderFileSelection() {
        ImGui::SeparatorText("Input Video");
        
        // Video file selection
        ImGui::Text("Video File:");
        ImGui::SameLine();
        
        std::string video_display = video_path_.empty() ? "No file selected" : video_path_.filename().string();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", video_display.c_str());
        
        if (ImGui::Button("Browse Video...")) {
            const auto path = OpenVideoFileDialog();
            if (!path.empty()) {
                video_path_ = path;
                // Auto-set output directory to video directory / video_name_frames
                if (output_dir_.empty()) {
                    output_dir_ = video_path_.parent_path() / (video_path_.stem().string() + "_frames");
                }
            }
        }

        ImGui::Spacing();

        // Output directory selection
        ImGui::Text("Output Directory:");
        ImGui::SameLine();
        
        std::string output_display = output_dir_.empty() ? "No directory selected" : output_dir_.string();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", output_display.c_str());
        
        if (ImGui::Button("Browse Output...")) {
            const auto path = SelectFolderDialog("Select Output Folder");
            if (!path.empty()) {
                output_dir_ = path;
            }
        }
        
    }

    void VideoExtractorDialog::renderExtractionSettings() {
        ImGui::SeparatorText("Extraction Settings");
        
        // Mode selection
        ImGui::Text("Extraction Mode:");
        std::array<const char*, 2> modes = {"FPS-based", "Frame Interval"};
        ImGui::Combo("##mode", &mode_selection_, modes.data(), static_cast<int>(modes.size()));
        
        ImGui::Spacing();
        
        // Mode-specific settings
        if (mode_selection_ == 0) {
            // FPS mode
            ImGui::Text("Frames Per Second:");
            ImGui::SliderFloat("##fps", &fps_, 0.1f, 30.0f, "%.1f FPS");
            ImGui::TextWrapped("Extract frames at the specified rate (e.g., 1 FPS = 1 frame per second)");
        } else {
            // Interval mode
            ImGui::Text("Frame Interval:");
            ImGui::SliderInt("##interval", &frame_interval_, 1, 100, "Every %d frames");
            ImGui::TextWrapped("Extract every Nth frame from the video");
        }
    }

    void VideoExtractorDialog::renderFormatSettings() {
        ImGui::SeparatorText("Output Format");
        
        // Format selection
        std::array<const char*, 2> formats = {"PNG (lossless)", "JPEG (smaller)"};
        ImGui::Combo("##format", &format_selection_, formats.data(), static_cast<int>(formats.size()));
        
        // JPEG quality slider (only show for JPEG)
        if (format_selection_ == 1) {
            ImGui::Spacing();
            ImGui::Text("JPEG Quality:");
            ImGui::SliderInt("##quality", &jpg_quality_, 50, 100, "%d%%");
        }
    }

    void VideoExtractorDialog::render(bool* p_open) {
        if (!*p_open) {
            return;
        }

        ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
        
        if (ImGui::Begin("Extract Frames from Video", p_open)) {
            ImGui::TextWrapped("Extract individual frames from a video file to prepare a dataset for 3D reconstruction.");
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            renderFileSelection();
            ImGui::Spacing();
            renderExtractionSettings();
            ImGui::Spacing();
            renderFormatSettings();
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Start button
            const bool can_start = !video_path_.empty() && !output_dir_.empty();
            
            if (!can_start) {
                ImGui::BeginDisabled();
            }
            
            if (ImGui::Button("Start Extraction", ImVec2(150, 30))) {
                if (on_start_extraction_ && !extracting_.load()) {
                    VideoExtractionParams params;
                    params.video_path = video_path_;
                    params.output_dir = output_dir_;
                    params.mode = mode_selection_ == 0 ? io::ExtractionMode::FPS : io::ExtractionMode::INTERVAL;
                    params.fps = static_cast<double>(fps_);
                    params.frame_interval = frame_interval_;
                    params.format = format_selection_ == 0 ? io::ImageFormat::PNG : io::ImageFormat::JPG;
                    params.jpg_quality = jpg_quality_;
                    
                    extracting_.store(true);
                    current_frame_.store(0);
                    total_frames_.store(0);
                    error_message_.clear();
                    show_completion_message_ = false;
                    
                    on_start_extraction_(params);
                }
            }
            
            if (!can_start) {
                ImGui::EndDisabled();
            }
            
            ImGui::SameLine();
            
            if (ImGui::Button("Cancel", ImVec2(100, 30))) {
                *p_open = false;
            }
            
            if (!can_start) {
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), 
                                   "Please select a video file and output directory");
            }
            
            // Show progress
            if (extracting_.load()) {
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                
                int current = current_frame_.load();
                int total = total_frames_.load();
                
                if (total > 0) {
                    float progress = static_cast<float>(current) / static_cast<float>(total);
                    ImGui::Text("Extracting: %d / %d frames", current, total);
                    ImGui::ProgressBar(progress, ImVec2(-1, 0));
                } else {
                    ImGui::Text("Starting extraction...");
                    ImGui::ProgressBar(0.0f, ImVec2(-1, 0));
                }
            }
            
            // Show completion message
            if (show_completion_message_ && !extracting_.load()) {
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                
                int total = current_frame_.load();
                ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), 
                                   "Extraction completed successfully!");
                ImGui::Text("Extracted %d frames", total);
                
                if (ImGui::Button("OK", ImVec2(100, 30))) {
                    show_completion_message_ = false;
                    current_frame_.store(0);
                    total_frames_.store(0);
                }
            }
            
            // Show error if any
            if (!error_message_.empty()) {
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "Error: %s", error_message_.c_str());
                
                if (ImGui::Button("Dismiss", ImVec2(100, 30))) {
                    error_message_.clear();
                }
            }
        }
        ImGui::End();
    }

} // namespace lfs::gui
