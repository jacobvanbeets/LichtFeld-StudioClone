/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sequencer_settings_panel.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "theme/theme.hpp"
#include <imgui.h>

namespace lfs::vis::gui::panels {

    using namespace lfs::io::video;
    using namespace lfs::core::events;

    namespace {
        constexpr int MIN_WIDTH = 320;
        constexpr int MAX_WIDTH = 7680;
        constexpr int MIN_HEIGHT = 240;
        constexpr int MAX_HEIGHT = 4320;
        constexpr const char* FPS_ITEMS[] = {"24 fps", "30 fps", "60 fps"};
        constexpr int FPS_VALUES[] = {24, 30, 60};
        constexpr const char* SNAP_ITEMS[] = {"0.25s", "0.5s", "1s", "2s"};
        constexpr float SNAP_VALUES[] = {0.25f, 0.5f, 1.0f, 2.0f};
        constexpr const char* SPEED_ITEMS[] = {"0.25x", "0.5x", "1x", "2x", "4x"};
        constexpr float SPEED_VALUES[] = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f};

        // Popup styling constants
        constexpr float POPUP_ALPHA = 0.98f;
        constexpr float BORDER_SIZE = 2.0f;
        constexpr ImVec2 POPUP_PADDING = {20.0f, 16.0f};
        constexpr ImVec2 BUTTON_SIZE = {80.0f, 0.0f};
        constexpr ImGuiWindowFlags POPUP_FLAGS = ImGuiWindowFlags_AlwaysAutoResize |
                                                 ImGuiWindowFlags_NoCollapse |
                                                 ImGuiWindowFlags_NoDocking;
    } // namespace

    void DrawSequencerSection(const UIContext& ctx, SequencerUIState& state) {
        widgets::SectionHeader("SEQUENCER", ctx.fonts);

        const bool has_keyframes = ctx.sequencer_controller &&
                                   !ctx.sequencer_controller->timeline().empty();

        // Camera path and playback settings
        ImGui::Checkbox("Show Camera Path", &state.show_camera_path);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Display camera path in viewport");
        }

        // Playback speed
        int speed_idx = 2; // Default 1x
        for (int i = 0; i < 5; ++i) {
            if (std::abs(state.playback_speed - SPEED_VALUES[i]) < 0.01f) {
                speed_idx = i;
                break;
            }
        }
        if (ImGui::Combo("Speed", &speed_idx, SPEED_ITEMS, 5)) {
            state.playback_speed = SPEED_VALUES[speed_idx];
            if (ctx.sequencer_controller) {
                ctx.sequencer_controller->setPlaybackSpeed(state.playback_speed);
            }
        }

        // Snap to grid
        ImGui::Checkbox("Snap to Grid", &state.snap_to_grid);
        if (state.snap_to_grid) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(60);
            int snap_idx = 1; // Default 0.5s
            for (int i = 0; i < 4; ++i) {
                if (std::abs(state.snap_interval - SNAP_VALUES[i]) < 0.01f) {
                    snap_idx = i;
                    break;
                }
            }
            if (ImGui::Combo("##snap_interval", &snap_idx, SNAP_ITEMS, 4)) {
                state.snap_interval = SNAP_VALUES[snap_idx];
            }
        }

        ImGui::Checkbox("Follow Playback", &state.follow_playback);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Camera follows playhead during playback");

        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::SliderFloat("Preview Size", &state.pip_preview_scale, 0.5f, 2.0f, "%.1fx");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Scale the preview window");

        ImGui::Spacing();

        // Save/Load camera path
        const float btn_width = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;

        if (!has_keyframes) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Save Path...", ImVec2(btn_width, 0))) {
            if (const auto path = SaveJsonFileDialog("camera_path"); !path.empty() && ctx.sequencer_controller) {
                if (ctx.sequencer_controller->timeline().saveToJson(path.string())) {
                    LOG_INFO("Camera path saved to {}", path.string());
                } else {
                    LOG_ERROR("Failed to save camera path to {}", path.string());
                }
            }
        }
        if (!has_keyframes) {
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Add keyframes first");
            }
        }

        ImGui::SameLine();

        if (ImGui::Button("Load Path...", ImVec2(btn_width, 0))) {
            if (const auto path = OpenJsonFileDialog(); !path.empty() && ctx.sequencer_controller) {
                if (ctx.sequencer_controller->timeline().loadFromJson(path.string())) {
                    LOG_INFO("Camera path loaded from {}", path.string());
                } else {
                    LOG_ERROR("Failed to load camera path from {}", path.string());
                }
            }
        }

        // Clear all keyframes button
        if (!has_keyframes) {
            ImGui::BeginDisabled();
        }
        if (widgets::ColoredButton("Clear All Keyframes", widgets::ButtonStyle::Error)) {
            ImGui::OpenPopup("ConfirmClearKeyframes");
        }
        if (!has_keyframes) {
            ImGui::EndDisabled();
        }

        // Themed confirmation popup
        const auto& t = theme();
        const ImVec4 popup_bg = {t.palette.surface.x, t.palette.surface.y, t.palette.surface.z, POPUP_ALPHA};
        const ImVec4 title_bg = darken(t.palette.surface, 0.1f);
        const ImVec4 title_bg_active = darken(t.palette.surface, 0.05f);

        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
        ImGui::PushStyleColor(ImGuiCol_WindowBg, popup_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, title_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, title_bg_active);
        ImGui::PushStyleColor(ImGuiCol_Border, t.palette.error);
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.text);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, BORDER_SIZE);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, POPUP_PADDING);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.popup_rounding);

        if (ImGui::BeginPopupModal("ConfirmClearKeyframes", nullptr, POPUP_FLAGS)) {
            ImGui::TextColored(t.palette.error, "Warning");
            ImGui::SameLine();
            ImGui::TextColored(t.palette.text_dim, "|");
            ImGui::SameLine();
            ImGui::TextUnformatted("Clear Camera Path");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("Delete all keyframes? This cannot be undone.");

            ImGui::Spacing();
            ImGui::Spacing();

            const float avail = ImGui::GetContentRegionAvail().x;
            const float total_width = BUTTON_SIZE.x * 2 + ImGui::GetStyle().ItemSpacing.x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - total_width);

            if (ImGui::Button("Cancel", BUTTON_SIZE) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (widgets::ColoredButton("Delete", widgets::ButtonStyle::Error, BUTTON_SIZE)) {
                if (ctx.sequencer_controller) {
                    ctx.sequencer_controller->timeline().clear();
                }
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(5);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Video Export section
        ImGui::Text("Video Export");
        ImGui::Spacing();

        const auto current_info = getPresetInfo(state.preset);
        if (ImGui::BeginCombo("Format", current_info.name)) {
            for (int i = 0; i < getPresetCount(); ++i) {
                const auto p = static_cast<VideoPreset>(i);
                const auto info = getPresetInfo(p);
                const bool selected = (state.preset == p);

                if (ImGui::Selectable(info.name, selected)) {
                    state.preset = p;
                    if (p != VideoPreset::CUSTOM) {
                        state.framerate = info.framerate;
                        state.quality = info.crf;
                    }
                }

                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", info.description);
                }

                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        if (state.preset == VideoPreset::CUSTOM) {
            ImGui::InputInt("Width", &state.custom_width, 16, 64);
            ImGui::InputInt("Height", &state.custom_height, 16, 64);
            state.custom_width = std::clamp(state.custom_width, MIN_WIDTH, MAX_WIDTH);
            state.custom_height = std::clamp(state.custom_height, MIN_HEIGHT, MAX_HEIGHT);

            int fps_idx = (state.framerate == 24) ? 0 : (state.framerate == 60) ? 2
                                                                                : 1;
            if (ImGui::Combo("Framerate", &fps_idx, FPS_ITEMS, 3)) {
                state.framerate = FPS_VALUES[fps_idx];
            }
        } else {
            ImGui::TextDisabled("%s", current_info.description);
        }

        ImGui::SliderInt("Quality", &state.quality, 15, 28, "CRF %d");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Lower = higher quality, larger file");
        }

        ImGui::Spacing();

        if (!has_keyframes) {
            ImGui::BeginDisabled();
        }

        if (widgets::ColoredButton("Export Video...", widgets::ButtonStyle::Primary)) {
            const auto info = getPresetInfo(state.preset);
            const int width = (state.preset == VideoPreset::CUSTOM) ? state.custom_width : info.width;
            const int height = (state.preset == VideoPreset::CUSTOM) ? state.custom_height : info.height;

            cmd::SequencerExportVideo{
                .width = width,
                .height = height,
                .framerate = state.framerate,
                .crf = state.quality}
                .emit();
        }

        if (!has_keyframes) {
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Add keyframes first (press K)");
            }
        }
    }

} // namespace lfs::vis::gui::panels
