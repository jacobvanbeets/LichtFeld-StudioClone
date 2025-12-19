/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sequencer_settings_panel.hpp"
#include "core/events.hpp"
#include "gui/ui_widgets.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "theme/theme.hpp"
#include <imgui.h>

namespace lfs::vis::gui::panels {

    using namespace lfs::io::video;
    using namespace lfs::core::events;

    void DrawSequencerSection(const UIContext& ctx, SequencerUIState& state) {
        widgets::SectionHeader("SEQUENCER", ctx.fonts);

        if (ImGui::Checkbox("Show Camera Path", &state.show_camera_path)) {
            // Path visibility is handled by gui_manager reading this state
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Display camera path in viewport");
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text("Video Export");
        ImGui::Spacing();

        // Resolution combo
        constexpr const char* RESOLUTION_ITEMS[] = {
            "720p (1280x720)",
            "1080p (1920x1080)",
            "4K (3840x2160)"
        };
        int res_idx = static_cast<int>(state.resolution);
        if (ImGui::Combo("Resolution", &res_idx, RESOLUTION_ITEMS, 3)) {
            state.resolution = static_cast<VideoResolution>(res_idx);
        }

        // Framerate combo
        constexpr const char* FPS_ITEMS[] = {"24 fps", "30 fps", "60 fps"};
        constexpr int FPS_VALUES[] = {24, 30, 60};
        int fps_idx = (state.framerate == 24) ? 0 : (state.framerate == 60) ? 2 : 1;
        if (ImGui::Combo("Framerate", &fps_idx, FPS_ITEMS, 3)) {
            state.framerate = FPS_VALUES[fps_idx];
        }

        // Quality slider (CRF: lower = better)
        ImGui::SliderInt("Quality", &state.quality, 15, 28, "CRF %d");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Lower = higher quality, larger file");
        }

        ImGui::Spacing();

        // Export button
        const bool has_keyframes = ctx.sequencer_controller &&
                                    !ctx.sequencer_controller->timeline().empty();

        if (!has_keyframes) {
            ImGui::BeginDisabled();
        }

        if (ImGui::Button("Export Video...", ImVec2(-1, 0))) {
            cmd::SequencerExportVideo{
                .resolution = static_cast<int>(state.resolution),
                .framerate = state.framerate,
                .crf = state.quality
            }.emit();
        }

        if (!has_keyframes) {
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Add keyframes first (press K)");
            }
        }
    }

} // namespace lfs::vis::gui::panels
