/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/main_panel.hpp"
#include "core/editor_context.hpp"
#include "core_new/events.hpp"
#include "gui/panels/tools_panel.hpp"
#include "gui/panels/training_panel.hpp"
#include "gui/ui_widgets.hpp"
#include "scene/scene_manager.hpp"
#include "tools/selection_tool.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <format>
#include <imgui.h>
#ifdef WIN32
#include <windows.h>
#endif

namespace lfs::vis::gui::panels {

    using namespace lfs::core::events;

    void DrawWindowControls(const UIContext& ctx) {

        ImGui::Text("Windows");
        ImGui::Checkbox("Scene Panel", &(*ctx.window_states)["scene_panel"]);
    }

    void DrawSystemConsoleButton(const UIContext& ctx) {

#ifdef WIN32
        // On non-Windows platforms, dont show the console toggle button

        if (!ctx.window_states->at("system_console")) {
            if (ImGui::Button("Show system Console", ImVec2(-1, 0))) {
                HWND hwnd = GetConsoleWindow();
                Sleep(1);
                HWND owner = GetWindow(hwnd, GW_OWNER);

                if (owner == NULL) {
                    ShowWindow(hwnd, SW_SHOW); // Windows 10
                } else {
                    ShowWindow(owner, SW_SHOW); // Windows 11
                }
                ctx.window_states->at("system_console") = true;
            }
        } else {
            if (ImGui::Button("Hide system Console", ImVec2(-1, 0))) {
                HWND hwnd = GetConsoleWindow();
                Sleep(1);
                HWND owner = GetWindow(hwnd, GW_OWNER);

                if (owner == NULL) {
                    ShowWindow(hwnd, SW_HIDE); // Windows 10
                } else {
                    ShowWindow(owner, SW_HIDE); // Windows 11
                }
                ctx.window_states->at("system_console") = false;
            }
        }
#endif // Win32
    }

    void DrawRenderingSettings(const UIContext& ctx) {
        auto render_manager = ctx.viewer->getRenderingManager();
        if (!render_manager)
            return;

        // Use EditorContext to check pre-training mode
        const bool force_point_cloud = ctx.editor && ctx.editor->forcePointCloudMode();

        // Get current render settings
        auto settings = render_manager->getSettings();
        bool settings_changed = false;

        // In pre-training mode, point cloud mode is always on
        if (force_point_cloud && !settings.point_cloud_mode) {
            settings.point_cloud_mode = true;
            settings_changed = true;
            ui::PointCloudModeChanged{.enabled = true, .voxel_size = settings.voxel_size}.emit();
        }

        // Point Cloud Mode checkbox (disabled in pre-training mode - always on)
        ImGui::BeginDisabled(force_point_cloud);
        if (ImGui::Checkbox("Point Cloud Mode", &settings.point_cloud_mode)) {
            settings_changed = true;

            ui::PointCloudModeChanged{
                .enabled = settings.point_cloud_mode,
                .voxel_size = settings.voxel_size}
                .emit();
        }
        ImGui::EndDisabled();
        if (force_point_cloud && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Always enabled in pre-training mode");
        }

        // Show voxel size slider only when in point cloud mode
        if (settings.point_cloud_mode) {
            if (widgets::SliderWithReset("Voxel Size", &settings.voxel_size, 0.001f, 0.1f, 0.03f)) {
                settings_changed = true;

                ui::PointCloudModeChanged{
                    .enabled = settings.point_cloud_mode,
                    .voxel_size = settings.voxel_size}
                    .emit();
            }
        }

        // Ring Mode and Center Markers are mutually exclusive
        ImGui::BeginDisabled(settings.point_cloud_mode);
        if (ImGui::Checkbox("Show Gaussian Rings", &settings.show_rings)) {
            if (settings.show_rings)
                settings.show_center_markers = false;
            settings_changed = true;
        }
        ImGui::EndDisabled();

        if (settings.show_rings && !settings.point_cloud_mode) {
            if (widgets::SliderWithReset("Ring Width", &settings.ring_width, 0.001f, 0.05f, 0.01f)) {
                settings_changed = true;
            }
        }

        ImGui::BeginDisabled(settings.point_cloud_mode);
        if (ImGui::Checkbox("Show Center Markers", &settings.show_center_markers)) {
            if (settings.show_center_markers)
                settings.show_rings = false;
            settings_changed = true;
        }
        ImGui::EndDisabled();

        // Background Color
        ImGui::Separator();
        ImGui::Text("Background");
        float bg_color[3] = {settings.background_color.x, settings.background_color.y, settings.background_color.z};
        if (ImGui::ColorEdit3("Color##Background", bg_color)) {
            settings.background_color = glm::vec3(bg_color[0], bg_color[1], bg_color[2]);
            settings_changed = true;
        }

        // Coordinate Axes
        ImGui::Separator();
        if (ImGui::Checkbox("Show Coordinate Axes", &settings.show_coord_axes)) {
            settings_changed = true;
        }

        if (settings.show_coord_axes) {
            ImGui::Indent();

            if (ImGui::SliderFloat("Axes Size", &settings.axes_size, 0.5f, 10.0f)) {
                settings_changed = true;
            }

            ImGui::Text("Visible Axes:");
            bool axes_changed = false;
            if (ImGui::Checkbox("X##axis", &settings.axes_visibility[0])) {
                axes_changed = true;
            }
            ImGui::SameLine();
            if (ImGui::Checkbox("Y##axis", &settings.axes_visibility[1])) {
                axes_changed = true;
            }
            ImGui::SameLine();
            if (ImGui::Checkbox("Z##axis", &settings.axes_visibility[2])) {
                axes_changed = true;
            }

            if (axes_changed) {
                settings_changed = true;
            }

            ImGui::Unindent();
        }

        // Camera Rotation
        if (ImGui::Checkbox("Lock Gimbal", &settings.lock_gimbal)) {
            settings_changed = true;
            cmd::ToggleGimbalLock{
                .locked = settings.lock_gimbal}
                .emit();
        }


        // Pivot Point
        ImGui::Separator();
        if (ImGui::Checkbox("Show Pivot Point", &settings.show_pivot)) {
            settings_changed = true;
        }

        // Grid checkbox and settings
        ImGui::Separator();
        if (ImGui::Checkbox("Show Grid", &settings.show_grid)) {
            settings_changed = true;

            // Emit grid settings changed event
            ui::GridSettingsChanged{
                .enabled = settings.show_grid,
                .plane = static_cast<int>(settings.grid_plane),
                .opacity = settings.grid_opacity}
                .emit();
        }

        // Show grid settings only when grid is enabled
        if (settings.show_grid) {
            ImGui::Indent();

            // Grid plane selection
            const char* planes[] = {"YZ (X-plane)", "XZ (Y-plane)", "XY (Z-plane)"};
            int current_plane = static_cast<int>(settings.grid_plane);
            if (ImGui::Combo("Plane", &current_plane, planes, IM_ARRAYSIZE(planes))) {
                settings.grid_plane = current_plane;
                settings_changed = true;

                ui::GridSettingsChanged{
                    .enabled = settings.show_grid,
                    .plane = current_plane,
                    .opacity = settings.grid_opacity}
                    .emit();
            }

            // Grid opacity
            if (ImGui::SliderFloat("Grid Opacity", &settings.grid_opacity, 0.0f, 1.0f)) {
                settings_changed = true;

                ui::GridSettingsChanged{
                    .enabled = settings.show_grid,
                    .plane = static_cast<int>(settings.grid_plane),
                    .opacity = settings.grid_opacity}
                    .emit();
            }

            ImGui::Unindent();
        }

        // Camera Frustums
        ImGui::Separator();
        if (widgets::SliderWithReset("Camera Scale", &settings.camera_frustum_scale, 0.01f, 10.0f, 0.25f)) {
            settings_changed = true;
        }

        // Selection Colors
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Selection Colors")) {
            ImGui::Indent();

            auto color_edit = [&](const char* label, glm::vec3& color) {
                float c[3] = {color.x, color.y, color.z};
                if (ImGui::ColorEdit3(label, c)) {
                    color = glm::vec3(c[0], c[1], c[2]);
                    settings_changed = true;
                }
            };

            color_edit("Committed##sel", settings.selection_color_committed);
            color_edit("Preview##sel", settings.selection_color_preview);
            color_edit("Center Marker##sel", settings.selection_color_center_marker);

            ImGui::Unindent();
        }

        // Apply settings changes if any
        if (settings_changed) {
            render_manager->updateSettings(settings);
        }

        ImGui::Separator();

        // Use direct accessors from RenderingManager
        float scaling_modifier = settings.scaling_modifier;
        if (widgets::SliderWithReset("Scale", &scaling_modifier, 0.01f, 3.0f, 1.0f)) {
            render_manager->setScalingModifier(scaling_modifier);

            ui::RenderSettingsChanged{
                .sh_degree = std::nullopt,
                .fov = std::nullopt,
                .scaling_modifier = scaling_modifier,
                .antialiasing = std::nullopt,
                .background_color = std::nullopt,
                .equirectangular = std::nullopt}
                .emit();
        }

        float fov = settings.fov;
        if (widgets::SliderWithReset("FoV", &fov, 45.0f, 120.0f, 75.0f)) {
            render_manager->setFov(fov);

            ui::RenderSettingsChanged{
                .fov = fov,
                .scaling_modifier = std::nullopt,
                .antialiasing = std::nullopt,
                .background_color = std::nullopt,
                .equirectangular = std::nullopt}
                .emit();
        }

        // SH DEGREE selection
        const char* sh_degrees[] = {"0", "1", "2", "3"};
        int current_sh_degree = static_cast<int>(settings.sh_degree);
        if (ImGui::Combo("SH Degree", &current_sh_degree, sh_degrees, IM_ARRAYSIZE(sh_degrees))) {
            settings.sh_degree = current_sh_degree;
            settings_changed = true;

            ui::RenderSettingsChanged{
                .sh_degree = current_sh_degree,
                .fov = std::nullopt,
                .scaling_modifier = std::nullopt,
                .antialiasing = std::nullopt,
                .background_color = std::nullopt,
                .equirectangular = std::nullopt}
                .emit();
        }

        if (ImGui::Checkbox("Equirectangular", &settings.equirectangular)) {
            settings_changed = true;

            ui::RenderSettingsChanged{
                .sh_degree = std::nullopt,
                .fov = std::nullopt,
                .scaling_modifier = std::nullopt,
                .antialiasing = std::nullopt,
                .background_color = std::nullopt,
                .equirectangular = settings.equirectangular}
                .emit();
        }

        // Display current FPS
        float average_fps = ctx.viewer->getAverageFPS();
        if (average_fps > 0.0f) {
            ImGui::Text("FPS: %6.1f", average_fps);
        }

#ifdef CUDA_GL_INTEROP_ENABLED
        ImGui::Text("Render Mode: GPU Direct (Interop)");
#else
        ImGui::Text("Render Mode: CPU Copy");
#endif
    }

    void DrawSelectionGroups(const UIContext& ctx) {
        // Only show when selection tool is active
        auto* selection_tool = ctx.viewer->getSelectionTool();
        if (!selection_tool || !selection_tool->isEnabled()) return;

        auto* scene_manager = ctx.viewer->getSceneManager();
        if (!scene_manager) return;

        Scene& scene = scene_manager->getScene();

        if (!ImGui::CollapsingHeader("Selection Groups", ImGuiTreeNodeFlags_DefaultOpen)) return;

        const float button_width = ImGui::GetContentRegionAvail().x;

        if (ImGui::Button("+ Add Group", ImVec2(button_width, 0))) {
            if (selection_tool->hasActivePolygon()) {
                selection_tool->clearPolygon();
            }
            scene.addSelectionGroup("", glm::vec3(0.0f));
        }

        const auto& groups = scene.getSelectionGroups();
        const uint8_t active_id = scene.getActiveSelectionGroup();

        if (groups.empty()) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No selection groups");
        } else {
            scene.updateSelectionGroupCounts();

            for (const auto& group : groups) {
                ImGui::PushID(group.id);

                const bool is_active = (group.id == active_id);

                // Highlight active group with colored background
                if (is_active) {
                    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(group.color.r * 0.4f, group.color.g * 0.4f, group.color.b * 0.4f, 0.6f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(group.color.r * 0.5f, group.color.g * 0.5f, group.color.b * 0.5f, 0.7f));
                }

                // Lock button
                const bool is_locked = group.locked;
                if (is_locked) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
                }
                if (ImGui::SmallButton(is_locked ? "L" : "U")) {
                    scene.setSelectionGroupLocked(group.id, !is_locked);
                }
                if (is_locked) {
                    ImGui::PopStyleColor();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(is_locked ? "Locked: other groups cannot overwrite" : "Unlocked: can be overwritten");
                }
                ImGui::SameLine();

                // Color picker
                ImVec4 color(group.color.r, group.color.g, group.color.b, 1.0f);
                if (ImGui::ColorEdit3("##color", &color.x,
                    ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_NoAlpha)) {
                    scene.setSelectionGroupColor(group.id, glm::vec3(color.x, color.y, color.z));
                }
                ImGui::SameLine();

                // Selectable group - show active marker
                const std::string label = is_active
                    ? std::format("> {} ({})", group.name, group.count)
                    : std::format("  {} ({})", group.name, group.count);
                if (ImGui::Selectable(label.c_str(), is_active)) {
                    if (selection_tool->hasActivePolygon()) {
                        selection_tool->clearPolygon();
                    }
                    scene.setActiveSelectionGroup(group.id);
                }

                if (is_active) {
                    ImGui::PopStyleColor(2);
                }

                // Context menu
                if (ImGui::BeginPopupContextItem("##ctx")) {
                    if (ImGui::MenuItem(is_locked ? "Unlock" : "Lock")) {
                        scene.setSelectionGroupLocked(group.id, !is_locked);
                    }
                    if (ImGui::MenuItem("Clear")) {
                        if (is_active && selection_tool->hasActivePolygon()) {
                            selection_tool->clearPolygon();
                        }
                        scene.clearSelectionGroup(group.id);
                    }
                    if (ImGui::MenuItem("Delete")) {
                        if (is_active && selection_tool->hasActivePolygon()) {
                            selection_tool->clearPolygon();
                        }
                        scene.removeSelectionGroup(group.id);
                    }
                    ImGui::EndPopup();
                }

                ImGui::PopID();
            }
        }
    }

    void DrawProgressInfo(const UIContext& ctx) {
        // Get training info from TrainerManager instead of ViewerStateManager
        auto* trainer_manager = ctx.viewer->getTrainerManager();
        if (!trainer_manager) {
            return;
        }

        int current_iter = trainer_manager->getCurrentIteration();
        int total_iter = trainer_manager->getTotalIterations();
        int num_splats = trainer_manager->getNumSplats();

        // Fix: Convert deque to vector
        std::deque<float> loss_deque = trainer_manager->getLossBuffer();
        std::vector<float> loss_data(loss_deque.begin(), loss_deque.end());

        float fraction = total_iter > 0 ? float(current_iter) / float(total_iter) : 0.0f;
        char overlay_text[64];
        std::snprintf(overlay_text, sizeof(overlay_text), "%d / %d", current_iter, total_iter);

        widgets::DrawProgressBar(fraction, overlay_text);

        if (loss_data.size() > 0) {
            auto [min_it, max_it] = std::minmax_element(loss_data.begin(), loss_data.end());
            float min_val = *min_it, max_val = *max_it;

            if (min_val == max_val) {
                min_val -= 1.0f;
                max_val += 1.0f;
            } else {
                float margin = (max_val - min_val) * 0.05f;
                min_val -= margin;
                max_val += margin;
            }

            char loss_label[64];
            std::snprintf(loss_label, sizeof(loss_label), "Loss: %.4f", loss_data.back());

            widgets::DrawLossPlot(loss_data.data(), static_cast<int>(loss_data.size()),
                                  min_val, max_val, loss_label);
        }
    }
} // namespace lfs::vis::gui::panels
