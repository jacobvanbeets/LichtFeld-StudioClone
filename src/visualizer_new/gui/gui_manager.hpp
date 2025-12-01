/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command/commands/cropbox_command.hpp"
#include "core_new/events.hpp"
#include "gui/panels/gizmo_toolbar.hpp"
#include "gui/panels/menu_bar.hpp"
#include "gui/ui_context.hpp"
#include "gui/windows/save_project_browser.hpp"
#include "windows/project_changed_dialog_box.hpp"
#include <GLFW/glfw3.h>
#include <filesystem>
#include <imgui.h>
#include <ImGuizmo.h>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace lfs::vis {
    class VisualizerImpl;

    namespace gui {
        class FileBrowser;
        class ScenePanel;
        class ProjectChangedDialogBox;

        class GuiManager {
        public:
            GuiManager(VisualizerImpl* viewer);
            ~GuiManager();

            // Lifecycle
            void init();
            void shutdown();
            void render();

            // State queries
            bool wantsInput() const;
            bool isAnyWindowActive() const;

            // Window visibility
            void showWindow(const std::string& name, bool show = true);
            void toggleWindow(const std::string& name);

            // Missing methods that visualizer_impl expects
            void setFileSelectedCallback(std::function<void(const std::filesystem::path&, bool)> callback);
            void handleProjectChangedDialogCallback(std::function<void(bool)> callback);

            // Viewport region access
            ImVec2 getViewportPos() const;
            ImVec2 getViewportSize() const;
            bool isMouseInViewport() const;
            bool isViewportFocused() const;
            bool isPositionInViewport(double x, double y) const;

            // Crop box gizmo state access
            void setCropGizmoOperation(ImGuizmo::OPERATION op) { crop_gizmo_operation_ = op; }
            void setCropGizmoMode(ImGuizmo::MODE mode) { crop_gizmo_mode_ = mode; }
            ImGuizmo::OPERATION getCropGizmoOperation() const { return crop_gizmo_operation_; }
            ImGuizmo::MODE getCropGizmoMode() const { return crop_gizmo_mode_; }

            // Selection sub-mode shortcuts (Ctrl+1..5)
            void setSelectionSubMode(panels::SelectionSubMode mode);
            panels::SelectionSubMode getSelectionSubMode() const { return gizmo_toolbar_state_.selection_mode; }
            panels::ToolMode getCurrentToolMode() const { return gizmo_toolbar_state_.current_tool; }

            bool isForceExit() const { return force_exit_; }

        private:
            void setupEventHandlers();
            void applyDefaultStyle();
            void updateViewportRegion();
            void updateViewportFocus();
            void initMenuBar();

            // Core dependencies
            VisualizerImpl* viewer_;

            // Owned components
            std::unique_ptr<FileBrowser> file_browser_;
            std::unique_ptr<ProjectChangedDialogBox> project_changed_dialog_box_;
            std::unique_ptr<ScenePanel> scene_panel_;

            // UI state only
            std::unordered_map<std::string, bool> window_states_;
            bool show_main_panel_ = true;
            bool show_viewport_gizmo_ = true;

            // Speed overlay state
            bool speed_overlay_visible_ = false;
            std::chrono::steady_clock::time_point speed_overlay_start_time_;
            std::chrono::milliseconds speed_overlay_duration_;
            float current_speed_;
            float max_speed_;

            // Zoom speed overlay state
            bool zoom_speed_overlay_visible_ = false;
            std::chrono::steady_clock::time_point zoom_speed_overlay_start_time_;
            float zoom_speed_ = 5.0f;
            float max_zoom_speed_ = 10.0f;

            // Viewport region tracking
            ImVec2 viewport_pos_;
            ImVec2 viewport_size_;
            bool viewport_has_focus_;
            bool force_exit_ = false;

            // Crop box gizmo state (shared between panel and rendering)
            ImGuizmo::OPERATION crop_gizmo_operation_ = ImGuizmo::TRANSLATE;
            ImGuizmo::MODE crop_gizmo_mode_ = ImGuizmo::WORLD;

            // Method declarations
            void renderSpeedOverlay();
            void showSpeedOverlay(float current_speed, float max_speed);
            void renderZoomSpeedOverlay();
            void showZoomSpeedOverlay(float zoom_speed, float max_zoom_speed);
            void renderCropBoxGizmo(const UIContext& ctx);
            void renderNodeTransformGizmo(const UIContext& ctx);

            std::unique_ptr<SaveProjectBrowser> save_project_browser_;
            std::unique_ptr<MenuBar> menu_bar_;

            // Node transform gizmo state
            bool show_node_gizmo_ = true;
            ImGuizmo::OPERATION node_gizmo_operation_ = ImGuizmo::TRANSLATE;

            // Gizmo toolbar state
            panels::GizmoToolbarState gizmo_toolbar_state_;

            // Cropbox undo/redo state
            bool cropbox_gizmo_active_ = false;
            std::optional<command::CropBoxState> cropbox_state_before_drag_;

            // Previous tool/selection mode for detecting changes
            panels::ToolMode previous_tool_ = panels::ToolMode::None;
            panels::SelectionSubMode previous_selection_mode_ = panels::SelectionSubMode::Centers;

            // Tool cleanup
            void deactivateAllTools();

            // Crop box flash effect
            std::chrono::steady_clock::time_point crop_flash_start_;
            bool crop_flash_active_ = false;
            void triggerCropFlash();
            void updateCropFlash();

            // Save PLY dialog state
            bool show_save_ply_dialog_ = false;
            std::string save_ply_node_name_;
            std::string save_ply_path_;
        };
    } // namespace gui
} // namespace lfs::vis