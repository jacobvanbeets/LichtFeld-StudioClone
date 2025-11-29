/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// glad must be included before OpenGL headers
// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/gui_manager.hpp"
#include "command/command_history.hpp"
#include "core_new/image_io.hpp"
#include "core_new/logger.hpp"
#include "core_new/splat_data_export.hpp"
#include "project_new/project.hpp"
#include "gui/panels/main_panel.hpp"
#include "gui/panels/scene_panel.hpp"
#include "gui/panels/tools_panel.hpp"
#include "gui/panels/training_panel.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "gui/windows/file_browser.hpp"
#include "gui/windows/project_changed_dialog_box.hpp"

#include "internal/resource_paths.hpp"
#include "tools/align_tool.hpp"
#include "tools/brush_tool.hpp"
#include "tools/selection_tool.hpp"
#include "rendering/rendering_manager.hpp"
#include "visualizer_impl.hpp"

#include <chrono>
#include <cstdarg>
#include <format>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_internal.h>
#include <ImGuizmo.h>

namespace lfs::vis::gui {

    GuiManager::GuiManager(VisualizerImpl* viewer)
        : viewer_(viewer) {

        // Create components
        file_browser_ = std::make_unique<FileBrowser>();
        project_changed_dialog_box_ = std::make_unique<ProjectChangedDialogBox>();
        scene_panel_ = std::make_unique<ScenePanel>(viewer->trainer_manager_);
        save_project_browser_ = std::make_unique<SaveProjectBrowser>();
        menu_bar_ = std::make_unique<MenuBar>();

        // Initialize window states
        window_states_["file_browser"] = false;
        window_states_["scene_panel"] = true;
        window_states_["project_changed_dialog_box"] = false;
        window_states_["save_project_browser_before_exit"] = false;
        window_states_["show_save_browser"] = false;
        window_states_["system_console"] = false;
        window_states_["training_tab"] = false;

        // Initialize speed overlay state
        speed_overlay_visible_ = false;
        speed_overlay_duration_ = std::chrono::milliseconds(3000); // 3 seconds

        // Initialize focus state
        viewport_has_focus_ = false;

        setupEventHandlers();
    }

    GuiManager::~GuiManager() {
        // Cleanup handled automatically
    }

    void GuiManager::initMenuBar() {
        menu_bar_->setOnImportDataset([this]() {
            window_states_["file_browser"] = true;
#ifdef WIN32
            // show native windows file dialog for project file selection
            OpenDatasetFolderDialog();

            // hide the file browser
            lfs::core::events::cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif // WIN32
        });

        menu_bar_->setOnOpenProject([this]() {
            window_states_["file_browser"] = true;
#ifdef WIN32
            // show native windows file dialog for project file selection
            OpenProjectFileDialog();

            // hide the file browser
            lfs::core::events::cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif // WIN32
        });

        menu_bar_->setOnImportPLY([this]() {
            window_states_["file_browser"] = true;
#ifdef WIN32
            // show native windows file dialog for project file selection
            OpenPlyFileDialog();

            // hide the file browser
            lfs::core::events::cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif // WIN32
        });

        menu_bar_->setOnSaveProjectAs([this]() {
            window_states_["show_save_browser"] = true;
        });

        menu_bar_->setOnSaveProject([this]() {
            if (viewer_->project_) {
                lfs::core::events::cmd::SaveProject{viewer_->project_->getProjectOutputFolder().string()}.emit();
            }
        });

        menu_bar_->setOnExit([this]() {
            glfwSetWindowShouldClose(viewer_->getWindow(), true);
        });
    }

    void GuiManager::init() {
        // ImGui initialization
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
        io.ConfigWindowsMoveFromTitleBarOnly = true;

        // Platform/Renderer initialization
        ImGui_ImplGlfw_InitForOpenGL(viewer_->getWindow(), true);
        ImGui_ImplOpenGL3_Init("#version 430");

        float xscale, yscale;
        glfwGetWindowContentScale(viewer_->getWindow(), &xscale, &yscale);

        // some clamping / safety net for weird DPI values
        xscale = std::clamp(xscale, 1.0f, 2.0f);

        // Set application icon - use the resource path helper
        try {
            const auto icon_path = lfs::vis::getAssetPath("lichtfeld-icon.png");
            const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(icon_path);

            GLFWimage image{width, height, data};
            glfwSetWindowIcon(viewer_->getWindow(), 1, &image);
            lfs::core::free_image(data);
        } catch (const std::exception& e) {
            LOG_WARN("Could not load application icon: {}", e.what());
        }

        // Load fonts - use the resource path helper
        try {
            auto font_path = lfs::vis::getAssetPath("JetBrainsMono-Regular.ttf");
            io.Fonts->AddFontFromFileTTF(font_path.string().c_str(), 14.0f * xscale);
        } catch (const std::exception& e) {
            // If font loading fails, just use the default font
            LOG_WARN("Could not load custom font: {}", e.what());
            LOG_DEBUG("Using default ImGui font");
        }

        applyDefaultStyle();

        // Configure file browser callback
        setFileSelectedCallback([this](const std::filesystem::path& path, bool is_dataset) {
            if (path.extension() == lfs::project::Project::EXTENSION) {
                lfs::core::events::cmd::LoadProject{.path = path}.emit();
            } else {
                lfs::core::events::cmd::LoadFile{.path = path, .is_dataset = is_dataset}.emit();
            }

            window_states_["file_browser"] = false;
        });

        handleProjectChangedDialogCallback([this](bool save) {
            if (save) {
                window_states_["save_project_browser_before_exit"] = true;
            } else {
                force_exit_ = true;
                glfwSetWindowShouldClose(viewer_->getWindow(), true);
                LOG_INFO("Exiting LichtFeldStudio gracefully without saving");
            }
            window_states_["project_changed_dialog_box"] = false;
        });

        scene_panel_->setOnDatasetLoad([this](const std::filesystem::path& path) {
            if (path.empty()) {
                window_states_["file_browser"] = true;
            } else {
                lfs::core::events::cmd::LoadFile{.path = path, .is_dataset = true}.emit();
            }
        });

        initMenuBar();
    }

    void GuiManager::shutdown() {
        // Cleanup toolbar textures
        panels::ShutdownGizmoToolbar(gizmo_toolbar_state_);

        if (ImGui::GetCurrentContext()) {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        }
    }

    void GuiManager::render() {
        // Start frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();

        // Check mouse state before ImGui::NewFrame() updates WantCaptureMouse
        ImVec2 mouse_pos = ImGui::GetMousePos();
        bool mouse_in_viewport = isPositionInViewport(mouse_pos.x, mouse_pos.y);

        ImGui::NewFrame();

        // Initialize ImGuizmo for this frame
        ImGuizmo::BeginFrame();

        if (menu_bar_) {
            menu_bar_->render();
        }

        // Override ImGui's mouse capture for gizmo interaction
        // If ImGuizmo is being used or hovered, let it handle the mouse
        if (ImGuizmo::IsOver() || ImGuizmo::IsUsing()) {
            ImGui::GetIO().WantCaptureMouse = false;
            ImGui::GetIO().WantCaptureKeyboard = false;
        }

        // Override ImGui's mouse capture for right/middle buttons when in viewport
        // This ensures that camera controls work properly
        if (mouse_in_viewport && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
                ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
                ImGui::GetIO().WantCaptureMouse = false;
            }
        }

        // In point cloud mode, disable ImGui mouse capture in viewport
        auto* rendering_manager = viewer_->getRenderingManager();
        if (rendering_manager) {
            const auto& settings = rendering_manager->getSettings();
            if (settings.point_cloud_mode && mouse_in_viewport &&
                !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
                ImGui::GetIO().WantCaptureMouse = false;
                ImGui::GetIO().WantCaptureKeyboard = false;
            }
        }

        // Create main dockspace
        const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(main_viewport->WorkPos);
        ImGui::SetNextWindowSize(main_viewport->WorkSize);
        ImGui::SetNextWindowViewport(main_viewport->ID);

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking |
                                        ImGuiWindowFlags_NoTitleBar |
                                        ImGuiWindowFlags_NoCollapse |
                                        ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoMove |
                                        ImGuiWindowFlags_NoBringToFrontOnFocus |
                                        ImGuiWindowFlags_NoNavFocus |
                                        ImGuiWindowFlags_NoBackground;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

        ImGui::Begin("DockSpace", nullptr, window_flags);
        ImGui::PopStyleVar(3);

        // DockSpace ID
        ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");

        // Create dockspace
        ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

        // Set up default layout on first run
        static bool first_time = true;
        if (first_time) {
            first_time = false;
            ImGui::DockBuilderRemoveNode(dockspace_id);
            ImGui::DockBuilderAddNode(dockspace_id, dockspace_flags | ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(dockspace_id, main_viewport->WorkSize);

            ImGuiID dock_id_left = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.25f, nullptr, &dockspace_id);
            ImGuiID dock_id_right = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Right, 0.2f, nullptr, &dockspace_id);

            // Dock windows
            ImGui::DockBuilderDockWindow("Rendering", dock_id_left);
            ImGui::DockBuilderDockWindow("Scene", dock_id_right);
            ImGui::DockBuilderDockWindow("Training", dock_id_left);

            ImGui::DockBuilderFinish(dockspace_id);
        }

        ImGui::End();

        // Create context for this frame
        UIContext ctx{
            .viewer = viewer_,
            .file_browser = file_browser_.get(),
            .window_states = &window_states_};

        // Draw docked panels
        if (show_main_panel_) {
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));
            if (ImGui::Begin("Rendering", nullptr)) {
                // Draw contents without the manual sizing/positioning
                widgets::DrawModeStatusWithContentSwitch(ctx);
                ImGui::Separator();
                panels::DrawRenderingSettings(ctx);
                ImGui::Separator();
                panels::DrawToolsPanel(ctx);
                panels::DrawSystemConsoleButton(ctx);
            }
            ImGui::End();

            if (viewer_->getTrainer() && !window_states_["training_tab"]) {
                ImGui::SetWindowFocus("Rendering");
                window_states_["training_tab"] = true;
            }

            if (!viewer_->getTrainer()) {
                window_states_["training_tab"] = false;
            }

            if (window_states_["training_tab"]) {
                if (ImGui::Begin("Training", nullptr)) {
                    panels::DrawTrainingControls(ctx);
                    ImGui::Separator();
                    panels::DrawProgressInfo(ctx);
                }
                ImGui::End();
            }

            ImGui::PopStyleColor();
        }

        // Draw Scene panel
        if (window_states_["scene_panel"]) {
            scene_panel_->render(&window_states_["scene_panel"]);
        }

        // Render floating windows (these remain movable)
        if (window_states_["file_browser"]) {
            file_browser_->render(&window_states_["file_browser"]);
        }

        if (window_states_["project_changed_dialog_box"]) {
            project_changed_dialog_box_->render(&window_states_["project_changed_dialog_box"]);
        }

        if (window_states_["save_project_browser_before_exit"]) {
#ifdef WIN32
            bool was_project_saved = save_project_browser_->SaveProjectFileDialog(&window_states_["save_project_browser_before_exit"]);
#else
            bool was_project_saved = save_project_browser_->render(&window_states_["save_project_browser_before_exit"]);
#endif
            if (was_project_saved) {
                force_exit_ = true;
                glfwSetWindowShouldClose(viewer_->getWindow(), true);
                LOG_INFO("Exiting LichtFeldStudio gracefully after project save");
            }
        }

        if (window_states_["show_save_browser"]) {
#ifdef WIN32
            save_project_browser_->SaveProjectFileDialog(&window_states_["show_save_browser"]);
#else
            save_project_browser_->render(&window_states_["show_save_browser"]);
#endif
        }

        // Save PLY dialog
        if (show_save_ply_dialog_) {
            // Materialize deletions before saving
            if (auto* sm = viewer_->getSceneManager()) {
                sm->applyDeleted();
            }

#ifdef WIN32
            // Use native Windows file dialog
            std::filesystem::path save_path = SavePlyFileDialog(save_ply_node_name_);
            if (!save_path.empty()) {
                if (auto* sm = viewer_->getSceneManager()) {
                    const auto* node = sm->getScene().getNode(save_ply_node_name_);
                    if (node && node->model) {
                        lfs::core::save_ply(*node->model, save_path.parent_path(), 0, true, save_path.stem().string());
                        LOG_INFO("Saved PLY to: {}", save_path.string());
                    }
                }
            }
            show_save_ply_dialog_ = false;
#else
            ImGui::OpenPopup("Save PLY As");
#endif
        }

#ifndef WIN32
        // ImGui fallback dialog for non-Windows platforms
        if (ImGui::BeginPopupModal("Save PLY As", &show_save_ply_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Save \"%s\" as:", save_ply_node_name_.c_str());
            ImGui::Separator();

            char path_buf[512];
            strncpy(path_buf, save_ply_path_.c_str(), sizeof(path_buf) - 1);
            path_buf[sizeof(path_buf) - 1] = '\0';

            ImGui::SetNextItemWidth(400);
            if (ImGui::InputText("##path", path_buf, sizeof(path_buf))) {
                save_ply_path_ = path_buf;
            }

            ImGui::Separator();

            if (ImGui::Button("Save", ImVec2(120, 0))) {
                if (auto* sm = viewer_->getSceneManager()) {
                    const auto* node = sm->getScene().getNode(save_ply_node_name_);
                    if (node && node->model) {
                        std::filesystem::path save_path(save_ply_path_);
                        lfs::core::save_ply(*node->model, save_path.parent_path(), 0, true, save_path.stem().string());
                        LOG_INFO("Saved PLY to: {}", save_ply_path_);
                    }
                }
                show_save_ply_dialog_ = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                show_save_ply_dialog_ = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
#endif

        if (menu_bar_ && viewer_) {
            auto project = viewer_->getProject();
            menu_bar_->setIsProjectTemp(project ? project->getIsTempProject() : false);
        }

        // Render gizmo toolbar (only when a node is selected)
        auto* scene_manager = ctx.viewer->getSceneManager();
        if (scene_manager && scene_manager->hasSelectedNode()) {
            panels::DrawGizmoToolbar(ctx, gizmo_toolbar_state_, viewport_pos_, viewport_size_);

            const auto current_tool = gizmo_toolbar_state_.current_tool;
            const bool is_transform_tool = (current_tool == panels::ToolMode::Translate ||
                                            current_tool == panels::ToolMode::Rotate ||
                                            current_tool == panels::ToolMode::Scale);
            show_node_gizmo_ = is_transform_tool;
            if (is_transform_tool) {
                node_gizmo_operation_ = gizmo_toolbar_state_.current_operation;
            }

            auto* brush_tool = ctx.viewer->getBrushTool();
            auto* align_tool = ctx.viewer->getAlignTool();
            auto* selection_tool = ctx.viewer->getSelectionTool();
            const bool is_brush_mode = (current_tool == panels::ToolMode::Brush);
            const bool is_align_mode = (current_tool == panels::ToolMode::Align);
            const bool is_selection_mode = (current_tool == panels::ToolMode::Selection);
            const bool is_cropbox_mode = (current_tool == panels::ToolMode::CropBox);

            // Materialize deletions when switching away from selection or cropbox tools
            const bool was_selection_mode = (previous_tool_ == panels::ToolMode::Selection);
            const bool was_cropbox_mode = (previous_tool_ == panels::ToolMode::CropBox);
            if ((was_selection_mode || was_cropbox_mode) && current_tool != previous_tool_) {
                if (auto* sm = ctx.viewer->getSceneManager()) {
                    sm->applyDeleted();
                }
            }
            previous_tool_ = current_tool;

            if (brush_tool) brush_tool->setEnabled(is_brush_mode);
            if (align_tool) align_tool->setEnabled(is_align_mode);
            if (selection_tool) selection_tool->setEnabled(is_selection_mode);

            // Update selection mode and auto-toggle ring rendering
            if (is_selection_mode) {
                if (auto* rm = ctx.viewer->getRenderingManager()) {
                    lfs::rendering::SelectionMode mode = lfs::rendering::SelectionMode::Centers;
                    switch (gizmo_toolbar_state_.selection_mode) {
                        case panels::SelectionSubMode::Centers:   mode = lfs::rendering::SelectionMode::Centers; break;
                        case panels::SelectionSubMode::Rectangle: mode = lfs::rendering::SelectionMode::Rectangle; break;
                        case panels::SelectionSubMode::Polygon:   mode = lfs::rendering::SelectionMode::Polygon; break;
                        case panels::SelectionSubMode::Lasso:     mode = lfs::rendering::SelectionMode::Lasso; break;
                        case panels::SelectionSubMode::Rings:     mode = lfs::rendering::SelectionMode::Rings; break;
                    }
                    rm->setSelectionMode(mode);

                    // Reset selection state when switching modes
                    if (gizmo_toolbar_state_.selection_mode != previous_selection_mode_) {
                        if (selection_tool) selection_tool->onSelectionModeChanged();

                        // Auto-enable rings when switching to Rings sub-mode
                        if (gizmo_toolbar_state_.selection_mode == panels::SelectionSubMode::Rings) {
                            auto settings = rm->getSettings();
                            settings.show_rings = true;
                            settings.show_center_markers = false;
                            rm->updateSettings(settings);
                        }

                        previous_selection_mode_ = gizmo_toolbar_state_.selection_mode;
                    }

                }
            }

            if (is_cropbox_mode) {
                switch (gizmo_toolbar_state_.cropbox_operation) {
                    case panels::CropBoxOperation::Bounds:    crop_gizmo_operation_ = ImGuizmo::BOUNDS; break;
                    case panels::CropBoxOperation::Translate: crop_gizmo_operation_ = ImGuizmo::TRANSLATE; break;
                    case panels::CropBoxOperation::Rotate:    crop_gizmo_operation_ = ImGuizmo::ROTATE; break;
                    case panels::CropBoxOperation::Scale:     crop_gizmo_operation_ = ImGuizmo::SCALE; break;
                }
            }

            // Toggle cropbox visibility with tool
            if (auto* render_manager = ctx.viewer->getRenderingManager()) {
                auto settings = render_manager->getSettings();
                if (is_cropbox_mode != settings.show_crop_box) {
                    settings.show_crop_box = is_cropbox_mode;
                    render_manager->updateSettings(settings);
                }
            }

            // Handle reset cropbox request from toolbar
            if (gizmo_toolbar_state_.reset_cropbox_requested) {
                gizmo_toolbar_state_.reset_cropbox_requested = false;
                if (auto* render_manager = ctx.viewer->getRenderingManager()) {
                    auto settings = render_manager->getSettings();

                    // Capture state before reset
                    const command::CropBoxState old_state{
                        .crop_min = settings.crop_min,
                        .crop_max = settings.crop_max,
                        .crop_transform = settings.crop_transform,
                        .crop_inverse = settings.crop_inverse
                    };

                    // Apply reset
                    settings.crop_min = glm::vec3(-1.0f, -1.0f, -1.0f);
                    settings.crop_max = glm::vec3(1.0f, 1.0f, 1.0f);
                    settings.crop_transform = lfs::geometry::EuclideanTransform();
                    settings.use_crop_box = false;
                    settings.crop_inverse = false;
                    render_manager->updateSettings(settings);

                    // Capture state after reset
                    const command::CropBoxState new_state{
                        .crop_min = settings.crop_min,
                        .crop_max = settings.crop_max,
                        .crop_transform = settings.crop_transform,
                        .crop_inverse = settings.crop_inverse
                    };

                    // Create undo command
                    auto cmd = std::make_unique<command::CropBoxCommand>(
                        render_manager, old_state, new_state);
                    viewer_->getCommandHistory().execute(std::move(cmd));
                }
            }
        } else {
            show_node_gizmo_ = false;
            auto* brush_tool = ctx.viewer->getBrushTool();
            auto* align_tool = ctx.viewer->getAlignTool();
            if (brush_tool) brush_tool->setEnabled(false);
            if (align_tool) align_tool->setEnabled(false);
        }

        auto* brush_tool = ctx.viewer->getBrushTool();
        if (brush_tool && brush_tool->isEnabled()) {
            brush_tool->renderUI(ctx, nullptr);
        }

        auto* selection_tool = ctx.viewer->getSelectionTool();
        if (selection_tool && selection_tool->isEnabled()) {
            selection_tool->renderUI(ctx, nullptr);
        }

        auto* align_tool = ctx.viewer->getAlignTool();
        if (align_tool && align_tool->isEnabled()) {
            align_tool->renderUI(ctx, nullptr);
        }

        // Render crop box gizmo over viewport
        renderCropBoxGizmo(ctx);

        // Render node transform gizmo (for translating selected PLY nodes)
        renderNodeTransformGizmo(ctx);

        // Render speed overlays if visible
        renderSpeedOverlay();
        renderZoomSpeedOverlay();

        // Render split view indicator if enabled
        if (rendering_manager) {
            auto split_info = rendering_manager->getSplitViewInfo();
            if (split_info.enabled) {
                // Create a small overlay showing what is being compared
                const ImGuiViewport* viewport = ImGui::GetMainViewport();

                // Position at top center
                ImVec2 overlay_pos(
                    viewport->WorkPos.x + (viewport->WorkSize.x - 400.0f) * 0.5f,
                    viewport->WorkPos.y + 10.0f);

                ImGui::SetNextWindowPos(overlay_pos, ImGuiCond_Always);
                ImGui::SetNextWindowSize(ImVec2(400.0f, 40.0f), ImGuiCond_Always);

                ImGuiWindowFlags overlay_flags =
                    ImGuiWindowFlags_NoTitleBar |
                    ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoScrollbar |
                    ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoSavedSettings |
                    ImGuiWindowFlags_NoInputs |
                    ImGuiWindowFlags_NoFocusOnAppearing;

                ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.8f));
                ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1.0f, 1.0f, 0.0f, 0.5f));
                ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 5.0f);

                if (ImGui::Begin("##SplitViewIndicator", nullptr, overlay_flags)) {
                    ImGui::SetCursorPos(ImVec2(10, 10));

                    // Draw the split view info - check mode
                    const auto& settings = rendering_manager->getSettings();
                    if (settings.split_view_mode == SplitViewMode::GTComparison) {
                        // GT comparison mode
                        int cam_id = rendering_manager->getCurrentCameraId();
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f),
                                           "GT Comparison - Camera %d", cam_id);
                        ImGui::SameLine();
                        ImGui::TextDisabled(" (G: toggle, V: cycle modes, arrows: change camera)");
                    } else {
                        // PLY comparison mode
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f),
                                           "Split View: %s | %s",
                                           split_info.left_name.c_str(),
                                           split_info.right_name.c_str());
                        ImGui::SameLine();
                        ImGui::TextDisabled(" (T: cycle, V: exit)");
                    }
                }
                ImGui::End();

                ImGui::PopStyleVar(2);
                ImGui::PopStyleColor(2);
            }
        }

        // Get the viewport region for 3D rendering
        updateViewportRegion();

        // Update viewport focus based on mouse position
        updateViewportFocus();

        // Render viewport gizmo BEFORE focus indicator - always render regardless of focus
        if (show_viewport_gizmo_ && viewport_size_.x > 0 && viewport_size_.y > 0) {
            if (rendering_manager) {
                auto* engine = rendering_manager->getRenderingEngine();
                if (engine) {
                    const auto& viewport = viewer_->getViewport();
                    glm::mat3 camera_rotation = viewport.getRotationMatrix();

                    engine->renderViewportGizmo(
                        camera_rotation,
                        glm::vec2(viewport_pos_.x, viewport_pos_.y),
                        glm::vec2(viewport_size_.x, viewport_size_.y));
                }
            }
        }

        // End frame
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Clean up GL state after ImGui rendering (ImGui can leave VAO/shader bindings corrupted)
        glBindVertexArray(0);
        glUseProgram(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        // Clear any errors ImGui might have generated
        while (glGetError() != GL_NO_ERROR) {}

        // Update and Render additional Platform Windows (for multi-viewport)
        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);

            // Clean up GL state after multi-viewport rendering too
            glBindVertexArray(0);
            glUseProgram(0);
            glBindTexture(GL_TEXTURE_2D, 0);
            while (glGetError() != GL_NO_ERROR) {}
        }
    }

    void GuiManager::updateViewportRegion() {
        const ImGuiViewport* main_viewport = ImGui::GetMainViewport();

        // Start with full window
        float left = 0;
        float top = 0;
        float right = main_viewport->WorkSize.x;
        float bottom = main_viewport->WorkSize.y;

        // Find our docked windows and calculate the remaining space
        ImGuiWindow* rendering_window = ImGui::FindWindowByName("Rendering");
        ImGuiWindow* training_window = ImGui::FindWindowByName("Training");
        ImGuiWindow* scene_window = ImGui::FindWindowByName("Scene");

        // Check both left-side panels
        if (rendering_window && rendering_window->DockNode && rendering_window->Active) {
            float panel_right = rendering_window->Pos.x + rendering_window->Size.x - main_viewport->WorkPos.x;
            left = std::max(left, panel_right);
        }

        if (training_window && training_window->DockNode && training_window->Active) {
            float panel_right = training_window->Pos.x + training_window->Size.x - main_viewport->WorkPos.x;
            left = std::max(left, panel_right);
        }

        if (scene_window && scene_window->DockNode && scene_window->Active) {
            // Scene panel is on the right
            float panel_left = scene_window->Pos.x - main_viewport->WorkPos.x;
            right = std::min(right, panel_left);
        }

        // Store in actual window coordinates (not relative to work area)
        // WorkPos accounts for the menu bar offset
        viewport_pos_ = ImVec2(left + main_viewport->WorkPos.x, top + main_viewport->WorkPos.y);
        viewport_size_ = ImVec2(right - left, bottom - top);
    }

    void GuiManager::updateViewportFocus() {
        // Viewport has focus unless actively using a GUI widget
        viewport_has_focus_ = !ImGui::IsAnyItemActive();
    }

    ImVec2 GuiManager::getViewportPos() const {
        return viewport_pos_;
    }

    ImVec2 GuiManager::getViewportSize() const {
        return viewport_size_;
    }

    bool GuiManager::isMouseInViewport() const {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        return mouse_pos.x >= viewport_pos_.x &&
               mouse_pos.y >= viewport_pos_.y &&
               mouse_pos.x < viewport_pos_.x + viewport_size_.x &&
               mouse_pos.y < viewport_pos_.y + viewport_size_.y;
    }

    bool GuiManager::isViewportFocused() const {
        return viewport_has_focus_;
    }

    bool GuiManager::isPositionInViewport(double x, double y) const {
        const ImGuiViewport* main_viewport = ImGui::GetMainViewport();

        // Convert to window-relative coordinates
        float rel_x = static_cast<float>(x) - main_viewport->WorkPos.x;
        float rel_y = static_cast<float>(y) - main_viewport->WorkPos.y;

        // Check if within viewport bounds
        return (rel_x >= viewport_pos_.x &&
                rel_x < viewport_pos_.x + viewport_size_.x &&
                rel_y >= viewport_pos_.y &&
                rel_y < viewport_pos_.y + viewport_size_.y);
    }

    void GuiManager::renderSpeedOverlay() {
        // Check if overlay should be hidden
        if (speed_overlay_visible_) {
            auto now = std::chrono::steady_clock::now();
            if (now - speed_overlay_start_time_ >= speed_overlay_duration_) {
                speed_overlay_visible_ = false;
            }
        } else {
            return;
        }

        // Position overlay centered in viewport, below all possible toolbars
        constexpr float OVERLAY_WIDTH = 300.0f;
        constexpr float OVERLAY_HEIGHT = 80.0f;
        constexpr float TOOLBAR_CLEARANCE = 100.0f;

        const ImVec2 overlay_pos(
            viewport_pos_.x + (viewport_size_.x - OVERLAY_WIDTH) * 0.5f,
            viewport_pos_.y + TOOLBAR_CLEARANCE);

        ImGui::SetNextWindowPos(overlay_pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(OVERLAY_WIDTH, OVERLAY_HEIGHT), ImGuiCond_Always);

        // Window flags to make it non-interactive and styled nicely
        ImGuiWindowFlags overlay_flags =
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoInputs |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoBringToFrontOnFocus;

        // Apply semi-transparent background
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1.0f, 1.0f, 1.0f, 0.3f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);

        if (ImGui::Begin("##SpeedOverlay", nullptr, overlay_flags)) {
            // Calculate fade effect based on remaining time
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - speed_overlay_start_time_);
            auto remaining = speed_overlay_duration_ - elapsed;

            float fade_alpha = 1.0f;
            if (remaining < std::chrono::milliseconds(500)) {
                // Fade out in the last 500ms
                fade_alpha = static_cast<float>(remaining.count()) / 500.0f;
            }

            // Center the text
            ImVec2 window_size = ImGui::GetWindowSize();

            // Speed text
            std::string speed_text = std::format("WASD Speed: {:.0f}", current_speed_);
            ImVec2 speed_text_size = ImGui::CalcTextSize(speed_text.c_str());
            ImGui::SetCursorPos(ImVec2(
                (window_size.x - speed_text_size.x) * 0.5f,
                window_size.y * 0.3f));

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, fade_alpha));
            ImGui::Text("%s", speed_text.c_str());
            ImGui::PopStyleColor();

            // Max speed text
            std::string max_text = std::format("Max: {:.0f}", max_speed_);
            ImVec2 max_text_size = ImGui::CalcTextSize(max_text.c_str());
            ImGui::SetCursorPos(ImVec2(
                (window_size.x - max_text_size.x) * 0.5f,
                window_size.y * 0.6f));

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.8f, fade_alpha * 0.8f));
            ImGui::Text("%s", max_text.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::End();

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(2);
    }

    void GuiManager::showSpeedOverlay(float current_speed, float max_speed) {
        current_speed_ = current_speed;
        max_speed_ = max_speed;
        speed_overlay_visible_ = true;
        speed_overlay_start_time_ = std::chrono::steady_clock::now();
    }

    void GuiManager::showZoomSpeedOverlay(const float zoom_speed, const float max_zoom_speed) {
        zoom_speed_ = zoom_speed;
        max_zoom_speed_ = max_zoom_speed;
        zoom_speed_overlay_visible_ = true;
        zoom_speed_overlay_start_time_ = std::chrono::steady_clock::now();
    }

    void GuiManager::renderZoomSpeedOverlay() {
        if (zoom_speed_overlay_visible_) {
            const auto now = std::chrono::steady_clock::now();
            if (now - zoom_speed_overlay_start_time_ >= speed_overlay_duration_) {
                zoom_speed_overlay_visible_ = false;
            }
        } else {
            return;
        }

        constexpr float OVERLAY_WIDTH = 300.0f;
        constexpr float OVERLAY_HEIGHT = 80.0f;
        constexpr float FADE_DURATION = 500.0f;
        constexpr float TOOLBAR_CLEARANCE = 100.0f;
        constexpr float WASD_OVERLAY_HEIGHT = 90.0f;

        const float y_offset = speed_overlay_visible_ ? (TOOLBAR_CLEARANCE + WASD_OVERLAY_HEIGHT) : TOOLBAR_CLEARANCE;
        const ImVec2 overlay_pos(
            viewport_pos_.x + (viewport_size_.x - OVERLAY_WIDTH) * 0.5f,
            viewport_pos_.y + y_offset);

        ImGui::SetNextWindowPos(overlay_pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(OVERLAY_WIDTH, OVERLAY_HEIGHT), ImGuiCond_Always);

        constexpr ImGuiWindowFlags kOverlayFlags =
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoBringToFrontOnFocus;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1.0f, 1.0f, 1.0f, 0.3f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);

        if (ImGui::Begin("##ZoomSpeedOverlay", nullptr, kOverlayFlags)) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - zoom_speed_overlay_start_time_);
            const auto remaining = speed_overlay_duration_ - elapsed;

            float fade_alpha = 1.0f;
            if (remaining < std::chrono::milliseconds(500)) {
                fade_alpha = static_cast<float>(remaining.count()) / FADE_DURATION;
            }

            const ImVec2 window_size = ImGui::GetWindowSize();

            // Display as 1-100 scale (internal is 0.1-10)
            const std::string speed_text = std::format("Zoom Speed: {:.0f}", zoom_speed_ * 10.0f);
            const ImVec2 speed_text_size = ImGui::CalcTextSize(speed_text.c_str());
            ImGui::SetCursorPos(ImVec2((window_size.x - speed_text_size.x) * 0.5f, window_size.y * 0.3f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, fade_alpha));
            ImGui::Text("%s", speed_text.c_str());
            ImGui::PopStyleColor();

            const std::string max_text = std::format("Max: {:.0f}", max_zoom_speed_ * 10.0f);
            const ImVec2 max_text_size = ImGui::CalcTextSize(max_text.c_str());
            ImGui::SetCursorPos(ImVec2((window_size.x - max_text_size.x) * 0.5f, window_size.y * 0.6f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.8f, fade_alpha * 0.8f));
            ImGui::Text("%s", max_text.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::End();

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(2);
    }

    void GuiManager::setupEventHandlers() {
        using namespace lfs::core::events;

        // Handle window visibility
        cmd::ShowWindow::when([this](const auto& e) {
            showWindow(e.window_name, e.show);
        });

        // Switch to translate tool when node is selected/deselected
        ui::NodeSelected::when([this](const auto&) {
            if (gizmo_toolbar_state_.current_tool == panels::ToolMode::Selection ||
                gizmo_toolbar_state_.current_tool == panels::ToolMode::Brush) {
                gizmo_toolbar_state_.current_tool = panels::ToolMode::Translate;
                gizmo_toolbar_state_.current_operation = ImGuizmo::TRANSLATE;
            }
        });

        ui::NodeDeselected::when([this](const auto&) {
            gizmo_toolbar_state_.current_tool = panels::ToolMode::Translate;
            gizmo_toolbar_state_.current_operation = ImGuizmo::TRANSLATE;
        });

        // Handle speed change events
        ui::SpeedChanged::when([this](const auto& e) {
            showSpeedOverlay(e.current_speed, e.max_speed);
        });

        ui::ZoomSpeedChanged::when([this](const auto& e) {
            showZoomSpeedOverlay(e.zoom_speed, e.max_zoom_speed);
        });

        lfs::core::events::tools::SetToolbarTool::when([this](const auto& e) {
            gizmo_toolbar_state_.current_tool = static_cast<panels::ToolMode>(e.tool_mode);
            if (gizmo_toolbar_state_.current_tool == panels::ToolMode::CropBox) {
                gizmo_toolbar_state_.cropbox_operation = panels::CropBoxOperation::Bounds;
                crop_gizmo_operation_ = ImGuizmo::BOUNDS;
            }
        });

        // Handle Enter key to apply crop box
        cmd::ApplyCropBox::when([this](const auto&) {
            if (gizmo_toolbar_state_.current_tool != panels::ToolMode::CropBox) {
                return;
            }
            auto* render_manager = viewer_->getRenderingManager();
            if (!render_manager) {
                return;
            }
            const auto& settings = render_manager->getSettings();

            lfs::geometry::BoundingBox crop_box;
            crop_box.setBounds(settings.crop_min, settings.crop_max);
            crop_box.setworld2BBox(settings.crop_transform.inv());
            cmd::CropPLY{.crop_box = crop_box, .inverse = settings.crop_inverse}.emit();

            // Refit crop box to remaining visible gaussians
            cmd::FitCropBoxToScene{.use_percentile = true}.emit();
        });

        // Handle Ctrl+T to toggle crop inverse mode
        cmd::ToggleCropInverse::when([this](const auto&) {
            if (gizmo_toolbar_state_.current_tool != panels::ToolMode::CropBox) {
                return;
            }
            auto* render_manager = viewer_->getRenderingManager();
            if (!render_manager) {
                return;
            }
            auto settings = render_manager->getSettings();

            // Capture state before toggle
            const command::CropBoxState old_state{
                .crop_min = settings.crop_min,
                .crop_max = settings.crop_max,
                .crop_transform = settings.crop_transform,
                .crop_inverse = settings.crop_inverse
            };

            // Toggle crop inverse
            settings.crop_inverse = !settings.crop_inverse;
            render_manager->updateSettings(settings);

            // Capture state after toggle
            const command::CropBoxState new_state{
                .crop_min = settings.crop_min,
                .crop_max = settings.crop_max,
                .crop_transform = settings.crop_transform,
                .crop_inverse = settings.crop_inverse
            };

            // Create undo command
            auto cmd = std::make_unique<command::CropBoxCommand>(
                render_manager, old_state, new_state);
            viewer_->getCommandHistory().execute(std::move(cmd));

            LOG_INFO("Crop inverse mode: {}", settings.crop_inverse);
        });

        // Handle Save PLY As command
        cmd::SavePLYAs::when([this](const auto& e) {
            save_ply_node_name_ = e.name;
            save_ply_path_ = (std::filesystem::current_path() / (e.name + ".ply")).string();
            show_save_ply_dialog_ = true;
        });

        // Cycle: normal -> center markers -> rings -> normal
        cmd::CycleSelectionVisualization::when([this](const auto&) {
            if (gizmo_toolbar_state_.current_tool != panels::ToolMode::Selection) return;
            auto* const rm = viewer_->getRenderingManager();
            if (!rm) return;

            auto settings = rm->getSettings();
            const bool centers = settings.show_center_markers;
            const bool rings = settings.show_rings;

            settings.show_center_markers = !centers && !rings;
            settings.show_rings = centers && !rings;
            rm->updateSettings(settings);
        });
    }

    void GuiManager::setSelectionSubMode(panels::SelectionSubMode mode) {
        if (gizmo_toolbar_state_.current_tool == panels::ToolMode::Selection) {
            gizmo_toolbar_state_.selection_mode = mode;
        }
    }

    void GuiManager::applyDefaultStyle() {
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
        style.WindowPadding = ImVec2(6.0f, 6.0f);
        style.WindowRounding = 6.0f;
        style.WindowBorderSize = 0.0f;
        style.FrameRounding = 2.0f;

        ImGui::StyleColorsLight();
    }

    void GuiManager::showWindow(const std::string& name, bool show) {
        window_states_[name] = show;
    }

    void GuiManager::toggleWindow(const std::string& name) {
        window_states_[name] = !window_states_[name];
    }

    bool GuiManager::wantsInput() const {
        ImGuiIO& io = ImGui::GetIO();
        return io.WantCaptureMouse || io.WantCaptureKeyboard;
    }

    bool GuiManager::isAnyWindowActive() const {
        return ImGui::IsAnyItemActive() ||
               ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) ||
               ImGui::GetIO().WantCaptureMouse ||
               ImGui::GetIO().WantCaptureKeyboard;
    }

    void GuiManager::setFileSelectedCallback(std::function<void(const std::filesystem::path&, bool)> callback) {
        if (file_browser_) {
            file_browser_->setOnFileSelected(callback);
        }
    }

    void GuiManager::handleProjectChangedDialogCallback(std::function<void(bool)> callback) {
        if (project_changed_dialog_box_) {
            project_changed_dialog_box_->setOnDialogClose(callback);
        }
    }

    void GuiManager::renderCropBoxGizmo(const UIContext& ctx) {
        auto* render_manager = ctx.viewer->getRenderingManager();
        if (!render_manager)
            return;

        auto settings = render_manager->getSettings();

        // Only draw gizmo if crop box is visible
        if (!settings.show_crop_box)
            return;

        // Get camera matrices - use viewport_size_ for aspect ratio to match bbox renderer
        auto& viewport = ctx.viewer->getViewport();
        const glm::mat4 view = viewport.getViewMatrix();

        // Create projection with correct aspect ratio matching the viewport region
        const float aspect = viewport_size_.x / viewport_size_.y;
        const float fov_rad = glm::radians(settings.fov);
        const glm::mat4 projection = glm::perspective(fov_rad, aspect,
            lfs::rendering::DEFAULT_NEAR_PLANE, lfs::rendering::DEFAULT_FAR_PLANE);

        // Build gizmo matrix: T * R * S where S encodes the box size
        // ImGuizmo BOUNDS mode expects unit cube [-0.5, 0.5] with size in matrix scale
        const glm::vec3 translation = settings.crop_transform.getTranslation();
        const glm::mat3 rotation3x3 = settings.crop_transform.getRotationMat();
        const glm::vec3 original_size = settings.crop_max - settings.crop_min;
        const glm::vec3 original_center = (settings.crop_min + settings.crop_max) * 0.5f;

        // Build T * R * S matrix
        glm::mat4 gizmo_matrix = glm::translate(glm::mat4(1.0f), translation + rotation3x3 * original_center);
        gizmo_matrix = gizmo_matrix * glm::mat4(rotation3x3);
        gizmo_matrix = glm::scale(gizmo_matrix, original_size);

        ImGuizmo::SetOrthographic(false);

        // viewport_pos_ already includes WorkPos offset, use directly
        ImGuizmo::SetRect(viewport_pos_.x, viewport_pos_.y, viewport_size_.x, viewport_size_.y);

        // Disable view angle culling - make everything visible before drag
        ImGuizmo::SetAxisLimit(0.0001f);
        ImGuizmo::SetPlaneLimit(0.0001f);

        // Axis mask depends on operation mode
        if (crop_gizmo_operation_ == ImGuizmo::BOUNDS) {
            // BOUNDS mode: always use full mask for visibility
            ImGuizmo::SetAxisMask(true, true, true);
        } else {
            // TRANSLATE/ROTATE/SCALE: dynamic masking based on hover state
            static bool hovered_axis = false;

            if (!ImGuizmo::IsUsing()) {
                hovered_axis = ImGuizmo::IsOver(ImGuizmo::TRANSLATE_X) ||
                              ImGuizmo::IsOver(ImGuizmo::TRANSLATE_Y) ||
                              ImGuizmo::IsOver(ImGuizmo::TRANSLATE_Z);
                ImGuizmo::SetAxisMask(false, false, false);
            } else {
                if (hovered_axis) {
                    ImGuizmo::SetAxisMask(true, true, true);
                } else {
                    ImGuizmo::SetAxisMask(false, false, false);
                }
            }
        }

        // Clip ImGuizmo rendering to viewport so it doesn't draw over GUI panels
        ImDrawList* overlay_drawlist = ImGui::GetForegroundDrawList();
        ImVec2 clip_min(viewport_pos_.x, viewport_pos_.y);
        ImVec2 clip_max(clip_min.x + viewport_size_.x, clip_min.y + viewport_size_.y);
        overlay_drawlist->PushClipRect(clip_min, clip_max, true);

        // Set drawlist after pushing clip rect
        ImGuizmo::SetDrawlist(overlay_drawlist);

        glm::mat4 deltaMatrix;
        constexpr float localBounds[6] = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };

        // Force correct mode based on operation:
        // - TRANSLATE: always WORLD coordinates
        // - ROTATE/SCALE/BOUNDS: always LOCAL coordinates
        ImGuizmo::MODE gizmo_mode = (crop_gizmo_operation_ == ImGuizmo::TRANSLATE)
                                    ? ImGuizmo::WORLD
                                    : ImGuizmo::LOCAL;

        bool gizmo_changed = ImGuizmo::Manipulate(glm::value_ptr(view),
                                                   glm::value_ptr(projection),
                                                   crop_gizmo_operation_,
                                                   gizmo_mode,
                                                   glm::value_ptr(gizmo_matrix),
                                                   glm::value_ptr(deltaMatrix),
                                                   nullptr,  // snap
                                                   crop_gizmo_operation_ == ImGuizmo::BOUNDS ? localBounds : nullptr);

        // Track gizmo usage for undo/redo
        const bool is_using = ImGuizmo::IsUsing();

        // Capture state when manipulation starts
        if (is_using && !cropbox_gizmo_active_) {
            cropbox_gizmo_active_ = true;
            cropbox_state_before_drag_ = command::CropBoxState{
                .crop_min = settings.crop_min,
                .crop_max = settings.crop_max,
                .crop_transform = settings.crop_transform,
                .crop_inverse = settings.crop_inverse
            };
        }

        if (gizmo_changed) {
            // Extract transform from gizmo matrix (T * R * S)
            float matrixTranslation[3], matrixRotation[3], matrixScale[3];
            ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(gizmo_matrix),
                                                  matrixTranslation,
                                                  matrixRotation,
                                                  matrixScale);

            const float rad_x = glm::radians(matrixRotation[0]);
            const float rad_y = glm::radians(matrixRotation[1]);
            const float rad_z = glm::radians(matrixRotation[2]);

            glm::vec3 new_size(matrixScale[0], matrixScale[1], matrixScale[2]);
            new_size = glm::max(new_size, glm::vec3(0.001f));

            const glm::mat3 new_rotation = glm::mat3(
                glm::rotate(glm::mat4(1.0f), rad_x, glm::vec3(1, 0, 0)) *
                glm::rotate(glm::mat4(1.0f), rad_y, glm::vec3(0, 1, 0)) *
                glm::rotate(glm::mat4(1.0f), rad_z, glm::vec3(0, 0, 1)));

            const glm::vec3 world_center(matrixTranslation[0], matrixTranslation[1], matrixTranslation[2]);

            if (crop_gizmo_operation_ == ImGuizmo::TRANSLATE) {
                const glm::vec3 transform_trans = world_center - new_rotation * original_center;
                settings.crop_transform = lfs::geometry::EuclideanTransform(
                    rad_x, rad_y, rad_z,
                    transform_trans.x, transform_trans.y, transform_trans.z);
            } else if (crop_gizmo_operation_ == ImGuizmo::SCALE ||
                       crop_gizmo_operation_ == ImGuizmo::BOUNDS) {
                const glm::vec3 new_half = new_size * 0.5f;
                settings.crop_min = -new_half;
                settings.crop_max = new_half;
                const glm::vec3 new_center = (settings.crop_min + settings.crop_max) * 0.5f;
                const glm::vec3 transform_trans = world_center - new_rotation * new_center;
                settings.crop_transform = lfs::geometry::EuclideanTransform(
                    rad_x, rad_y, rad_z,
                    transform_trans.x, transform_trans.y, transform_trans.z);
            } else if (crop_gizmo_operation_ == ImGuizmo::ROTATE) {
                const glm::vec3 transform_trans = world_center - new_rotation * original_center;
                settings.crop_transform = lfs::geometry::EuclideanTransform(
                    rad_x, rad_y, rad_z,
                    transform_trans.x, transform_trans.y, transform_trans.z);
            }

            // Always update settings for visual feedback during dragging
            render_manager->updateSettings(settings);
        }

        // Create undo command when manipulation ends
        if (!is_using && cropbox_gizmo_active_) {
            cropbox_gizmo_active_ = false;

            if (cropbox_state_before_drag_.has_value()) {
                const command::CropBoxState new_state{
                    .crop_min = settings.crop_min,
                    .crop_max = settings.crop_max,
                    .crop_transform = settings.crop_transform,
                    .crop_inverse = settings.crop_inverse
                };

                auto cmd = std::make_unique<command::CropBoxCommand>(
                    render_manager, *cropbox_state_before_drag_, new_state);
                viewer_->getCommandHistory().execute(std::move(cmd));

                cropbox_state_before_drag_.reset();

                using namespace lfs::core::events;
                ui::CropBoxChanged{
                    .min_bounds = settings.crop_min,
                    .max_bounds = settings.crop_max,
                    .enabled = settings.use_crop_box}
                    .emit();
            }
        }

        // Restore clip rect after ImGuizmo rendering
        overlay_drawlist->PopClipRect();
    }

    void GuiManager::renderNodeTransformGizmo(const UIContext& ctx) {
        // Only show gizmo if enabled
        if (!show_node_gizmo_)
            return;

        auto* scene_manager = ctx.viewer->getSceneManager();
        if (!scene_manager || !scene_manager->hasSelectedNode())
            return;

        auto* render_manager = ctx.viewer->getRenderingManager();
        if (!render_manager)
            return;

        auto settings = render_manager->getSettings();

        // Get camera matrices
        auto& viewport = ctx.viewer->getViewport();
        glm::mat4 view = viewport.getViewMatrix();
        glm::mat4 projection = viewport.getProjectionMatrix(settings.fov);

        // Get current node transform and centroid
        glm::vec3 centroid = scene_manager->getSelectedNodeCentroid();
        glm::mat4 node_transform = scene_manager->getSelectedNodeTransform();

        // The gizmo should be positioned at centroid, with the node's rotation/scale applied
        // Node transform stores: T * R * S relative to origin
        // We need gizmo at: centroid + translation, with rotation/scale from node_transform

        // Extract translation from node transform
        glm::vec3 translation(node_transform[3]);

        // Extract rotation and scale (upper 3x3)
        glm::mat3 rotation_scale(node_transform);

        // Build gizmo matrix: translate to centroid + node_translation, apply rotation/scale
        glm::mat4 gizmo_matrix = glm::mat4(1.0f);

        // Set position at centroid + translation
        glm::vec3 gizmo_position = centroid + translation;
        gizmo_matrix[3] = glm::vec4(gizmo_position, 1.0f);

        // Apply rotation/scale to the gizmo
        gizmo_matrix[0] = glm::vec4(rotation_scale[0], 0.0f);
        gizmo_matrix[1] = glm::vec4(rotation_scale[1], 0.0f);
        gizmo_matrix[2] = glm::vec4(rotation_scale[2], 0.0f);

        ImGuizmo::SetOrthographic(false);

        // viewport_pos_ already includes WorkPos offset, use directly
        ImGuizmo::SetRect(viewport_pos_.x, viewport_pos_.y, viewport_size_.x, viewport_size_.y);

        // Clip ImGuizmo rendering to viewport so it doesn't draw over GUI panels
        ImDrawList* overlay_drawlist = ImGui::GetForegroundDrawList();
        ImVec2 clip_min(viewport_pos_.x, viewport_pos_.y);
        ImVec2 clip_max(clip_min.x + viewport_size_.x, clip_min.y + viewport_size_.y);
        overlay_drawlist->PushClipRect(clip_min, clip_max, true);

        // Set drawlist after pushing clip rect
        ImGuizmo::SetDrawlist(overlay_drawlist);

        // Use LOCAL mode for rotation/scale so they rotate around the gizmo center
        ImGuizmo::MODE gizmo_mode = (node_gizmo_operation_ == ImGuizmo::TRANSLATE)
                                        ? ImGuizmo::WORLD
                                        : ImGuizmo::LOCAL;

        // Draw the gizmo
        glm::mat4 deltaMatrix;
        bool gizmo_changed = ImGuizmo::Manipulate(
            glm::value_ptr(view),
            glm::value_ptr(projection),
            node_gizmo_operation_,
            gizmo_mode,
            glm::value_ptr(gizmo_matrix),
            glm::value_ptr(deltaMatrix),
            nullptr);  // snap

        if (gizmo_changed) {
            // Extract new position
            glm::vec3 new_gizmo_position = glm::vec3(gizmo_matrix[3]);

            // The node's translation offset is new_gizmo_position - centroid
            glm::vec3 new_translation = new_gizmo_position - centroid;

            // Build new node transform with the updated values
            glm::mat4 new_transform = gizmo_matrix;
            new_transform[3] = glm::vec4(new_translation, 1.0f);

            // Update the node's full transform
            scene_manager->setSelectedNodeTransform(new_transform);
        }

        // Restore clip rect after ImGuizmo rendering
        overlay_drawlist->PopClipRect();
    }

} // namespace lfs::vis::gui
