/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_controller.hpp"
#include "core_new/logger.hpp"
#include "gui/gui_manager.hpp"
#include "loader_new/loader.hpp"
#include "rendering/rendering_manager.hpp"
#include "tools/align_tool.hpp"
#include "tools/brush_tool.hpp"
#include "tools/selection_tool.hpp"
#include "tools/tool_base.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <format>
#include <imgui.h>
#include <ImGuizmo.h>

namespace lfs::vis {

    using namespace lfs::core::events;

    InputController* InputController::instance_ = nullptr;

    InputController::InputController(GLFWwindow* window, Viewport& viewport)
        : window_(window),
          viewport_(viewport) {
        // Subscribe to GoToCamView events
        cmd::GoToCamView::when([this](const auto& e) {
            handleGoToCamView(e);
        });
        // Subscribe to WindowFocusLost to reset states
        internal::WindowFocusLost::when([this](const auto&) {
            drag_mode_ = DragMode::None;
            std::fill(std::begin(keys_wasd_), std::end(keys_wasd_), false);
            hovered_camera_id_ = -1;
        });
        // Subscribe to GimbalLock events
        cmd::ToggleGimbalLock::when([this](const cmd::ToggleGimbalLock& e) {
            gimbal_locked = e.locked;
        });

        LOG_DEBUG("InputController created");
    }

    InputController::~InputController() {
        if (instance_ == this) {
            instance_ = nullptr;
        }

        // Clean up cursor resources
        if (resize_cursor_) {
            glfwDestroyCursor(resize_cursor_);
            resize_cursor_ = nullptr;
        }
        if (hand_cursor_) {
            glfwDestroyCursor(hand_cursor_);
            hand_cursor_ = nullptr;
        }

        // Reset cursor to default before destruction
        if (window_ && current_cursor_ != CursorType::Default) {
            glfwSetCursor(window_, nullptr);
        }
    }

    void InputController::initialize() {
        // Must be called after ImGui_ImplGlfw_InitForOpenGL
        instance_ = this;

        // Store ImGui's callbacks so we can chain to them
        imgui_callbacks_.mouse_button = glfwSetMouseButtonCallback(window_, mouseButtonCallback);
        imgui_callbacks_.cursor_pos = glfwSetCursorPosCallback(window_, cursorPosCallback);
        imgui_callbacks_.scroll = glfwSetScrollCallback(window_, scrollCallback);
        imgui_callbacks_.key = glfwSetKeyCallback(window_, keyCallback);
        imgui_callbacks_.drop = glfwSetDropCallback(window_, dropCallback);
        imgui_callbacks_.focus = glfwSetWindowFocusCallback(window_, windowFocusCallback);

        // Get initial mouse position
        double x, y;
        glfwGetCursorPos(window_, &x, &y);
        last_mouse_pos_ = {x, y};

        // Initialize frame timer
        last_frame_time_ = std::chrono::high_resolution_clock::now();

        // Create the cursors once at initialization
        resize_cursor_ = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
        hand_cursor_ = glfwCreateStandardCursor(GLFW_HAND_CURSOR);

        LOG_DEBUG("InputController initialized - callbacks set");
    }

    // Static callbacks - chain to ImGui then handle ourselves
    void InputController::mouseButtonCallback(GLFWwindow* w, int button, int action, int mods) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.mouse_button) {
            instance_->imgui_callbacks_.mouse_button(w, button, action, mods);
        }

        // Then handle for camera
        if (instance_) {
            double x, y;
            glfwGetCursorPos(w, &x, &y);
            instance_->handleMouseButton(button, action, x, y);
        }
    }

    void InputController::cursorPosCallback(GLFWwindow* w, double x, double y) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.cursor_pos) {
            instance_->imgui_callbacks_.cursor_pos(w, x, y);
        }

        // Then handle for camera
        if (instance_) {
            instance_->handleMouseMove(x, y);
        }
    }

    void InputController::scrollCallback(GLFWwindow* w, double xoff, double yoff) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.scroll) {
            instance_->imgui_callbacks_.scroll(w, xoff, yoff);
        }

        // Then handle for camera
        if (instance_) {
            instance_->handleScroll(xoff, yoff);
        }
    }

    void InputController::keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.key) {
            instance_->imgui_callbacks_.key(w, key, scancode, action, mods);
        }

        // Then handle for camera
        if (instance_) {
            instance_->handleKey(key, action, mods);
        }
    }

    void InputController::dropCallback(GLFWwindow* w, int count, const char** paths) {
        // Let ImGui handle first (though it probably doesn't use this)
        if (instance_ && instance_->imgui_callbacks_.drop) {
            instance_->imgui_callbacks_.drop(w, count, paths);
        }

        // Then handle file drops
        if (instance_) {
            std::vector<std::string> files(paths, paths + count);
            instance_->handleFileDrop(files);
        }
    }

    void InputController::windowFocusCallback(GLFWwindow* w, int focused) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.focus) {
            instance_->imgui_callbacks_.focus(w, focused);
        }

        // Reset states on focus loss
        if (!focused) {
            if (instance_) {
                instance_->drag_mode_ = DragMode::None;
                std::fill(std::begin(instance_->keys_wasd_),
                          std::end(instance_->keys_wasd_), false);
                instance_->hovered_camera_id_ = -1; // Reset hovered camera

                // Reset cursor to default when losing focus
                if (instance_->current_cursor_ != CursorType::Default) {
                    glfwSetCursor(instance_->window_, nullptr);
                    instance_->current_cursor_ = CursorType::Default;
                }
            }
            lfs::core::events::internal::WindowFocusLost{}.emit();
            LOG_DEBUG("Window lost focus - input states reset");
        } else {
            LOG_DEBUG("Window gained focus");
        }
    }

    bool InputController::isNearSplitter(double x) const {
        if (!rendering_manager_ || rendering_manager_->getSettings().split_view_mode == SplitViewMode::Disabled) {
            return false;
        }

        float split_pos = rendering_manager_->getSettings().split_position;
        float split_x = viewport_bounds_.x + viewport_bounds_.width * split_pos;

        // Increase the hit area to 10 pixels for easier grabbing
        return std::abs(x - split_x) < 10.0;
    }

    // Core handlers
    void InputController::handleMouseButton(int button, int action, double x, double y) {
        // Check for splitter drag FIRST
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            // Check for double-click on camera frustum
            auto now = std::chrono::steady_clock::now();
            auto time_since_last = std::chrono::duration<double>(now - last_click_time_).count();
            double dist = glm::length(glm::dvec2(x, y) - last_click_pos_);

            constexpr double DOUBLE_CLICK_TIME = 0.5;
            constexpr double DOUBLE_CLICK_DISTANCE = 10.0;

            bool is_double_click = (time_since_last < DOUBLE_CLICK_TIME &&
                                    dist < DOUBLE_CLICK_DISTANCE);

            // If we have a hovered camera, check for double-click
            if (hovered_camera_id_ >= 0) {
                if (is_double_click && hovered_camera_id_ == last_clicked_camera_id_) {
                    LOG_DEBUG("Double-clicked on camera ID: {}", hovered_camera_id_);
                    cmd::GoToCamView{.cam_id = hovered_camera_id_}.emit();

                    // Reset click tracking to prevent triple-click
                    last_click_time_ = std::chrono::steady_clock::time_point();
                    last_click_pos_ = {-1000, -1000}; // Far away position
                    last_clicked_camera_id_ = -1;
                    return;
                }
                // First click on a camera - record it
                last_click_time_ = now;
                last_click_pos_ = {x, y};
                last_clicked_camera_id_ = hovered_camera_id_;
            } else {
                last_click_time_ = std::chrono::steady_clock::time_point();
                last_click_pos_ = {-1000, -1000};
                last_clicked_camera_id_ = -1;
            }

            // Check for splitter drag
            if (isNearSplitter(x) && rendering_manager_) {
                drag_mode_ = DragMode::Splitter;
                splitter_start_pos_ = rendering_manager_->getSettings().split_position;
                splitter_start_x_ = x;
                glfwSetCursor(window_, resize_cursor_);
                LOG_TRACE("Started splitter drag");
                return;
            }
        }

        if (action == GLFW_RELEASE && drag_mode_ == DragMode::Splitter) {
            drag_mode_ = DragMode::None;
            glfwSetCursor(window_, nullptr); // Reset cursor
            LOG_TRACE("Ended splitter drag");
            return;
        }

        const bool over_gui = ImGui::GetIO().WantCaptureMouse;

        if (brush_tool_ && brush_tool_->isEnabled() && tool_context_) {
            const int mods = getModifierKeys();
            if (!over_gui && brush_tool_->handleMouseButton(button, action, mods, x, y, *tool_context_)) {
                if (action == GLFW_PRESS) {
                    drag_mode_ = DragMode::Brush;
                } else if (action == GLFW_RELEASE && drag_mode_ == DragMode::Brush) {
                    drag_mode_ = DragMode::None;
                }
                return;
            }
        }

        if (selection_tool_ && selection_tool_->isEnabled() && tool_context_) {
            const int mods = getModifierKeys();
            if (!over_gui && selection_tool_->handleMouseButton(button, action, mods, x, y, *tool_context_)) {
                if (action == GLFW_PRESS) {
                    drag_mode_ = DragMode::Brush;
                } else if (action == GLFW_RELEASE && drag_mode_ == DragMode::Brush) {
                    drag_mode_ = DragMode::None;
                }
                return;
            }
        }

        if (align_tool_ && align_tool_->isEnabled() && tool_context_) {
            if (!over_gui && align_tool_->handleMouseButton(button, action, x, y, *tool_context_)) {
                return;
            }
        }

        if (action == GLFW_PRESS) {
            // Block if hovering over GUI window
            if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
                return;
            }

            // Block if ImGuizmo is being used or hovered
            if (ImGuizmo::IsOver() || ImGuizmo::IsUsing()) {
                return;
            }

            // Only handle camera controls if clicking within the viewport
            if (!isInViewport(x, y)) {
                return;
            }

            // Start camera interaction
            viewport_.camera.initScreenPos(glm::vec2(x, y));

            if (button == GLFW_MOUSE_BUTTON_LEFT) {
                // Double-click: set pivot and center camera on it
                const auto now = std::chrono::steady_clock::now();
                const double time_since_last = std::chrono::duration<double>(now - last_pivot_click_time_).count();
                const double dist = glm::length(glm::dvec2(x, y) - last_pivot_click_pos_);

                if (time_since_last < DOUBLE_CLICK_TIME && dist < DOUBLE_CLICK_DISTANCE) {
                    const glm::vec3 new_pivot = unprojectScreenPoint(x, y);
                    const float current_distance = glm::length(viewport_.camera.getPivot() - viewport_.camera.t);
                    const glm::vec3 forward = glm::normalize(viewport_.camera.R * glm::vec3(0, 0, 1));
                    viewport_.camera.t = new_pivot - forward * current_distance;
                    viewport_.camera.setPivot(new_pivot);
                    publishCameraMove();
                }

                last_pivot_click_time_ = now;
                last_pivot_click_pos_ = {x, y};
                drag_mode_ = DragMode::Pan;
            } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                drag_mode_ = DragMode::Rotate;
            } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
                drag_mode_ = DragMode::Orbit;
                const float current_time = static_cast<float>(glfwGetTime());
                viewport_.camera.startRotateAroundCenter(glm::vec2(x, y), current_time);
            }
        } else if (action == GLFW_RELEASE) {
            // Always handle our own releases if we were dragging
            bool was_dragging = false;
            if (button == GLFW_MOUSE_BUTTON_LEFT && drag_mode_ == DragMode::Pan) {
                drag_mode_ = DragMode::None;
                was_dragging = true;
                LOG_TRACE("Ended camera pan");
            } else if (button == GLFW_MOUSE_BUTTON_RIGHT && drag_mode_ == DragMode::Rotate) {
                drag_mode_ = DragMode::None;
                was_dragging = true;
                LOG_TRACE("Ended camera rotate");
            } else if (button == GLFW_MOUSE_BUTTON_MIDDLE && drag_mode_ == DragMode::Orbit) {
                viewport_.camera.endRotateAroundCenter();
                drag_mode_ = DragMode::None;
                was_dragging = true;
                LOG_TRACE("Ended camera orbit");
            }

            // Force publish on mouse release
            if (was_dragging) {
                ui::CameraMove{
                    .rotation = viewport_.getRotationMatrix(),
                    .translation = viewport_.getTranslation()}
                    .emit();
                onCameraMovementEnd();
            }
        }
    }

    void InputController::handleMouseMove(double x, double y) {
        // Track if we moved significantly
        glm::dvec2 current_pos{x, y};

        // Handle splitter dragging
        if (drag_mode_ == DragMode::Splitter && rendering_manager_) {
            double delta = x - splitter_start_x_;
            float new_pos = splitter_start_pos_ + static_cast<float>(delta / viewport_bounds_.width);

            // FIX: Allow dragging all the way to the edges - no margins!
            new_pos = std::clamp(new_pos, 0.0f, 1.0f);

            ui::SplitPositionChanged{.position = new_pos}.emit();
            last_mouse_pos_ = {x, y};
            return;
        }

        // Camera frustum hover detection with improved throttling
        if (rendering_manager_ &&
            rendering_manager_->getSettings().show_camera_frustums &&
            isInViewport(x, y) &&
            drag_mode_ == DragMode::None &&
            !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {

            // Additional throttling based on movement distance
            static glm::dvec2 last_pick_pos{-1, -1};
            static constexpr double MIN_PICK_DISTANCE = 3.0; // pixels

            bool should_pick = false;

            // Check if we moved enough from last pick position
            if (last_pick_pos.x < 0) {
                // First pick
                should_pick = true;
                last_pick_pos = current_pos;
            } else {
                double pick_distance = glm::length(current_pos - last_pick_pos);
                if (pick_distance >= MIN_PICK_DISTANCE) {
                    should_pick = true;
                    last_pick_pos = current_pos;
                }
            }

            if (should_pick) {
                auto result = rendering_manager_->pickCameraFrustum(glm::vec2(x, y));
                if (result >= 0) {
                    const int cam_id = result;
                    if (cam_id != hovered_camera_id_) {
                        hovered_camera_id_ = cam_id;
                        LOG_TRACE("Hovering over camera ID: {}", cam_id);

                        // Change cursor to hand
                        if (current_cursor_ != CursorType::Hand) {
                            glfwSetCursor(window_, hand_cursor_);
                            current_cursor_ = CursorType::Hand;
                        }
                    }
                } else {
                    // No camera under cursor
                    if (hovered_camera_id_ != -1) {
                        hovered_camera_id_ = -1;
                        LOG_TRACE("No longer hovering over camera");
                        if (current_cursor_ == CursorType::Hand) {
                            glfwSetCursor(window_, nullptr);
                            current_cursor_ = CursorType::Default;
                        }
                    }
                }
            }
        } else {
            // Not in conditions for camera picking
            if (hovered_camera_id_ != -1) {
                hovered_camera_id_ = -1;
                if (current_cursor_ == CursorType::Hand) {
                    glfwSetCursor(window_, nullptr);
                    current_cursor_ = CursorType::Default;
                }
            }
        }

        // Determine if we should show resize cursor for splitter
        bool should_show_resize = false;
        if (rendering_manager_ && rendering_manager_->getSettings().split_view_mode != SplitViewMode::Disabled) {
            should_show_resize = (drag_mode_ == DragMode::None &&
                                  isInViewport(x, y) &&
                                  isNearSplitter(x) &&
                                  !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow));
        }

        // Only call glfwSetCursor when state actually changes
        if (should_show_resize && current_cursor_ != CursorType::Resize) {
            glfwSetCursor(window_, resize_cursor_);
            current_cursor_ = CursorType::Resize;
        } else if (!should_show_resize && current_cursor_ == CursorType::Resize) {
            glfwSetCursor(window_, nullptr);
            current_cursor_ = CursorType::Default;
        }

        if (brush_tool_ && brush_tool_->isEnabled() && tool_context_) {
            if (drag_mode_ == DragMode::Brush) {
                brush_tool_->handleMouseMove(x, y, *tool_context_);
                last_mouse_pos_ = {x, y};
                return;
            } else if (brush_tool_->handleMouseMove(x, y, *tool_context_)) {
                last_mouse_pos_ = {x, y};
                return;
            }
        }

        if (selection_tool_ && selection_tool_->isEnabled() && tool_context_) {
            if (drag_mode_ == DragMode::Brush) {
                selection_tool_->handleMouseMove(x, y, *tool_context_);
                last_mouse_pos_ = {x, y};
                return;
            } else if (selection_tool_->handleMouseMove(x, y, *tool_context_)) {
                last_mouse_pos_ = {x, y};
                return;
            }
        }

        glm::vec2 pos(x, y);
        last_mouse_pos_ = current_pos;

        // Block camera dragging if ImGuizmo is being used
        if (ImGuizmo::IsUsing()) {
            return;
        }

        // Handle camera dragging
        if (drag_mode_ != DragMode::None &&
            drag_mode_ != DragMode::Gizmo &&
            drag_mode_ != DragMode::Splitter) {

            switch (drag_mode_) {
            case DragMode::Pan:
                viewport_.camera.translate(pos);
                break;
            case DragMode::Rotate:
                viewport_.camera.rotate(pos, gimbal_locked);
                break;
            case DragMode::Orbit: {
                float current_time = static_cast<float>(glfwGetTime());
                viewport_.camera.updateRotateAroundCenter(pos, current_time);
                break;
            }
            default:
                break;
            }
            // Signal continuous camera movement
            onCameraMovementStart();
            publishCameraMove();
        }
    }

    void InputController::handleScroll([[maybe_unused]] double xoff, double yoff) {
        if (brush_tool_ && brush_tool_->isEnabled() && tool_context_) {
            if (brush_tool_->handleScroll(xoff, yoff, getModifierKeys(), *tool_context_)) {
                return;
            }
        }

        if (selection_tool_ && selection_tool_->isEnabled() && tool_context_) {
            if (selection_tool_->handleScroll(xoff, yoff, getModifierKeys(), *tool_context_)) {
                return;
            }
        }

        if (drag_mode_ == DragMode::Gizmo || drag_mode_ == DragMode::Splitter) {
            return;
        }

        // Block scroll when hovering over GUI windows (panels)
        if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
            return;
        }

        if (!shouldCameraHandleInput())
            return;

        const float delta = static_cast<float>(yoff);
        if (std::abs(delta) < 0.01f)
            return;

        if (key_r_pressed_) {
            viewport_.camera.rotate_roll(delta);
        } else {
            viewport_.camera.zoom(delta);
        }

        onCameraMovementStart();
        publishCameraMove();
    }

    void InputController::handleKey(int key, int action, [[maybe_unused]] int mods) {
        // Track modifier keys (always, even if GUI has focus)
        if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) {
            key_ctrl_pressed_ = (action != GLFW_RELEASE);
        }
        if (key == GLFW_KEY_LEFT_ALT || key == GLFW_KEY_RIGHT_ALT) {
            key_alt_pressed_ = (action != GLFW_RELEASE);
        }
        if (key == GLFW_KEY_R) {
            key_r_pressed_ = (action != GLFW_RELEASE);
        }

        // T: cycle PLY, Ctrl+T: cycle visualization mode in Selection tool
        if (key == GLFW_KEY_T && action == GLFW_PRESS && !ImGui::IsAnyItemActive()) {
            const bool ctrl = glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                              glfwGetKey(window_, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
            if (ctrl && gui_manager_ &&
                gui_manager_->getCurrentToolMode() == gui::panels::ToolMode::Selection) {
                cmd::CycleSelectionVisualization{}.emit();
                return;
            }
            if (!ctrl) {
                cmd::CyclePLY{}.emit();
            }
            return;
        }


        if (key == GLFW_KEY_V && action == GLFW_PRESS && !ImGui::IsAnyItemActive()) {
            cmd::ToggleSplitView{}.emit();
            return;
        }

        if (key == GLFW_KEY_G && action == GLFW_PRESS && !ImGui::IsAnyItemActive()) {
            cmd::ToggleGTComparison{}.emit();
            LOG_DEBUG("Toggled GT comparison mode");
            return;
        }

        // H - reset camera to home position
        if (key == GLFW_KEY_H && action == GLFW_PRESS && !ImGui::IsAnyItemActive()) {
            viewport_.camera.resetToHome();
            publishCameraMove();
            return;
        }

        // B key cycles brush mode when brush tool is active
        if (key == GLFW_KEY_B && action == GLFW_PRESS && !ImGui::IsAnyItemActive()) {
            if (brush_tool_ && brush_tool_->isEnabled() && tool_context_) {
                if (brush_tool_->handleKeyPress(key, mods, *tool_context_)) {
                    return;
                }
            }
        }

        // Selection tool key handling (Enter/Escape for polygon mode)
        if (action == GLFW_PRESS && !ImGui::IsAnyItemActive()) {
            if (selection_tool_ && selection_tool_->isEnabled() && tool_context_) {
                if (selection_tool_->handleKeyPress(key, mods, *tool_context_)) {
                    return;
                }
            }
        }

        // DEL key - delete selected Gaussians
        if (key == GLFW_KEY_DELETE && action == GLFW_PRESS && !ImGui::IsAnyItemActive()) {
            cmd::DeleteSelected{}.emit();
            return;
        }

        if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS && !ImGui::IsAnyItemActive()) {
            if (!training_manager_) {
                LOG_WARN("Training manager is not set; cannot cycle camera view.");
                return;
            }
            int num_cams = training_manager_->getCamList().size();
            if (num_cams == 0) {
                return;
            }

            last_camview++;
            if (last_camview >= num_cams) {
                last_camview = 0; // Wrap to beginning
            }

            cmd::GoToCamView{
                .cam_id = last_camview}
                .emit();
            return;
        }

        if (key == GLFW_KEY_LEFT && action == GLFW_PRESS && !ImGui::IsAnyItemActive()) {
            if (!training_manager_) {
                LOG_WARN("Training manager is not set; cannot cycle camera view.");
                return;
            }
            int num_cams = training_manager_->getCamList().size();
            if (num_cams == 0) {
                return;
            }

            last_camview--;
            if (last_camview < 0) {
                last_camview = num_cams - 1; // Wrap to end
            }

            cmd::GoToCamView{
                .cam_id = last_camview}
                .emit();
            return;
        }

        // Speed control works even when GUI has focus
        if (key_ctrl_pressed_ && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            const bool shift_pressed = glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                                       glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
            if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) {
                if (shift_pressed) {
                    updateZoomSpeed(true);
                } else {
                    updateCameraSpeed(true);
                }
                return;
            }
            if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) {
                if (shift_pressed) {
                    updateZoomSpeed(false);
                } else {
                    updateCameraSpeed(false);
                }
                return;
            }
        }

        // Ctrl shortcuts
        if (key_ctrl_pressed_ && action == GLFW_PRESS && !ImGui::IsAnyItemActive()) {
            if (key == GLFW_KEY_Z) {
                cmd::Undo{}.emit();
                return;
            }
            if (key == GLFW_KEY_Y) {
                cmd::Redo{}.emit();
                return;
            }
            if (key == GLFW_KEY_I) {
                cmd::InvertSelection{}.emit();
                return;
            }
            if (key == GLFW_KEY_D) {
                cmd::DeselectAll{}.emit();
                return;
            }
            // Ctrl+1..5: selection sub-mode shortcuts
            if (gui_manager_ && key >= GLFW_KEY_1 && key <= GLFW_KEY_5) {
                using Mode = gui::panels::SelectionSubMode;
                static constexpr Mode SELECTION_MODES[] = {
                    Mode::Centers, Mode::Rectangle, Mode::Polygon, Mode::Lasso, Mode::Rings
                };
                gui_manager_->setSelectionSubMode(SELECTION_MODES[key - GLFW_KEY_1]);
                return;
            }
        }

        // Enter - apply crop box
        if (key == GLFW_KEY_ENTER && action == GLFW_PRESS) {
            LOG_INFO("Enter pressed, ImGui::IsAnyItemActive()={}", ImGui::IsAnyItemActive());
            if (!ImGui::IsAnyItemActive()) {
                cmd::ApplyCropBox{}.emit();
            }
            return;
        }

        // WASD only works when viewport has focus and gizmo isn't active
        if (!shouldCameraHandleInput() || drag_mode_ == DragMode::Gizmo || drag_mode_ == DragMode::Splitter)
            return;

        bool pressed = (action != GLFW_RELEASE);
        bool changed = false;

        switch (key) {
        case GLFW_KEY_W:
            keys_wasd_[0] = pressed;
            changed = true;
            break;
        case GLFW_KEY_A:
            keys_wasd_[1] = pressed;
            changed = true;
            break;
        case GLFW_KEY_S:
            keys_wasd_[2] = pressed;
            changed = true;
            break;
        case GLFW_KEY_D:
            keys_wasd_[3] = pressed;
            changed = true;
            break;
        case GLFW_KEY_Q:
            keys_wasd_[4] = pressed;
            changed = true;
            break;
        case GLFW_KEY_E:
            keys_wasd_[5] = pressed;
            changed = true;
            break;
        }

        if (changed) {
            LOG_TRACE("WASD state changed - W:{} A:{} S:{} D:{} Q:{} E:{}",
                      keys_wasd_[0], keys_wasd_[1], keys_wasd_[2], keys_wasd_[3], keys_wasd_[4], keys_wasd_[5]);
        }
    }

    void InputController::update(float delta_time) {
        // This catches cases where mouse release events are missed (e.g., outside window)
        if (drag_mode_ == DragMode::Orbit &&
            glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) != GLFW_PRESS) {
            viewport_.camera.endRotateAroundCenter();
            drag_mode_ = DragMode::None;
            LOG_TRACE("Orbit stopped - button released outside window");
        }

        if (drag_mode_ == DragMode::Rotate &&
            glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS) {
            drag_mode_ = DragMode::None;
            LOG_TRACE("Rotate stopped - button released outside window");
        }

        if (drag_mode_ == DragMode::Pan &&
            glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
            drag_mode_ = DragMode::None;
            LOG_TRACE("Pan stopped - button released outside window");
        }

        if (drag_mode_ == DragMode::Splitter &&
            glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
            drag_mode_ = DragMode::None;
            glfwSetCursor(window_, nullptr); // Reset cursor
            LOG_TRACE("Splitter drag stopped - button released outside window");
        }

        if (drag_mode_ == DragMode::Brush &&
            glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
            drag_mode_ = DragMode::None;
        }

        // Prevent stuck keys by syncing with actual keyboard state
        if (keys_wasd_[0] && glfwGetKey(window_, GLFW_KEY_W) != GLFW_PRESS) {
            keys_wasd_[0] = false;
            LOG_TRACE("W key unstuck");
        }
        if (keys_wasd_[1] && glfwGetKey(window_, GLFW_KEY_A) != GLFW_PRESS) {
            keys_wasd_[1] = false;
            LOG_TRACE("A key unstuck");
        }
        if (keys_wasd_[2] && glfwGetKey(window_, GLFW_KEY_S) != GLFW_PRESS) {
            keys_wasd_[2] = false;
            LOG_TRACE("S key unstuck");
        }
        if (keys_wasd_[3] && glfwGetKey(window_, GLFW_KEY_D) != GLFW_PRESS) {
            keys_wasd_[3] = false;
            LOG_TRACE("D key unstuck");
        }
        if (keys_wasd_[4] && glfwGetKey(window_, GLFW_KEY_Q) != GLFW_PRESS) {
            keys_wasd_[4] = false;
            LOG_TRACE("Q key unstuck");
        }
        if (keys_wasd_[5] && glfwGetKey(window_, GLFW_KEY_E) != GLFW_PRESS) {
            keys_wasd_[5] = false;
            LOG_TRACE("E key unstuck");
        }

        // Handle continuous WASD movement
        if (shouldCameraHandleInput() && drag_mode_ != DragMode::Gizmo && drag_mode_ != DragMode::Splitter) {
            if (keys_wasd_[0]) {
                viewport_.camera.advance_forward(delta_time);
            }
            if (keys_wasd_[1]) {
                viewport_.camera.advance_left(delta_time);
            }
            if (keys_wasd_[2]) {
                viewport_.camera.advance_backward(delta_time);
            }
            if (keys_wasd_[3]) {
                viewport_.camera.advance_right(delta_time);
            }
            if (keys_wasd_[4]) {
                viewport_.camera.advance_up(delta_time);
            }
            if (keys_wasd_[5]) {
                viewport_.camera.advance_down(delta_time);
            }
        }

        // Publish if moving (removed inertia check)
        bool moving = keys_wasd_[0] || keys_wasd_[1] || keys_wasd_[2] || keys_wasd_[3] || keys_wasd_[4] || keys_wasd_[5];
        if (moving) {
            onCameraMovementStart();
            publishCameraMove();
        }

        // Check if camera movement has timed out and should resume training
        checkCameraMovementTimeout();
    }

    void InputController::handleFileDrop(const std::vector<std::string>& paths) {
        LOG_DEBUG("Handling file drop with {} files", paths.size());

        std::vector<std::filesystem::path> splat_files;
        std::optional<std::filesystem::path> dataset_path;

        for (const auto& path_str : paths) {
            std::filesystem::path filepath(path_str);
            LOG_TRACE("Processing dropped file: {}", filepath.string());

            auto ext = filepath.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".ply" || ext == ".sog") {
                splat_files.push_back(filepath);
            } else if (!dataset_path && std::filesystem::is_directory(filepath)) {
                // Check for dataset markers
                LOG_TRACE("Checking directory for dataset markers: {}", filepath.string());
                if (lfs::loader::Loader::isDatasetPath(filepath)) {
                    dataset_path = filepath;
                    LOG_DEBUG("Dataset detected in dropped directory");
                }
            }
        }

        // Load splat files (PLY or SOG)
        for (const auto& splat : splat_files) {
            cmd::LoadFile{.path = splat, .is_dataset = false}.emit();
            LOG_INFO("Loading {} via drag-and-drop: {}",
                     splat.extension().string(), splat.filename().string());
        }

        // Load dataset if found
        if (dataset_path) {
            cmd::LoadFile{.path = *dataset_path, .is_dataset = true}.emit();
            LOG_INFO("Loading dataset via drag-and-drop: {}", dataset_path->filename().string());
        }

        if (paths.size() == 1) {
            auto project_path = std::filesystem::path(paths[0]);
            if (project_path.extension() == lfs::project::Project::EXTENSION) {
                cmd::LoadProject{.path = project_path}.emit();
                LOG_INFO("Loading LS Project via drag-and-drop: {}", project_path.filename().string());
            }
        }
    }

    void InputController::handleGoToCamView(const lfs::core::events::cmd::GoToCamView& event) {
        LOG_TIMER_TRACE("HandleGoToCamView");

        if (!training_manager_) {
            LOG_ERROR("GoToCamView: trainer_manager_ not initialized");
            return;
        }

        auto cam_data = training_manager_->getCamById(event.cam_id);
        if (!cam_data) {
            LOG_ERROR("Camera ID {} not found", event.cam_id);
            return;
        }

        LOG_DEBUG("Moving camera to view ID: {} ({})", event.cam_id, cam_data->image_name());

        // Get rotation and translation tensors and ensure they're on CPU
        auto R_tensor = cam_data->R().cpu();
        auto T_tensor = cam_data->T().cpu();

        // Get raw CPU pointers - safer and more efficient
        const float* R_data = R_tensor.ptr<float>();
        const float* T_data = T_tensor.ptr<float>();

        if (!R_data || !T_data) {
            LOG_ERROR("Failed to get camera R/T data pointers");
            return;
        }

        // R_data is world_to_cam rotation stored row-major
        // We need cam_to_world for the viewport
        glm::mat3 world_to_cam_R;

        // Load the matrix properly: R_data is row-major, GLM is column-major
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                // R_data[row * 3 + col] is element at [row][col] in row-major
                // GLM[col][row] is element at [row][col] when thinking row-major
                world_to_cam_R[col][row] = R_data[row * 3 + col];
            }
        }

        glm::vec3 world_to_cam_T(T_data[0], T_data[1], T_data[2]);

        // Convert to camera-to-world transform
        glm::mat3 cam_to_world_R = glm::transpose(world_to_cam_R);
        glm::vec3 cam_to_world_T = -cam_to_world_R * world_to_cam_T;

        viewport_.camera.R = cam_to_world_R;
        viewport_.camera.t = cam_to_world_T;

        // Update pivot point to be in front of camera
        viewport_.camera.updatePivotFromCamera();

        // Save as home position if this is the first camera view
        if (!viewport_.camera.home_saved) {
            viewport_.camera.saveHomePosition();
            LOG_DEBUG("Saved home position");
        }

        // Get camera intrinsics using the proper method
        auto [focal_x, focal_y, center_x, center_y] = cam_data->get_intrinsics();
        const float width = static_cast<float>(cam_data->image_width());
        const float height = static_cast<float>(cam_data->image_height());

        // Calculate vertical FOV using the actual focal length
        const float fov_y_rad = 2.0f * std::atan(height / (2.0f * focal_y));
        const float fov_y_deg = glm::degrees(fov_y_rad);

        LOG_DEBUG("Camera params - focal: ({:.1f}, {:.1f}), center: ({:.1f}, {:.1f}), image: {}x{}, FOV: {:.2f}°",
                  focal_x, focal_y, center_x, center_y, width, height, fov_y_deg);

        // Check for principal point offset (should be near center)
        const float cx_expected = width / 2.0f;
        const float cy_expected = height / 2.0f;

        if (std::abs(center_x - cx_expected) > 1.0f || std::abs(center_y - cy_expected) > 1.0f) {
            LOG_WARN("Camera has non-centered principal point: ({:.1f}, {:.1f}) vs expected ({:.1f}, {:.1f})",
                     center_x, center_y, cx_expected, cy_expected);
        }

        // Set the FOV
        ui::RenderSettingsChanged{
            .sh_degree = std::nullopt,
            .fov = fov_y_deg,
            .scaling_modifier = std::nullopt,
            .antialiasing = std::nullopt,
            .background_color = std::nullopt,
            .equirectangular = std::nullopt}
            .emit();

        // Force immediate camera update
        ui::CameraMove{
            .rotation = viewport_.getRotationMatrix(),
            .translation = viewport_.getTranslation()}
            .emit();

        // Set this as the current camera for GT comparison
        if (rendering_manager_) {
            rendering_manager_->setCurrentCameraId(event.cam_id);
        }

        last_camview = event.cam_id;
    }

    // Helpers
    bool InputController::isInViewport(double x, double y) const {
        return x >= viewport_bounds_.x &&
               x < viewport_bounds_.x + viewport_bounds_.width &&
               y >= viewport_bounds_.y &&
               y < viewport_bounds_.y + viewport_bounds_.height;
    }

    bool InputController::shouldCameraHandleInput() const {
        // Don't handle if gizmo or splitter is active
        if (drag_mode_ == DragMode::Gizmo || drag_mode_ == DragMode::Splitter) {
            return false;
        }

        // Only block when actively using a GUI widget
        return !ImGui::IsAnyItemActive();
    }

    void InputController::updateCameraSpeed(const bool increase) {
        increase ? viewport_.camera.increaseWasdSpeed() : viewport_.camera.decreaseWasdSpeed();
        ui::SpeedChanged{
            .current_speed = viewport_.camera.getWasdSpeed(),
            .max_speed = viewport_.camera.getMaxWasdSpeed()
        }.emit();
    }

    void InputController::updateZoomSpeed(const bool increase) {
        increase ? viewport_.camera.increaseZoomSpeed() : viewport_.camera.decreaseZoomSpeed();
        ui::ZoomSpeedChanged{
            .zoom_speed = viewport_.camera.getZoomSpeed(),
            .max_zoom_speed = viewport_.camera.getMaxZoomSpeed()
        }.emit();
    }

    void InputController::publishCameraMove() {
        auto now = std::chrono::steady_clock::now();
        if (now - last_camera_publish_ >= camera_publish_interval_) {
            ui::CameraMove{
                .rotation = viewport_.getRotationMatrix(),
                .translation = viewport_.getTranslation()}
                .emit();
            last_camera_publish_ = now;
        }
    }

    void InputController::onCameraMovementStart() {
        if (!camera_is_moving_) {
            camera_is_moving_ = true;
            last_camera_movement_time_ = std::chrono::steady_clock::now();

            // Pause training f it's running
            if (training_manager_ && training_manager_->isRunning()) {
                training_manager_->pauseTrainingTemporary();
                training_was_paused_by_camera_ = true;
                LOG_INFO("Camera movement detected - pausing training temporarily");
            }
        } else {
            // Update movement time
            last_camera_movement_time_ = std::chrono::steady_clock::now();
        }
    }

    void InputController::onCameraMovementEnd() {
        // Don't immediately resume - let the timeout handle it
        last_camera_movement_time_ = std::chrono::steady_clock::now();
    }

    void InputController::checkCameraMovementTimeout() {
        if (!camera_is_moving_) {
            return;
        }

        auto now = std::chrono::steady_clock::now();
        if (now - last_camera_movement_time_ >= camera_movement_timeout_) {
            camera_is_moving_ = false;

            // Resume training if we paused it
            if (training_was_paused_by_camera_ && training_manager_ && training_manager_->isRunning()) {
                training_manager_->resumeTrainingTemporary();
                training_was_paused_by_camera_ = false;
                LOG_INFO("Camera movement stopped - resuming training temporarily");
            }
        }
    }

    int InputController::getModifierKeys() const {
        int mods = 0;
        if (glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
            glfwGetKey(window_, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS) {
            mods |= GLFW_MOD_CONTROL;
        }
        if (glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
            mods |= GLFW_MOD_SHIFT;
        }
        return mods;
    }

    glm::vec3 InputController::unprojectScreenPoint(double x, double y, float fallback_distance) const {
        if (!rendering_manager_) {
            glm::vec3 forward = viewport_.camera.R * glm::vec3(0, 0, 1);
            return viewport_.camera.t + forward * fallback_distance;
        }

        // Convert window coordinates to viewport-local coordinates
        const float local_x = static_cast<float>(x) - viewport_bounds_.x;
        const float local_y = static_cast<float>(y) - viewport_bounds_.y;

        // Try to get depth from rendering manager using viewport-local coordinates
        const float depth = rendering_manager_->getDepthAtPixel(
            static_cast<int>(local_x), static_cast<int>(local_y));

        // If no valid depth, use fallback distance along view direction
        if (depth < 0.0f) {
            glm::vec3 forward = viewport_.camera.R * glm::vec3(0, 0, 1);
            return viewport_.camera.t + forward * fallback_distance;
        }

        // Use viewport dimensions for unprojection
        const float width = viewport_bounds_.width;
        const float height = viewport_bounds_.height;

        // Pinhole camera unprojection matching the rasterizer
        const float fov_y = glm::radians(rendering_manager_->getFovDegrees());
        const float aspect = width / height;
        const float fov_x = 2.0f * std::atan(std::tan(fov_y / 2.0f) * aspect);

        const float fx = width / (2.0f * std::tan(fov_x / 2.0f));
        const float fy = height / (2.0f * std::tan(fov_y / 2.0f));
        const float cx = width / 2.0f;
        const float cy = height / 2.0f;

        // Point in camera space (using viewport-local coordinates)
        const glm::vec4 view_pos(
            (local_x - cx) * depth / fx,
            (local_y - cy) * depth / fy,
            depth,
            1.0f);

        // Build world-to-camera matrix matching rasterizer: w2c = [R^T | -R^T*t]
        const glm::mat3 R = viewport_.getRotationMatrix();
        const glm::vec3 t = viewport_.getTranslation();
        const glm::mat3 R_inv = glm::transpose(R);
        const glm::vec3 t_inv = -R_inv * t;

        glm::mat4 w2c(1.0f);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                w2c[i][j] = R_inv[i][j];
        w2c[3][0] = t_inv.x;
        w2c[3][1] = t_inv.y;
        w2c[3][2] = t_inv.z;

        // Transform from camera space to world space
        return glm::vec3(glm::inverse(w2c) * view_pos);
    }

} // namespace lfs::vis