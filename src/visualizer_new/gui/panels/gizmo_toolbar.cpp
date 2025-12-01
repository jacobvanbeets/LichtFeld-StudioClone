/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <glad/glad.h>

#include "gui/panels/gizmo_toolbar.hpp"
#include "core_new/image_io.hpp"
#include "core_new/logger.hpp"
#include "internal/resource_paths.hpp"
#include <imgui.h>

namespace lfs::vis::gui::panels {

    // Toolbar layout constants
    constexpr float BUTTON_SIZE = 24.0f;
    constexpr float PADDING = 6.0f;
    constexpr float ITEM_SPACING = 4.0f;
    constexpr float WINDOW_ROUNDING = 6.0f;
    constexpr float SUBTOOLBAR_OFFSET_Y = 8.0f;

    // Button colors
    const ImVec4 BTN_SELECTED{0.3f, 0.5f, 0.8f, 1.0f};
    const ImVec4 BTN_NORMAL{0.2f, 0.2f, 0.2f, 1.0f};
    const ImVec4 BTN_SELECTED_HOVER{0.4f, 0.6f, 0.9f, 1.0f};
    const ImVec4 BTN_NORMAL_HOVER{0.3f, 0.3f, 0.3f, 1.0f};

    // Computes toolbar dimensions for N buttons
    static ImVec2 ComputeToolbarSize(const int num_buttons) {
        const float width = num_buttons * BUTTON_SIZE +
                            (num_buttons - 1) * ITEM_SPACING +
                            2.0f * PADDING;
        const float height = BUTTON_SIZE + 2.0f * PADDING;
        return ImVec2(width, height);
    }

    // RAII helper for toolbar style setup
    struct ToolbarStyle {
        ToolbarStyle() {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, WINDOW_ROUNDING);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(PADDING, PADDING));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ITEM_SPACING, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.15f, 0.9f));
        }
        ~ToolbarStyle() {
            ImGui::PopStyleColor();
            ImGui::PopStyleVar(4);
        }
    };

    // Secondary toolbar style (slightly darker)
    struct SubToolbarStyle {
        SubToolbarStyle() {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, WINDOW_ROUNDING);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(PADDING, PADDING));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ITEM_SPACING, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.12f, 0.95f));
        }
        ~SubToolbarStyle() {
            ImGui::PopStyleColor();
            ImGui::PopStyleVar(4);
        }
    };

    static unsigned int LoadIconTexture(const std::string& icon_name) {
        try {
            const auto path = lfs::vis::getAssetPath("icon/" + icon_name);
            const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(path);

            unsigned int texture_id;
            glGenTextures(1, &texture_id);
            glBindTexture(GL_TEXTURE_2D, texture_id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            const GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

            lfs::core::free_image(data);
            glBindTexture(GL_TEXTURE_2D, 0);
            return texture_id;
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load toolbar icon {}: {}", icon_name, e.what());
            return 0;
        }
    }

    void InitGizmoToolbar(GizmoToolbarState& state) {
        if (state.initialized) return;

        state.selection_texture = LoadIconTexture("selection.png");
        state.rectangle_texture = LoadIconTexture("rectangle.png");
        state.polygon_texture = LoadIconTexture("polygon.png");
        state.lasso_texture = LoadIconTexture("lasso.png");
        state.ring_texture = LoadIconTexture("ring.png");
        state.translation_texture = LoadIconTexture("translation.png");
        state.rotation_texture = LoadIconTexture("rotation.png");
        state.scaling_texture = LoadIconTexture("scaling.png");
        state.brush_texture = LoadIconTexture("brush.png");
        state.align_texture = LoadIconTexture("align.png");
        state.cropbox_texture = LoadIconTexture("cropbox.png");
        state.bounds_texture = LoadIconTexture("bounds.png");
        state.reset_texture = LoadIconTexture("reset.png");
        state.local_texture = LoadIconTexture("local.png");
        state.world_texture = LoadIconTexture("world.png");
        state.initialized = true;
    }

    void ShutdownGizmoToolbar(GizmoToolbarState& state) {
        if (!state.initialized) return;

        if (state.selection_texture) glDeleteTextures(1, &state.selection_texture);
        if (state.rectangle_texture) glDeleteTextures(1, &state.rectangle_texture);
        if (state.polygon_texture) glDeleteTextures(1, &state.polygon_texture);
        if (state.lasso_texture) glDeleteTextures(1, &state.lasso_texture);
        if (state.ring_texture) glDeleteTextures(1, &state.ring_texture);
        if (state.translation_texture) glDeleteTextures(1, &state.translation_texture);
        if (state.rotation_texture) glDeleteTextures(1, &state.rotation_texture);
        if (state.scaling_texture) glDeleteTextures(1, &state.scaling_texture);
        if (state.brush_texture) glDeleteTextures(1, &state.brush_texture);
        if (state.align_texture) glDeleteTextures(1, &state.align_texture);
        if (state.cropbox_texture) glDeleteTextures(1, &state.cropbox_texture);
        if (state.bounds_texture) glDeleteTextures(1, &state.bounds_texture);
        if (state.reset_texture) glDeleteTextures(1, &state.reset_texture);
        if (state.local_texture) glDeleteTextures(1, &state.local_texture);
        if (state.world_texture) glDeleteTextures(1, &state.world_texture);

        state.selection_texture = 0;
        state.rectangle_texture = 0;
        state.polygon_texture = 0;
        state.lasso_texture = 0;
        state.ring_texture = 0;
        state.translation_texture = 0;
        state.rotation_texture = 0;
        state.scaling_texture = 0;
        state.brush_texture = 0;
        state.align_texture = 0;
        state.cropbox_texture = 0;
        state.bounds_texture = 0;
        state.reset_texture = 0;
        state.local_texture = 0;
        state.world_texture = 0;
        state.initialized = false;
    }

    void DrawGizmoToolbar([[maybe_unused]] const UIContext& ctx, GizmoToolbarState& state,
                          const ImVec2& viewport_pos, const ImVec2& viewport_size) {
        if (!state.initialized) {
            InitGizmoToolbar(state);
        }

        const ImGuiViewport* viewport = ImGui::GetMainViewport();

        constexpr ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoSavedSettings;

        // Main toolbar: 7 buttons
        constexpr int NUM_MAIN_BUTTONS = 7;
        const ImVec2 toolbar_size = ComputeToolbarSize(NUM_MAIN_BUTTONS);

        const float pos_x = viewport->WorkPos.x + viewport_pos.x + (viewport_size.x - toolbar_size.x) * 0.5f;
        const float pos_y = viewport->WorkPos.y + viewport_pos.y + 5.0f;

        ImGui::SetNextWindowPos(ImVec2(pos_x, pos_y), ImGuiCond_Always);
        ImGui::SetNextWindowSize(toolbar_size, ImGuiCond_Always);

        {
            const ToolbarStyle style;
            if (ImGui::Begin("##GizmoToolbar", nullptr, flags)) {
                const ImVec2 btn_size(BUTTON_SIZE, BUTTON_SIZE);

                const auto IconButton = [&](const char* id, const unsigned int texture,
                                            const ToolMode tool, const ImGuizmo::OPERATION op,
                                            const char* fallback, const char* tooltip) {
                    const bool is_selected = (state.current_tool == tool);
                    ImGui::PushStyleColor(ImGuiCol_Button, is_selected ?
                        ImVec4(0.3f, 0.5f, 0.8f, 1.0f) : ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, is_selected ?
                        ImVec4(0.4f, 0.6f, 0.9f, 1.0f) : ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

                    const bool clicked = texture ?
                        ImGui::ImageButton(id, (ImTextureID)(intptr_t)texture, btn_size,
                                           ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0)) :
                        ImGui::Button(fallback, btn_size);

                    ImGui::PopStyleColor(2);
                    if (clicked) {
                        state.current_tool = is_selected ? ToolMode::None : tool;
                        if (!is_selected) state.current_operation = op;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
                    return clicked;
                };

                IconButton("##selection", state.selection_texture, ToolMode::Selection,
                           ImGuizmo::TRANSLATE, "S", "Selection");
                ImGui::SameLine();
                IconButton("##translate", state.translation_texture, ToolMode::Translate,
                           ImGuizmo::TRANSLATE, "T", "Translate");
                ImGui::SameLine();
                IconButton("##rotate", state.rotation_texture, ToolMode::Rotate,
                           ImGuizmo::ROTATE, "R", "Rotate");
                ImGui::SameLine();
                IconButton("##scale", state.scaling_texture, ToolMode::Scale,
                           ImGuizmo::SCALE, "S", "Scale");
                ImGui::SameLine();
                IconButton("##brush", state.brush_texture, ToolMode::Brush,
                           ImGuizmo::TRANSLATE, "B", "Brush Selection");
                ImGui::SameLine();
                IconButton("##align", state.align_texture, ToolMode::Align,
                           ImGuizmo::TRANSLATE, "A", "3-Point Align");
                ImGui::SameLine();
                IconButton("##cropbox", state.cropbox_texture, ToolMode::CropBox,
                           ImGuizmo::BOUNDS, "C", "Crop Box");
            }
            ImGui::End();
        }

        // Secondary toolbar for selection mode
        if (state.current_tool == ToolMode::Selection) {
            constexpr int NUM_SEL_BUTTONS = 5;
            const ImVec2 sub_size = ComputeToolbarSize(NUM_SEL_BUTTONS);

            const float sub_pos_x = viewport->WorkPos.x + viewport_pos.x + (viewport_size.x - sub_size.x) * 0.5f;
            const float sub_pos_y = viewport->WorkPos.y + viewport_pos.y + toolbar_size.y + 8.0f;

            ImGui::SetNextWindowPos(ImVec2(sub_pos_x, sub_pos_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(sub_size, ImGuiCond_Always);

            {
                const SubToolbarStyle style;
                if (ImGui::Begin("##SelectionModeToolbar", nullptr, flags)) {
                    const ImVec2 btn_size(BUTTON_SIZE, BUTTON_SIZE);

                    const auto SelectionModeButton = [&](const char* id, const unsigned int texture,
                                                         const SelectionSubMode mode, const char* fallback,
                                                         const char* tooltip) {
                        const bool is_selected = (state.selection_mode == mode);
                        ImGui::PushStyleColor(ImGuiCol_Button, is_selected ?
                            ImVec4(0.3f, 0.5f, 0.8f, 1.0f) : ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
                        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, is_selected ?
                            ImVec4(0.4f, 0.6f, 0.9f, 1.0f) : ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

                        const bool clicked = texture ?
                            ImGui::ImageButton(id, (ImTextureID)(intptr_t)texture, btn_size,
                                               ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0)) :
                            ImGui::Button(fallback, btn_size);

                        ImGui::PopStyleColor(2);
                        if (clicked) state.selection_mode = mode;
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
                    };

                    SelectionModeButton("##centers", state.selection_texture, SelectionSubMode::Centers,
                                        "C", "Brush selection (Ctrl+1)");
                    ImGui::SameLine();
                    SelectionModeButton("##rect", state.rectangle_texture, SelectionSubMode::Rectangle,
                                        "R", "Rectangle selection (Ctrl+2)");
                    ImGui::SameLine();
                    SelectionModeButton("##polygon", state.polygon_texture, SelectionSubMode::Polygon,
                                        "P", "Polygon selection (Ctrl+3)");
                    ImGui::SameLine();
                    SelectionModeButton("##lasso", state.lasso_texture, SelectionSubMode::Lasso,
                                        "L", "Lasso selection (Ctrl+4)");
                    ImGui::SameLine();
                    SelectionModeButton("##rings", state.ring_texture, SelectionSubMode::Rings,
                                        "O", "Ring selection (Ctrl+5)");
                }
                ImGui::End();
            }
        }

        // Transform space toolbar (Local/World toggle)
        const bool is_transform_tool = (state.current_tool == ToolMode::Translate ||
                                        state.current_tool == ToolMode::Rotate ||
                                        state.current_tool == ToolMode::Scale);
        if (is_transform_tool) {
            constexpr int NUM_BUTTONS = 2;
            const ImVec2 sub_size = ComputeToolbarSize(NUM_BUTTONS);
            const float sub_x = viewport->WorkPos.x + viewport_pos.x + (viewport_size.x - sub_size.x) * 0.5f;
            const float sub_y = viewport->WorkPos.y + viewport_pos.y + toolbar_size.y + SUBTOOLBAR_OFFSET_Y;

            ImGui::SetNextWindowPos(ImVec2(sub_x, sub_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(sub_size, ImGuiCond_Always);

            const SubToolbarStyle style;
            if (ImGui::Begin("##TransformSpaceToolbar", nullptr, flags)) {
                const ImVec2 btn_size(BUTTON_SIZE, BUTTON_SIZE);

                const auto SpaceButton = [&](const char* id, const unsigned int tex,
                                             const TransformSpace space, const char* fallback,
                                             const char* tooltip) {
                    const bool selected = (state.transform_space == space);
                    ImGui::PushStyleColor(ImGuiCol_Button, selected ? BTN_SELECTED : BTN_NORMAL);
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, selected ? BTN_SELECTED_HOVER : BTN_NORMAL_HOVER);

                    const bool clicked = tex
                        ? ImGui::ImageButton(id, static_cast<ImTextureID>(tex),
                                             btn_size, ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0))
                        : ImGui::Button(fallback, btn_size);

                    ImGui::PopStyleColor(2);
                    if (clicked) { state.transform_space = space; }
                    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("%s", tooltip); }
                };

                SpaceButton("##local", state.local_texture, TransformSpace::Local, "L", "Local Space");
                ImGui::SameLine();
                SpaceButton("##world", state.world_texture, TransformSpace::World, "W", "World Space");
            }
            ImGui::End();
        }

        // Cropbox operations toolbar
        if (state.current_tool == ToolMode::CropBox) {
            constexpr int NUM_CROP_BUTTONS = 5;
            const ImVec2 sub_size = ComputeToolbarSize(NUM_CROP_BUTTONS);
            const float sub_x = viewport->WorkPos.x + viewport_pos.x + (viewport_size.x - sub_size.x) * 0.5f;
            const float sub_y = viewport->WorkPos.y + viewport_pos.y + toolbar_size.y + SUBTOOLBAR_OFFSET_Y;

            ImGui::SetNextWindowPos(ImVec2(sub_x, sub_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(sub_size, ImGuiCond_Always);

            const SubToolbarStyle style;
            if (ImGui::Begin("##CropBoxToolbar", nullptr, flags)) {
                const ImVec2 btn_size(BUTTON_SIZE, BUTTON_SIZE);

                const auto CropOpButton = [&](const char* id, const unsigned int tex,
                                              const CropBoxOperation op, const char* fallback,
                                              const char* tooltip) {
                    const bool selected = (state.cropbox_operation == op);
                    ImGui::PushStyleColor(ImGuiCol_Button, selected ? BTN_SELECTED : BTN_NORMAL);
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, selected ? BTN_SELECTED_HOVER : BTN_NORMAL_HOVER);

                    const bool clicked = tex
                        ? ImGui::ImageButton(id, static_cast<ImTextureID>(tex), btn_size,
                                             ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0))
                        : ImGui::Button(fallback, btn_size);

                    ImGui::PopStyleColor(2);
                    if (clicked) { state.cropbox_operation = op; }
                    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("%s", tooltip); }
                };

                CropOpButton("##crop_bounds", state.bounds_texture, CropBoxOperation::Bounds, "B", "Resize Bounds");
                ImGui::SameLine();
                CropOpButton("##crop_translate", state.translation_texture, CropBoxOperation::Translate, "T", "Translate");
                ImGui::SameLine();
                CropOpButton("##crop_rotate", state.rotation_texture, CropBoxOperation::Rotate, "R", "Rotate");
                ImGui::SameLine();
                CropOpButton("##crop_scale", state.scaling_texture, CropBoxOperation::Scale, "S", "Scale");
                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Button, BTN_NORMAL);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, BTN_NORMAL_HOVER);
                const bool reset_clicked = state.reset_texture
                    ? ImGui::ImageButton("##crop_reset", static_cast<ImTextureID>(state.reset_texture),
                                         btn_size, ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0))
                    : ImGui::Button("X", btn_size);
                ImGui::PopStyleColor(2);
                if (reset_clicked) { state.reset_cropbox_requested = true; }
                if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Reset to Default"); }
            }
            ImGui::End();
        }
    }

} // namespace lfs::vis::gui::panels
