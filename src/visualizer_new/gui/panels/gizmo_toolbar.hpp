/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include <imgui.h>
#include <ImGuizmo.h>

namespace lfs::vis::gui::panels {

    enum class ToolMode {
        None,
        Selection,
        Translate,
        Rotate,
        Scale,
        Brush,
        Align,
        CropBox
    };

    enum class CropBoxOperation {
        Bounds,
        Translate,
        Rotate,
        Scale
    };

    enum class SelectionSubMode {
        Centers,
        Rectangle,
        Polygon,
        Lasso,
        Rings
    };

    enum class TransformSpace {
        Local,
        World
    };

    struct GizmoToolbarState {
        ImGuizmo::OPERATION current_operation = ImGuizmo::TRANSLATE;
        ToolMode current_tool = ToolMode::Translate;
        CropBoxOperation cropbox_operation = CropBoxOperation::Bounds;
        bool reset_cropbox_requested = false;
        SelectionSubMode selection_mode = SelectionSubMode::Centers;
        TransformSpace transform_space = TransformSpace::Local;
        bool initialized = false;
        unsigned int selection_texture = 0;
        unsigned int rectangle_texture = 0;
        unsigned int polygon_texture = 0;
        unsigned int lasso_texture = 0;
        unsigned int ring_texture = 0;
        unsigned int translation_texture = 0;
        unsigned int rotation_texture = 0;
        unsigned int scaling_texture = 0;
        unsigned int brush_texture = 0;
        unsigned int align_texture = 0;
        unsigned int cropbox_texture = 0;
        unsigned int bounds_texture = 0;
        unsigned int reset_texture = 0;
        unsigned int local_texture = 0;
        unsigned int world_texture = 0;
    };

    void InitGizmoToolbar(GizmoToolbarState& state);
    void ShutdownGizmoToolbar(GizmoToolbarState& state);
    void DrawGizmoToolbar(const UIContext& ctx, GizmoToolbarState& state,
                          const ImVec2& viewport_pos, const ImVec2& viewport_size);

} // namespace lfs::vis::gui::panels
