/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/tools_panel.hpp"
#include "gui/panels/crop_box_panel.hpp"
#include "gui/panels/transform_panel.hpp"
#include "gui/gui_manager.hpp"
#include "visualizer_impl.hpp"
#include <imgui.h>

namespace lfs::vis::gui::panels {

    void DrawToolsPanel(const UIContext& ctx) {
        auto* const gui_manager = ctx.viewer->getGuiManager();
        if (!gui_manager) return;

        const ToolMode current_tool = gui_manager->getCurrentToolMode();

        // Draw transform controls for translate/rotate/scale tools
        DrawTransformControls(ctx, current_tool);

        // Draw crop box controls only when crop box tool is active
        if (current_tool == ToolMode::CropBox) {
            DrawCropBoxControls(ctx);
        }
    }

} // namespace lfs::vis::gui::panels
