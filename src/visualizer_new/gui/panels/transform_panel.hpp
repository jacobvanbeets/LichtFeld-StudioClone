/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command/commands/cropbox_command.hpp"
#include "gui/ui_context.hpp"
#include "gui/panels/gizmo_toolbar.hpp"
#include <glm/glm.hpp>
#include <string>

namespace lfs::vis::gui::panels {

    struct TransformPanelState {
        bool editing_active = false;
        std::string editing_node_name;
        glm::mat4 transform_before_edit{1.0f};
        command::CropBoxState crop_before_edit{};

        // Cached decomposed values at edit start
        glm::vec3 initial_translation{0.0f};
        glm::vec3 initial_scale{1.0f};

        // For incremental world-space rotation
        glm::vec3 world_euler{0.0f};      // Cumulative world-space euler (displayed)
        glm::vec3 prev_world_euler{0.0f}; // Previous frame's value (for delta)
    };

    void DrawTransformControls(const UIContext& ctx, ToolMode current_tool,
                               TransformSpace transform_space, TransformPanelState& state);

} // namespace lfs::vis::gui::panels
