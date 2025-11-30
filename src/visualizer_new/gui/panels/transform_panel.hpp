/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include "gui/panels/gizmo_toolbar.hpp"

namespace lfs::vis::gui::panels {

    void DrawTransformControls(const UIContext& ctx, ToolMode current_tool);

} // namespace lfs::vis::gui::panels
