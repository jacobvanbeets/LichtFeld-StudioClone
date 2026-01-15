/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "view_context.hpp"

namespace lfs::vis {

    static GetViewCallback g_view_callback = nullptr;
    static GetViewportRenderCallback g_viewport_render_callback = nullptr;

    void set_view_callback(GetViewCallback callback) {
        g_view_callback = std::move(callback);
    }

    void set_viewport_render_callback(GetViewportRenderCallback callback) {
        g_viewport_render_callback = std::move(callback);
    }

    std::optional<ViewInfo> get_current_view_info() {
        if (!g_view_callback)
            return std::nullopt;
        return g_view_callback();
    }

    std::optional<ViewportRender> get_viewport_render() {
        if (!g_viewport_render_callback)
            return std::nullopt;
        return g_viewport_render_callback();
    }

} // namespace lfs::vis
