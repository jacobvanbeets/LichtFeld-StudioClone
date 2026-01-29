/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "view_context.hpp"

namespace lfs::vis {

    static GetViewCallback g_view_callback = nullptr;
    static GetViewportRenderCallback g_viewport_render_callback = nullptr;
    static GetRenderSettingsCallback g_get_render_settings_callback = nullptr;
    static SetRenderSettingsCallback g_set_render_settings_callback = nullptr;

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

    void set_render_settings_callbacks(GetRenderSettingsCallback get_cb, SetRenderSettingsCallback set_cb) {
        g_get_render_settings_callback = std::move(get_cb);
        g_set_render_settings_callback = std::move(set_cb);
    }

    std::optional<RenderSettingsProxy> get_render_settings() {
        if (!g_get_render_settings_callback)
            return std::nullopt;
        return g_get_render_settings_callback();
    }

    void update_render_settings(const RenderSettingsProxy& settings) {
        if (g_set_render_settings_callback) {
            g_set_render_settings_callback(settings);
        }
    }

} // namespace lfs::vis
