/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_panel_registry.hpp"
#include "core/logger.hpp"

namespace lfs::python {

    namespace {
        DrawPanelsCallback g_draw_callback;
        HasPanelsCallback g_has_callback;
        CleanupCallback g_cleanup_callback;
    } // namespace

    void set_panel_draw_callback(DrawPanelsCallback cb) {
        g_draw_callback = std::move(cb);
        LOG_INFO("Python panel draw callback registered");
    }

    void set_panel_has_callback(HasPanelsCallback cb) {
        g_has_callback = std::move(cb);
    }

    void set_python_cleanup_callback(CleanupCallback cb) {
        g_cleanup_callback = std::move(cb);
    }

    void clear_panel_callbacks() {
        g_draw_callback = nullptr;
        g_has_callback = nullptr;
        g_cleanup_callback = nullptr;
    }

    void invoke_python_cleanup() {
        if (g_cleanup_callback) {
            g_cleanup_callback();
        }
    }

    void draw_python_panels(PanelSpace space) {
        if (g_draw_callback) {
            g_draw_callback(space);
        } else {
            // Callback not set - Python module not loaded yet
            static bool warned = false;
            if (!warned) {
                LOG_DEBUG("Python panel draw callback not set (Python module not loaded?)");
                warned = true;
            }
        }
    }

    bool has_python_panels(PanelSpace space) {
        if (g_has_callback) {
            return g_has_callback(space);
        }
        return false;
    }

} // namespace lfs::python
