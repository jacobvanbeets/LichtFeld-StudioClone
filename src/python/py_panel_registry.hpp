/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>
#include <string>
#include <vector>

namespace lfs::vis {
    class Scene;
}

namespace lfs::python {

    // Panel space types
    enum class PanelSpace {
        SidePanel,
        Floating,
        ViewportOverlay
    };

    // Callback types for the Python panel system
    using DrawPanelsCallback = std::function<void(PanelSpace)>;
    using DrawSinglePanelCallback = std::function<void(const std::string&)>;
    using HasPanelsCallback = std::function<bool(PanelSpace)>;
    using GetPanelNamesCallback = std::function<std::vector<std::string>(PanelSpace)>;
    using CleanupCallback = std::function<void()>;

    // Register callbacks from the Python module
    void set_panel_draw_callback(DrawPanelsCallback cb);
    void set_panel_draw_single_callback(DrawSinglePanelCallback cb);
    void set_panel_has_callback(HasPanelsCallback cb);
    void set_panel_names_callback(GetPanelNamesCallback cb);
    void set_python_cleanup_callback(CleanupCallback cb);
    void clear_panel_callbacks();

    // C++ interface for the visualizer
    void draw_python_panels(PanelSpace space, lfs::vis::Scene* scene = nullptr);
    void draw_python_panel(const std::string& name, lfs::vis::Scene* scene = nullptr);
    bool has_python_panels(PanelSpace space);
    std::vector<std::string> get_python_panel_names(PanelSpace space);
    void invoke_python_cleanup();

} // namespace lfs::python
