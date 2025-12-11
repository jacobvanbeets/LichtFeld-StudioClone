/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

struct ImFont;

namespace lfs::vis {
    // Forward declarations
    class VisualizerImpl;
    class EditorContext;

    namespace gui {
        class FileBrowser;

        // Font set for typography hierarchy
        struct FontSet {
            ImFont* regular = nullptr;
            ImFont* bold = nullptr;
            ImFont* heading = nullptr;
            ImFont* small = nullptr;
            ImFont* section = nullptr;
        };

        struct UIContext {
            VisualizerImpl* viewer = nullptr;
            FileBrowser* file_browser = nullptr;
            std::unordered_map<std::string, bool>* window_states = nullptr;
            EditorContext* editor = nullptr;
            FontSet fonts;
        };

    } // namespace gui
} // namespace lfs::vis
