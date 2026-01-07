/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <string>
#include <vector>
#include <imgui.h>

namespace lfs::vis::editor {

    enum class ConsoleMessageType {
        Input,  // User input (prefixed with >>>)
        Output, // Standard output
        Error,  // Error output
        Info    // System info messages
    };

    struct ConsoleMessage {
        std::string text;
        ConsoleMessageType type;
    };

    class ConsoleOutput {
    public:
        ConsoleOutput();
        ~ConsoleOutput();

        // Add messages
        void addInput(const std::string& text);
        void addOutput(const std::string& text);
        void addError(const std::string& text);
        void addInfo(const std::string& text);

        // Clear all messages
        void clear();

        // Render the output area
        void render(const ImVec2& size);

        // Copy functionality
        std::string getAllText() const;
        std::string getSelectedText() const;
        bool hasSelection() const { return selection_start_ != selection_end_; }

        // Scroll control
        void scrollToBottom() { scroll_to_bottom_ = true; }

    private:
        void renderContextMenu();
        void handleSelection();
        void copyToClipboard(const std::string& text);

        std::vector<ConsoleMessage> messages_;
        bool scroll_to_bottom_ = false;
        bool auto_scroll_ = true;

        // Selection state
        int selection_start_ = 0;
        int selection_end_ = 0;
        bool is_selecting_ = false;
        int hover_line_ = -1;
    };

} // namespace lfs::vis::editor
