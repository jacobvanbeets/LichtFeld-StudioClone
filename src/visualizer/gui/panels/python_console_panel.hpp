/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace lfs::vis::editor {
    class PythonEditor;
    class ConsoleOutput;
} // namespace lfs::vis::editor

namespace lfs::vis::gui::panels {

    class PythonConsoleState {
    public:
        static PythonConsoleState& getInstance();

        void addOutput(const std::string& text, uint32_t color = 0xFFFFFFFF);
        void addError(const std::string& text);
        void addInput(const std::string& text);
        void clear();

        void addToHistory(const std::string& cmd);
        void resetHistoryIndex() { history_index_ = -1; }
        void historyUp();
        void historyDown();
        int historyIndex() const { return history_index_; }

        editor::ConsoleOutput* getConsoleOutput();
        editor::PythonEditor* getEditor();

    private:
        PythonConsoleState();
        ~PythonConsoleState();

        std::vector<std::string> command_history_;
        int history_index_ = -1;
        mutable std::mutex mutex_;
        static constexpr size_t MAX_MESSAGES = 1000;

        std::unique_ptr<editor::ConsoleOutput> console_output_;
        std::unique_ptr<editor::PythonEditor> editor_;
    };

    // Draw the Python console window
    void DrawPythonConsole(const UIContext& ctx, bool* open);

} // namespace lfs::vis::gui::panels
