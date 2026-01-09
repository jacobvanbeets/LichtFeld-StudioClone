/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include <algorithm>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <imgui.h>

namespace lfs::vis::editor {
    class PythonEditor;
} // namespace lfs::vis::editor

namespace lfs::vis::terminal {
    class TerminalWidget;
} // namespace lfs::vis::terminal

namespace lfs::vis::gui::panels {

    class PythonConsoleState {
    public:
        static PythonConsoleState& getInstance();

        void addOutput(const std::string& text, uint32_t color = 0xFFFFFFFF);
        void addError(const std::string& text);
        void addInput(const std::string& text);
        void addInfo(const std::string& text);
        void clear();

        void addToHistory(const std::string& cmd);
        void resetHistoryIndex() { history_index_ = -1; }
        void historyUp();
        void historyDown();
        int historyIndex() const { return history_index_; }

        terminal::TerminalWidget* getTerminal();
        terminal::TerminalWidget* getOutputTerminal();
        editor::PythonEditor* getEditor();

        // Tab selection (0 = Output, 1 = Terminal)
        int getActiveTab() const { return active_tab_; }
        void setActiveTab(int tab) { active_tab_ = tab; }

        // Focus tracking for input routing
        bool isTerminalFocused() const { return terminal_focused_; }
        void setTerminalFocused(bool focused) { terminal_focused_ = focused; }

        // Script file management
        void setScriptPath(const std::filesystem::path& path) { script_path_ = path; }
        const std::filesystem::path& getScriptPath() const { return script_path_; }
        void setModified(bool modified) { is_modified_ = modified; }
        bool isModified() const { return is_modified_; }

        // Font scaling
        float getFontScale() const { return font_scale_; }
        void setFontScale(float scale) { font_scale_ = std::clamp(scale, 0.5f, 3.0f); }
        void increaseFontScale() { setFontScale(font_scale_ + 0.1f); }
        void decreaseFontScale() { setFontScale(font_scale_ - 0.1f); }
        void resetFontScale() { font_scale_ = 1.0f; }

    private:
        PythonConsoleState();
        ~PythonConsoleState();

        std::vector<std::string> command_history_;
        int history_index_ = -1;
        mutable std::mutex mutex_;
        static constexpr size_t MAX_MESSAGES = 1000;

        std::unique_ptr<terminal::TerminalWidget> terminal_;
        std::unique_ptr<terminal::TerminalWidget> output_terminal_;
        std::unique_ptr<editor::PythonEditor> editor_;
        int active_tab_ = 0;
        bool terminal_focused_ = false;

        // Script file tracking
        std::filesystem::path script_path_;
        bool is_modified_ = false;

        // Font scaling
        float font_scale_ = 1.0f;
    };

    // Draw the Python console window (floating)
    void DrawPythonConsole(const UIContext& ctx, bool* open);

    // Draw the Python console as a docked panel (fixed position/size)
    void DrawDockedPythonConsole(const UIContext& ctx, const ImVec2& pos, const ImVec2& size);

} // namespace lfs::vis::gui::panels
