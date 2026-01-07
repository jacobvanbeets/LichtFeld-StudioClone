/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_editor.hpp"
#include "editor_theme.hpp"
#include "python_language_def.hpp"
#include "theme/theme.hpp"
#include <cctype>

namespace lfs::vis::editor {

    PythonEditor::PythonEditor() {
        editor_.SetLanguageDefinition(getPythonLanguageDef());
        editor_.SetShowWhitespaces(false);
        editor_.SetTabSize(4);
        updateTheme(theme());
    }

    PythonEditor::~PythonEditor() = default;

    bool PythonEditor::render(const ImVec2& size) {
        execute_requested_ = false;

        // Handle focus request
        if (request_focus_) {
            ImGui::SetKeyboardFocusHere();
            request_focus_ = false;
        }

        // Check for Ctrl+Enter BEFORE rendering (to prevent newline insertion)
        ImGuiIO& io = ImGui::GetIO();
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
            execute_requested_ = true;
            autocomplete_.hide();
        }

        // Check for autocomplete navigation BEFORE rendering
        if (autocomplete_.isVisible()) {
            if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                autocomplete_.hide();
            } else if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, false)) {
                autocomplete_.selectPrevious();
                editor_.SetHandleKeyboardInputs(false);
            } else if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, false)) {
                autocomplete_.selectNext();
                editor_.SetHandleKeyboardInputs(false);
            } else if (ImGui::IsKeyPressed(ImGuiKey_Tab, false) ||
                       (!io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Enter, false))) {
                std::string selected;
                if (autocomplete_.acceptSelected(selected)) {
                    insertCompletion(selected);
                }
                editor_.SetHandleKeyboardInputs(false);
            } else {
                editor_.SetHandleKeyboardInputs(true);
            }
        } else {
            editor_.SetHandleKeyboardInputs(true);
        }

        // Render editor
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 4));
        editor_.Render("##python_input", size, true);
        ImGui::PopStyleVar();

        is_focused_ = ImGui::IsItemFocused() || ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);

        // Handle post-render input (Ctrl+Space, history)
        if (is_focused_) {
            // Ctrl+Space to force autocomplete
            if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
                autocomplete_triggered_ = true;
                autocomplete_.show();
            }

            // History navigation (only when single line and autocomplete not visible)
            if (!autocomplete_.isVisible() && editor_.GetTotalLines() <= 1) {
                if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, false)) {
                    historyUp();
                } else if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, false)) {
                    historyDown();
                }
            }

            // Update autocomplete based on current text
            updateAutocomplete();

            // Render autocomplete popup
            if (autocomplete_.isVisible()) {
                auto cursor_pos = editor_.GetCursorPosition();
                ImVec2 popup_pos = ImGui::GetItemRectMin();
                popup_pos.y += (cursor_pos.mLine + 1) * ImGui::GetTextLineHeightWithSpacing();
                popup_pos.x += 50;

                std::string selected;
                if (autocomplete_.renderPopup(popup_pos, selected)) {
                    insertCompletion(selected);
                }
            }
        }

        return execute_requested_;
    }

    void PythonEditor::updateAutocomplete() {
        std::string word = getWordBeforeCursor();
        std::string context = getContextBeforeCursor();

        // Don't show autocomplete for empty input unless explicitly triggered
        if (word.empty() && !autocomplete_triggered_) {
            autocomplete_.hide();
            return;
        }

        // Update completions
        autocomplete_.updateCompletions(word, context);

        // Auto-hide if no completions
        if (!autocomplete_.hasCompletions()) {
            autocomplete_.hide();
        }

        autocomplete_triggered_ = false;
    }

    std::string PythonEditor::getWordBeforeCursor() const {
        auto pos = editor_.GetCursorPosition();
        auto text = editor_.GetCurrentLineText();

        if (text.empty() || pos.mColumn == 0) {
            return "";
        }

        int col = std::min(pos.mColumn, static_cast<int>(text.length()));
        int start = col;

        while (start > 0) {
            char c = text[start - 1];
            if (!std::isalnum(c) && c != '_') {
                break;
            }
            --start;
        }

        return text.substr(start, col - start);
    }

    std::string PythonEditor::getContextBeforeCursor() const {
        auto pos = editor_.GetCursorPosition();
        auto text = editor_.GetCurrentLineText();

        if (text.empty() || pos.mColumn == 0) {
            return "";
        }

        int col = std::min(pos.mColumn, static_cast<int>(text.length()));
        int start = std::max(0, col - 50);
        return text.substr(start, col - start);
    }

    void PythonEditor::insertCompletion(const std::string& text) {
        std::string word = getWordBeforeCursor();

        // Delete the partial word
        if (!word.empty()) {
            for (size_t i = 0; i < word.length(); ++i) {
                editor_.MoveLeft(1, true);
            }
            editor_.Delete();
        }

        // Insert completion
        editor_.InsertText(text);
    }

    std::string PythonEditor::getText() const {
        return editor_.GetText();
    }

    void PythonEditor::setText(const std::string& text) {
        editor_.SetText(text);
    }

    void PythonEditor::clear() {
        editor_.SetText("");
        history_index_ = -1;
    }

    void PythonEditor::updateTheme(const Theme& t) {
        applyThemeToEditor(editor_, t);
    }

    void PythonEditor::addToHistory(const std::string& cmd) {
        if (cmd.empty()) {
            return;
        }

        if (!history_.empty() && history_.back() == cmd) {
            return;
        }

        history_.push_back(cmd);

        const size_t max_history = 100;
        if (history_.size() > max_history) {
            history_.erase(history_.begin());
        }

        history_index_ = -1;
    }

    void PythonEditor::historyUp() {
        if (history_.empty()) {
            return;
        }

        if (history_index_ == -1) {
            current_input_ = getText();
            history_index_ = static_cast<int>(history_.size()) - 1;
        } else if (history_index_ > 0) {
            --history_index_;
        }

        setText(history_[history_index_]);
    }

    void PythonEditor::historyDown() {
        if (history_index_ == -1) {
            return;
        }

        if (history_index_ < static_cast<int>(history_.size()) - 1) {
            ++history_index_;
            setText(history_[history_index_]);
        } else {
            history_index_ = -1;
            setText(current_input_);
        }
    }

} // namespace lfs::vis::editor
