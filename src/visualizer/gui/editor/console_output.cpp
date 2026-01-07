/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "console_output.hpp"
#include "theme/theme.hpp"
#include <algorithm>
#include <sstream>

namespace lfs::vis::editor {

    ConsoleOutput::ConsoleOutput() = default;
    ConsoleOutput::~ConsoleOutput() = default;

    void ConsoleOutput::addInput(const std::string& text) {
        messages_.push_back({text, ConsoleMessageType::Input});
        if (auto_scroll_) {
            scroll_to_bottom_ = true;
        }
    }

    void ConsoleOutput::addOutput(const std::string& text) {
        messages_.push_back({text, ConsoleMessageType::Output});
        if (auto_scroll_) {
            scroll_to_bottom_ = true;
        }
    }

    void ConsoleOutput::addError(const std::string& text) {
        messages_.push_back({text, ConsoleMessageType::Error});
        if (auto_scroll_) {
            scroll_to_bottom_ = true;
        }
    }

    void ConsoleOutput::addInfo(const std::string& text) {
        messages_.push_back({text, ConsoleMessageType::Info});
        if (auto_scroll_) {
            scroll_to_bottom_ = true;
        }
    }

    void ConsoleOutput::clear() {
        messages_.clear();
    }

    void ConsoleOutput::render(const ImVec2& size) {
        const auto& t = theme();

        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.palette.surface);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));

        if (ImGui::BeginChild("##console_output", size, true,
                              ImGuiWindowFlags_HorizontalScrollbar)) {

            for (const auto& msg : messages_) {
                ImVec4 color;
                const char* prefix = "";
                switch (msg.type) {
                case ConsoleMessageType::Input:
                    color = t.palette.success;
                    prefix = ">>> ";
                    break;
                case ConsoleMessageType::Output:
                    color = t.palette.text;
                    break;
                case ConsoleMessageType::Error:
                    color = t.palette.error;
                    break;
                case ConsoleMessageType::Info:
                    color = t.palette.info;
                    break;
                }

                ImGui::PushStyleColor(ImGuiCol_Text, color);

                // Handle multi-line messages
                std::istringstream stream(msg.text);
                std::string line;
                bool first_line = true;

                while (std::getline(stream, line)) {
                    if (first_line && prefix[0] != '\0') {
                        ImGui::TextUnformatted(prefix);
                        ImGui::SameLine(0, 0);
                        first_line = false;
                    } else if (!first_line && msg.type == ConsoleMessageType::Input) {
                        ImGui::TextUnformatted("... ");
                        ImGui::SameLine(0, 0);
                    }
                    ImGui::TextUnformatted(line.c_str());
                }

                ImGui::PopStyleColor();
            }

            // Context menu for copy
            if (ImGui::BeginPopupContextWindow("##console_context")) {
                if (ImGui::MenuItem("Copy All", "Ctrl+C")) {
                    ImGui::SetClipboardText(getAllText().c_str());
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Clear")) {
                    clear();
                }
                ImGui::Separator();
                ImGui::MenuItem("Auto-scroll", nullptr, &auto_scroll_);
                ImGui::EndPopup();
            }

            // Ctrl+C to copy all when focused
            if (ImGui::IsWindowFocused() && ImGui::GetIO().KeyCtrl &&
                ImGui::IsKeyPressed(ImGuiKey_C, false)) {
                ImGui::SetClipboardText(getAllText().c_str());
            }

            // Auto scroll
            if (scroll_to_bottom_ && auto_scroll_) {
                ImGui::SetScrollHereY(1.0f);
                scroll_to_bottom_ = false;
            }
        }
        ImGui::EndChild();

        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
    }

    std::string ConsoleOutput::getAllText() const {
        std::ostringstream oss;
        for (const auto& msg : messages_) {
            switch (msg.type) {
            case ConsoleMessageType::Input:
                oss << ">>> " << msg.text << "\n";
                break;
            default:
                oss << msg.text << "\n";
                break;
            }
        }
        return oss.str();
    }

    std::string ConsoleOutput::getSelectedText() const {
        return getAllText();
    }

    void ConsoleOutput::copyToClipboard(const std::string& text) {
        ImGui::SetClipboardText(text.c_str());
    }

} // namespace lfs::vis::editor
