/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/python_console_panel.hpp"
#include "gui/editor/console_output.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/ui_widgets.hpp"
#include "theme/theme.hpp"

#include <imgui.h>

#ifdef LFS_BUILD_PYTHON_BINDINGS
#include <Python.h>
#include <filesystem>
#include <mutex>

#include "python/runner.hpp"

namespace {
    std::once_flag g_console_init_once;
    std::once_flag g_syspath_init_once;

    void setup_sys_path() {
        std::call_once(g_syspath_init_once, [] {
            PyGILState_STATE gil = PyGILState_Ensure();

            // Add build directory (where lichtfeld.so lives) to sys.path
            const std::filesystem::path build_python_dir =
                std::filesystem::path(PROJECT_ROOT_PATH) / "build" / "src" / "python";

            PyObject* sys_path = PySys_GetObject("path");
            if (sys_path) {
                PyObject* py_path = PyUnicode_FromString(build_python_dir.string().c_str());
                if (py_path) {
                    PyList_Insert(sys_path, 0, py_path);
                    Py_DECREF(py_path);
                }
            }

            PyGILState_Release(gil);
        });
    }

    void setup_console_output_capture() {
        std::call_once(g_console_init_once, [] {
            lfs::python::set_output_callback([](const std::string& text, bool is_error) {
                auto& state = lfs::vis::gui::panels::PythonConsoleState::getInstance();
                auto* output = state.getConsoleOutput();
                if (output) {
                    if (is_error) {
                        output->addError(text);
                    } else {
                        output->addOutput(text);
                    }
                }
            });
        });
    }
} // namespace
#endif

namespace lfs::vis::gui::panels {

    namespace {
        // Color constants (legacy - kept for backward compatibility)
        constexpr uint32_t COLOR_INPUT = 0xFF88FF88;
        constexpr uint32_t COLOR_OUTPUT = 0xFFFFFFFF;
        constexpr uint32_t COLOR_ERROR = 0xFF5555FF;
    } // namespace

    PythonConsoleState::PythonConsoleState()
        : console_output_(std::make_unique<editor::ConsoleOutput>()),
          editor_(std::make_unique<editor::PythonEditor>()) {
    }

    PythonConsoleState::~PythonConsoleState() = default;

    PythonConsoleState& PythonConsoleState::getInstance() {
        static PythonConsoleState instance;
        return instance;
    }

    void PythonConsoleState::addOutput(const std::string& text, uint32_t color) {
        std::lock_guard lock(mutex_);
        messages_.push_back({text, color, false});
        while (messages_.size() > MAX_MESSAGES) {
            messages_.pop_front();
        }
        // Also add to new console output
        if (console_output_) {
            console_output_->addOutput(text);
        }
    }

    void PythonConsoleState::addError(const std::string& text) {
        std::lock_guard lock(mutex_);
        messages_.push_back({text, COLOR_ERROR, false});
        while (messages_.size() > MAX_MESSAGES) {
            messages_.pop_front();
        }
        if (console_output_) {
            console_output_->addError(text);
        }
    }

    void PythonConsoleState::addInput(const std::string& text) {
        std::lock_guard lock(mutex_);
        messages_.push_back({"> " + text, COLOR_INPUT, true});
        while (messages_.size() > MAX_MESSAGES) {
            messages_.pop_front();
        }
        if (console_output_) {
            console_output_->addInput(text);
        }
    }

    void PythonConsoleState::clear() {
        std::lock_guard lock(mutex_);
        messages_.clear();
        if (console_output_) {
            console_output_->clear();
        }
    }

    void PythonConsoleState::addToHistory(const std::string& cmd) {
        std::lock_guard lock(mutex_);
        if (!cmd.empty() && (command_history_.empty() || command_history_.back() != cmd)) {
            command_history_.push_back(cmd);
        }
        history_index_ = -1;
        if (editor_) {
            editor_->addToHistory(cmd);
        }
    }

    std::string PythonConsoleState::getHistoryEntry(int offset) const {
        std::lock_guard lock(mutex_);
        if (command_history_.empty())
            return "";
        int idx = static_cast<int>(command_history_.size()) + offset;
        if (idx >= 0 && idx < static_cast<int>(command_history_.size())) {
            return command_history_[idx];
        }
        return "";
    }

    void PythonConsoleState::historyUp() {
        std::lock_guard lock(mutex_);
        if (command_history_.empty())
            return;
        if (history_index_ < 0) {
            history_index_ = static_cast<int>(command_history_.size()) - 1;
        } else if (history_index_ > 0) {
            history_index_--;
        }
    }

    void PythonConsoleState::historyDown() {
        std::lock_guard lock(mutex_);
        if (history_index_ < 0)
            return;
        if (history_index_ < static_cast<int>(command_history_.size()) - 1) {
            history_index_++;
        } else {
            history_index_ = -1;
        }
    }

    editor::ConsoleOutput* PythonConsoleState::getConsoleOutput() {
        return console_output_.get();
    }

    editor::PythonEditor* PythonConsoleState::getEditor() {
        return editor_.get();
    }

    void DrawPythonConsole(const UIContext& /*ctx*/, bool* open) {
        if (!open || !*open)
            return;

#ifndef LFS_BUILD_PYTHON_BINDINGS
        ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Python Console", open)) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f),
                               "Python bindings not available. Rebuild with -DBUILD_PYTHON_BINDINGS=ON");
        }
        ImGui::End();
        return;
#else
        // Initialize Python and set up output capture
        lfs::python::ensure_initialized();
        lfs::python::install_output_redirect();
        setup_sys_path();
        setup_console_output_capture();

        auto& state = PythonConsoleState::getInstance();
        const auto& t = theme();

        ImGui::SetNextWindowSize(ImVec2(700, 500), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Python Console", open, ImGuiWindowFlags_MenuBar)) {
            ImGui::End();
            return;
        }

        // Menu bar
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("Console")) {
                if (ImGui::MenuItem("Clear", "Ctrl+L")) {
                    state.clear();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Copy All")) {
                    if (auto* output = state.getConsoleOutput()) {
                        ImGui::SetClipboardText(output->getAllText().c_str());
                    }
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Help")) {
                ImGui::MenuItem("Ctrl+Enter to execute", nullptr, false, false);
                ImGui::MenuItem("Ctrl+Space for autocomplete", nullptr, false, false);
                ImGui::MenuItem("Up/Down for history", nullptr, false, false);
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        // Calculate sizes
        const float input_height = 100.0f; // Height for multi-line input
        const float spacing = ImGui::GetStyle().ItemSpacing.y;
        ImVec2 content_size = ImGui::GetContentRegionAvail();

        // Output area (takes remaining space minus input area)
        ImVec2 output_size(content_size.x, content_size.y - input_height - spacing * 2);
        if (auto* output = state.getConsoleOutput()) {
            output->render(output_size);
        }

        ImGui::Spacing();

        // Prompt indicator
        ImGui::TextColored(t.palette.success, ">>>");
        ImGui::SameLine();

        // Input editor
        ImVec2 input_size(content_size.x - ImGui::GetCursorPosX() + ImGui::GetStyle().WindowPadding.x,
                          input_height);

        if (auto* editor = state.getEditor()) {
            if (editor->render(input_size)) {
                // Execute was triggered
                std::string cmd = editor->getText();

                // Trim whitespace
                while (!cmd.empty() && (cmd.back() == '\n' || cmd.back() == '\r' || cmd.back() == ' ')) {
                    cmd.pop_back();
                }

                if (!cmd.empty()) {
                    state.addInput(cmd);
                    state.addToHistory(cmd);

                    // Execute Python code
                    PyGILState_STATE gil = PyGILState_Ensure();

                    // For multi-line code, use Py_file_input mode
                    int start_mode = (cmd.find('\n') != std::string::npos) ? Py_file_input : Py_single_input;

                    PyObject* main_module = PyImport_AddModule("__main__");
                    PyObject* globals = PyModule_GetDict(main_module);

                    PyObject* result = PyRun_String(cmd.c_str(), start_mode, globals, globals);

                    if (result) {
                        if (result != Py_None && start_mode == Py_single_input) {
                            PyObject* str = PyObject_Str(result);
                            if (str) {
                                const char* output_str = PyUnicode_AsUTF8(str);
                                if (output_str && output_str[0] != '\0') {
                                    state.addOutput(output_str);
                                }
                                Py_DECREF(str);
                            }
                        }
                        Py_DECREF(result);
                    } else {
                        PyErr_Print();
                    }

                    PyGILState_Release(gil);

                    editor->clear();
                }
            }
        }

        // Handle Ctrl+L to clear
        if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_L)) {
            state.clear();
        }

        ImGui::End();
#endif // LFS_BUILD_PYTHON_BINDINGS
    }

} // namespace lfs::vis::gui::panels
