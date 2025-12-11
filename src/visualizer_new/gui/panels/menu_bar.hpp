/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include "input/input_bindings.hpp"
#include <chrono>
#include <functional>
#include <optional>

namespace lfs::vis::gui {

    class MenuBar {
    public:
        MenuBar();
        ~MenuBar();

        void render();
        void setFonts(const FontSet& fonts) { fonts_ = fonts; }

        void setOnNewProject(std::function<void()> callback);
        void setOnImportDataset(std::function<void()> callback);
        void setOnOpenProject(std::function<void()> callback);
        void setOnImportPLY(std::function<void()> callback);
        void setOnSaveProjectAs(std::function<void()> callback);
        void setOnSaveProject(std::function<void()> callback);
        void setOnExport(std::function<void()> callback);
        void setOnExit(std::function<void()> callback);

        void renderGettingStartedWindow();
        void renderAboutWindow();
        void renderInputSettingsWindow();

        void setIsProjectTemp(bool is_temp) { is_project_temp_ = is_temp; }
        [[nodiscard]] bool getIsProjectTemp() const { return is_project_temp_; }

        void setInputBindings(input::InputBindings* bindings) { input_bindings_ = bindings; }

        bool isCapturingInput() const { return rebinding_action_.has_value(); }
        bool isInputSettingsOpen() const { return show_input_settings_; }
        void captureKey(int key, int mods);
        void captureMouseButton(int button, int mods);
        void cancelCapture();

    private:
        static constexpr double DOUBLE_CLICK_WAIT_TIME = 0.35;

        void openURL(const char* url);
        void renderBindingRow(input::Action action, input::ToolMode mode);
        void updateCapture();

        std::function<void()> on_new_project_;
        std::function<void()> on_import_dataset_;
        std::function<void()> on_open_project_;
        std::function<void()> on_import_ply_;
        std::function<void()> on_save_project_as_;
        std::function<void()> on_save_project_;
        std::function<void()> on_export_;
        std::function<void()> on_exit_;

        bool show_about_window_ = false;
        bool show_getting_started_ = false;
        bool show_input_settings_ = false;
        bool is_project_temp_ = true;

        input::InputBindings* input_bindings_ = nullptr;

        std::optional<input::Action> rebinding_action_;
        input::ToolMode rebinding_mode_ = input::ToolMode::GLOBAL;
        input::ToolMode selected_tool_mode_ = input::ToolMode::GLOBAL;

        bool waiting_for_double_click_ = false;
        int pending_button_ = -1;
        int pending_mods_ = 0;
        std::chrono::steady_clock::time_point first_click_time_;

        FontSet fonts_;
    };

} // namespace lfs::vis::gui
