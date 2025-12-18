/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <imgui.h>

namespace lfs::vis::gui {

    class SaveDirectoryPopup {
    public:
        using ConfirmCallback = std::function<void(const std::filesystem::path& dataset_path,
                                                   const std::filesystem::path& output_path)>;

        void render(const ImVec2& viewport_pos = {0, 0}, const ImVec2& viewport_size = {0, 0});
        void show(const std::filesystem::path& dataset_path);
        void setOnConfirm(ConfirmCallback callback) { on_confirm_ = std::move(callback); }
        [[nodiscard]] bool isOpen() const { return popup_open_; }

    private:
        [[nodiscard]] static std::filesystem::path deriveDefaultOutputPath(const std::filesystem::path& dataset_path);

        ConfirmCallback on_confirm_;
        std::filesystem::path dataset_path_;
        std::string output_path_buffer_;
        bool popup_open_ = false;
        bool should_open_ = false;
    };

} // namespace lfs::vis::gui
