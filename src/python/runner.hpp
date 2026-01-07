/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <expected>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace lfs::python {

    /**
     * @brief Execute a list of Python script files. Each script is expected to import `lichtfeld`
     *        and register its callbacks (e.g., with register_opacity_scaler or Session hooks).
     *
     * @return std::expected<void, std::string> error on failure (file missing, execution error,
     *         or interpreter unavailable when bindings are disabled).
     */
    std::expected<void, std::string> run_scripts(const std::vector<std::filesystem::path>& scripts);

    /**
     * @brief Set the callback for Python stdout/stderr capture.
     * @param callback Function(text, is_error) called when Python prints output.
     */
    void set_output_callback(std::function<void(const std::string&, bool)> callback);

    /**
     * @brief Initialize Python interpreter if not already done.
     */
    void ensure_initialized();

    /**
     * @brief Install Python stdout/stderr redirect. Call after Python is initialized.
     */
    void install_output_redirect();

    /**
     * @brief Finalize Python interpreter. Call before program exit to avoid cleanup issues.
     */
    void finalize();

} // namespace lfs::python
