/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <filesystem>
#include <string>

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <Windows.h>
#include <Shobjidl.h>
#endif

namespace lfs::vis::gui {

#ifdef WIN32

    namespace utils {
        /**
         * Opens a native Windows file/folder selection dialog
         * @param strDirectory Output path selected by the user
         * @param rgSpec File type filters (can be nullptr)
         * @param cFileTypes Number of file type filters
         * @param blnDirectory True to select folders, false for files
         * @return HRESULT indicating success or failure
         */
        HRESULT selectFileNative(PWSTR& strDirectory,
                                 COMDLG_FILTERSPEC rgSpec[] = nullptr,
                                 UINT cFileTypes = 0,
                                 bool blnDirectory = false);

        /**
         * Opens a native Windows save file dialog
         * @param outPath Output path selected by the user
         * @param rgSpec File type filters (can be nullptr)
         * @param cFileTypes Number of file type filters
         * @param defaultName Default filename
         * @return HRESULT indicating success or failure
         */
        HRESULT saveFileNative(PWSTR& outPath,
                               COMDLG_FILTERSPEC rgSpec[] = nullptr,
                               UINT cFileTypes = 0,
                               const wchar_t* defaultName = nullptr);
    } // namespace utils

    // in windows- open file browser that search for lfs project
    void OpenProjectFileDialog();
    // in windows- open file browser that search for ply files
    void OpenPlyFileDialog();
    // in windows- open file browser that search directories
    void OpenDatasetFolderDialog();
#endif

    // Cross-platform file dialogs
    std::filesystem::path SavePlyFileDialog(const std::string& defaultName);
    std::filesystem::path SaveJsonFileDialog(const std::string& defaultName);
    std::filesystem::path OpenJsonFileDialog();
    std::filesystem::path SaveSogFileDialog(const std::string& defaultName);
    std::filesystem::path SaveHtmlFileDialog(const std::string& defaultName);

} // namespace lfs::vis::gui