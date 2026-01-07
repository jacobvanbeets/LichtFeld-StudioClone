/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <TextEditor.h>

namespace lfs::vis {
    struct Theme;
}

namespace lfs::vis::editor {

    // Create TextEditor palette from application theme
    TextEditor::Palette createEditorPalette(const Theme& theme);

    // Update an existing TextEditor's palette from theme
    void applyThemeToEditor(TextEditor& editor, const Theme& theme);

} // namespace lfs::vis::editor
