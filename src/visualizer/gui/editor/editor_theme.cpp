/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "editor_theme.hpp"
#include "theme/theme.hpp"

namespace lfs::vis::editor {

    TextEditor::Palette createEditorPalette(const Theme& theme) {
        auto palette = TextEditor::GetDarkPalette();

        // Map application theme colors to editor palette
        const auto& p = theme.palette;

        // Convert ImVec4 to ImU32
        auto toU32 = [](const ImVec4& c) {
            return IM_COL32(
                static_cast<int>(c.x * 255),
                static_cast<int>(c.y * 255),
                static_cast<int>(c.z * 255),
                static_cast<int>(c.w * 255));
        };

        // Background and editor chrome
        palette[static_cast<size_t>(TextEditor::PaletteIndex::Background)] = toU32(p.surface);
        palette[static_cast<size_t>(TextEditor::PaletteIndex::Cursor)] = toU32(p.text);
        palette[static_cast<size_t>(TextEditor::PaletteIndex::Selection)] = toU32(withAlpha(p.primary, 0.35f));
        palette[static_cast<size_t>(TextEditor::PaletteIndex::LineNumber)] = toU32(p.text_dim);
        palette[static_cast<size_t>(TextEditor::PaletteIndex::CurrentLineFill)] = toU32(withAlpha(p.surface_bright, 0.15f));
        palette[static_cast<size_t>(TextEditor::PaletteIndex::CurrentLineFillInactive)] = toU32(withAlpha(p.surface_bright, 0.08f));
        palette[static_cast<size_t>(TextEditor::PaletteIndex::CurrentLineEdge)] = toU32(withAlpha(p.border, 0.3f));

        // Syntax colors - using semantic mappings
        palette[static_cast<size_t>(TextEditor::PaletteIndex::Default)] = toU32(p.text);
        palette[static_cast<size_t>(TextEditor::PaletteIndex::Identifier)] = toU32(p.text);

        // Keywords in primary color (blue-ish accent)
        palette[static_cast<size_t>(TextEditor::PaletteIndex::Keyword)] = toU32(p.primary);

        // Known identifiers (builtins) slightly different
        palette[static_cast<size_t>(TextEditor::PaletteIndex::KnownIdentifier)] = toU32(lighten(p.secondary, 0.1f));

        // Numbers in secondary color
        palette[static_cast<size_t>(TextEditor::PaletteIndex::Number)] = toU32(p.warning);

        // Strings in success color (greenish)
        palette[static_cast<size_t>(TextEditor::PaletteIndex::String)] = toU32(p.success);
        palette[static_cast<size_t>(TextEditor::PaletteIndex::CharLiteral)] = toU32(p.success);

        // Comments in dim text
        palette[static_cast<size_t>(TextEditor::PaletteIndex::Comment)] = toU32(p.text_dim);
        palette[static_cast<size_t>(TextEditor::PaletteIndex::MultiLineComment)] = toU32(p.text_dim);

        // Preprocessor/decorators in info color
        palette[static_cast<size_t>(TextEditor::PaletteIndex::Preprocessor)] = toU32(p.info);
        palette[static_cast<size_t>(TextEditor::PaletteIndex::PreprocIdentifier)] = toU32(p.info);

        // Punctuation slightly dimmer
        palette[static_cast<size_t>(TextEditor::PaletteIndex::Punctuation)] = toU32(darken(p.text, 0.15f));

        // Error markers
        palette[static_cast<size_t>(TextEditor::PaletteIndex::ErrorMarker)] = toU32(withAlpha(p.error, 0.3f));
        palette[static_cast<size_t>(TextEditor::PaletteIndex::Breakpoint)] = toU32(p.error);

        return palette;
    }

    void applyThemeToEditor(TextEditor& editor, const Theme& theme) {
        editor.SetPalette(createEditorPalette(theme));
    }

} // namespace lfs::vis::editor
