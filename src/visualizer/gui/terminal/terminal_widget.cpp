/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "terminal_widget.hpp"
#include <algorithm>
#include <core/logger.hpp>
#include <cstring>

namespace lfs::vis::terminal {

    namespace {

        constexpr ImU32 COLOR_PALETTE[] = {
            IM_COL32(30, 30, 30, 255),    // Black
            IM_COL32(180, 90, 90, 255),   // Red
            IM_COL32(90, 160, 90, 255),   // Green
            IM_COL32(180, 160, 90, 255),  // Yellow
            IM_COL32(90, 120, 180, 255),  // Blue
            IM_COL32(160, 100, 160, 255), // Magenta
            IM_COL32(90, 160, 160, 255),  // Cyan
            IM_COL32(180, 180, 180, 255), // White
            IM_COL32(100, 100, 100, 255), // Bright Black
            IM_COL32(210, 110, 110, 255), // Bright Red
            IM_COL32(120, 190, 120, 255), // Bright Green
            IM_COL32(210, 190, 110, 255), // Bright Yellow
            IM_COL32(120, 150, 210, 255), // Bright Blue
            IM_COL32(190, 130, 190, 255), // Bright Magenta
            IM_COL32(120, 190, 190, 255), // Bright Cyan
            IM_COL32(220, 220, 220, 255), // Bright White
        };

        constexpr ImU32 BG_COLOR = IM_COL32(30, 30, 30, 255);
        constexpr ImU32 CURSOR_COLOR = IM_COL32(200, 200, 200, 180);
        constexpr ImU32 SELECTION_COLOR = IM_COL32(100, 100, 200, 200);
        constexpr ImU32 DEFAULT_FG = IM_COL32(229, 229, 229, 255);

        struct KeyMapping {
            ImGuiKey key;
            const char* seq;
        };

        constexpr KeyMapping KEY_MAPPINGS[] = {
            {ImGuiKey_Enter, "\r"},
            {ImGuiKey_Backspace, "\x7f"},
            {ImGuiKey_Tab, "\t"},
            {ImGuiKey_Escape, "\x1b"},
            {ImGuiKey_UpArrow, "\x1b[A"},
            {ImGuiKey_DownArrow, "\x1b[B"},
            {ImGuiKey_RightArrow, "\x1b[C"},
            {ImGuiKey_LeftArrow, "\x1b[D"},
            {ImGuiKey_Home, "\x1b[H"},
            {ImGuiKey_End, "\x1b[F"},
            {ImGuiKey_PageUp, "\x1b[5~"},
            {ImGuiKey_PageDown, "\x1b[6~"},
            {ImGuiKey_Delete, "\x1b[3~"},
            {ImGuiKey_Insert, "\x1b[2~"},
            {ImGuiKey_F1, "\x1bOP"},
            {ImGuiKey_F2, "\x1bOQ"},
            {ImGuiKey_F3, "\x1bOR"},
            {ImGuiKey_F4, "\x1bOS"},
            {ImGuiKey_F5, "\x1b[15~"},
            {ImGuiKey_F6, "\x1b[17~"},
            {ImGuiKey_F7, "\x1b[18~"},
            {ImGuiKey_F8, "\x1b[19~"},
            {ImGuiKey_F9, "\x1b[20~"},
            {ImGuiKey_F10, "\x1b[21~"},
            {ImGuiKey_F11, "\x1b[23~"},
            {ImGuiKey_F12, "\x1b[24~"},
        };

        size_t encodeUtf8(uint32_t codepoint, char (&out)[4]) {
            if (codepoint < 0x80) {
                out[0] = static_cast<char>(codepoint);
                return 1;
            }
            if (codepoint < 0x800) {
                out[0] = static_cast<char>(0xC0 | (codepoint >> 6));
                out[1] = static_cast<char>(0x80 | (codepoint & 0x3F));
                return 2;
            }
            if (codepoint < 0x10000) {
                out[0] = static_cast<char>(0xE0 | (codepoint >> 12));
                out[1] = static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                out[2] = static_cast<char>(0x80 | (codepoint & 0x3F));
                return 3;
            }
            if (codepoint > 0x10FFFF) {
                return 0;
            }
            out[0] = static_cast<char>(0xF0 | (codepoint >> 18));
            out[1] = static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
            out[2] = static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            out[3] = static_cast<char>(0x80 | (codepoint & 0x3F));
            return 4;
        }

    } // namespace

    TerminalWidget::TerminalWidget(int cols, int rows) : cols_(cols),
                                                         rows_(rows) {
        initVterm();
    }

    TerminalWidget::~TerminalWidget() {
        destroyVterm();
    }

    void TerminalWidget::initVterm() {
        vt_ = vterm_new(rows_, cols_);
        vterm_set_utf8(vt_, true);
        screen_ = vterm_obtain_screen(vt_);

        static VTermScreenCallbacks callbacks = {
            .damage = onDamage,
            .moverect = nullptr,
            .movecursor = onMoveCursor,
            .settermprop = nullptr,
            .bell = onBell,
            .resize = onResize,
            .sb_pushline = onPushline,
            .sb_popline = onPopline,
            .sb_clear = nullptr,
            .sb_pushline4 = nullptr,
        };

        vterm_screen_set_callbacks(screen_, &callbacks, this);
        vterm_screen_reset(screen_, 1);
    }

    void TerminalWidget::destroyVterm() {
        if (vt_) {
            vterm_free(vt_);
            vt_ = nullptr;
            screen_ = nullptr;
        }
    }

    void TerminalWidget::spawn(const std::string& shell) {
        pty_.close();
        destroyVterm();
        initVterm();
        scrollback_.clear();
        scroll_offset_ = 0;
        needs_redraw_ = true;

        if (!pty_.spawn(shell, cols_, rows_)) {
            LOG_ERROR("Failed to spawn: {}", shell.empty() ? "default shell" : shell);
        }
    }

    void TerminalWidget::pump() {
        if (!pty_.is_running())
            return;

        ssize_t n;
        while ((n = pty_.read(read_buffer_, sizeof(read_buffer_))) > 0) {
            std::lock_guard lock(mutex_);
            vterm_input_write(vt_, read_buffer_, static_cast<size_t>(n));
            has_new_output_ = true;
            needs_redraw_ = true;
        }
    }

    bool TerminalWidget::render(ImFont* mono_font) {
        pump();

        if (mono_font)
            ImGui::PushFont(mono_font);

        char_width_ = ImGui::CalcTextSize("M").x;
        char_height_ = ImGui::GetTextLineHeight();

        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const int new_cols = std::max(1, static_cast<int>(avail.x / char_width_));
        const int new_rows = std::max(1, static_cast<int>(avail.y / char_height_));

        if (new_cols != cols_ || new_rows != rows_) {
            handleResize(new_cols, new_rows);
        }

        const ImVec2 origin = ImGui::GetCursorScreenPos();

        ImGui::InvisibleButton("##terminal_area", avail, ImGuiButtonFlags_MouseButtonLeft);
        const bool area_hovered = ImGui::IsItemHovered();
        const bool area_clicked = ImGui::IsItemClicked();

        if (area_clicked)
            is_focused_ = true;
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !area_hovered)
            is_focused_ = false;

        if (is_focused_ && !read_only_) {
            processInput();
            ImGui::GetIO().InputQueueCharacters.resize(0);
        }

        // Ctrl+C to copy
        if (is_focused_ && ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_C, false)) {
            const std::string sel = getSelection();
            if (!sel.empty())
                ImGui::SetClipboardText(sel.c_str());
        }

        // Context menu
        if (area_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            ImGui::OpenPopup("##terminal_context");
        }
        if (ImGui::BeginPopup("##terminal_context")) {
            if (ImGui::MenuItem("Copy", "Ctrl+C")) {
                const std::string sel = getSelection();
                if (!sel.empty())
                    ImGui::SetClipboardText(sel.c_str());
            }
            if (!read_only_ && ImGui::MenuItem("Paste", "Ctrl+Shift+V")) {
                if (const char* clipboard = ImGui::GetClipboardText()) {
                    paste(clipboard);
                }
            }
            if (ImGui::MenuItem("Clear"))
                clear();
            ImGui::EndPopup();
        }

        // Mouse selection
        if (area_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            const ImVec2 mouse = ImGui::GetMousePos();
            const int col = static_cast<int>((mouse.x - origin.x) / char_width_);
            const int row = static_cast<int>((mouse.y - origin.y) / char_height_);
            selection_start_ = {row, col};
            selection_end_ = selection_start_;
            is_selecting_ = true;
        }

        if (is_selecting_ && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            const ImVec2 mouse = ImGui::GetMousePos();
            selection_end_ = {
                static_cast<int>((mouse.y - origin.y) / char_height_),
                static_cast<int>((mouse.x - origin.x) / char_width_)};
            needs_redraw_ = true;
        }

        if (is_selecting_ && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            is_selecting_ = false;
            const std::string sel = getSelection();
            if (!sel.empty())
                ImGui::SetClipboardText(sel.c_str());
        }

        ImDrawList* const dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(origin, {origin.x + avail.x, origin.y + avail.y}, BG_COLOR);

        {
            std::lock_guard lock(mutex_);
            for (int row = 0; row < rows_; ++row) {
                for (int col = 0; col < cols_; ++col) {
                    drawCell(dl, row, col, origin);
                }
            }
        }

        // Cursor
        if (cursor_visible_ && is_focused_) {
            cursor_blink_time_ += ImGui::GetIO().DeltaTime;
            const bool show = (static_cast<int>(cursor_blink_time_ * 2) % 2) == 0;
            if (show && scroll_offset_ == 0) {
                const ImVec2 cpos = {origin.x + cursor_pos_.col * char_width_,
                                     origin.y + cursor_pos_.row * char_height_};
                dl->AddRectFilled(cpos, {cpos.x + char_width_, cpos.y + char_height_}, CURSOR_COLOR);
            }
        }

        if (mono_font)
            ImGui::PopFont();

        const bool changed = needs_redraw_;
        needs_redraw_ = false;
        has_new_output_ = false;
        return changed;
    }

    void TerminalWidget::processInput() {
        ImGuiIO& io = ImGui::GetIO();

        for (int i = 0; i < io.InputQueueCharacters.Size; ++i) {
            char utf8[4];
            const size_t len = encodeUtf8(io.InputQueueCharacters[i], utf8);
            if (len == 0)
                continue;
            if (pty_.write(utf8, len) < 0) {
                LOG_ERROR("PTY write failed");
                return;
            }
        }

        if (io.KeyCtrl) {
            for (int key = ImGuiKey_A; key <= ImGuiKey_Z; ++key) {
                if (ImGui::IsKeyPressed(static_cast<ImGuiKey>(key), false)) {
                    const char c = static_cast<char>(1 + (key - ImGuiKey_A));
                    if (pty_.write(&c, 1) < 0) {
                        LOG_ERROR("PTY write failed");
                    }
                    return;
                }
            }
        }

        for (const auto& m : KEY_MAPPINGS) {
            if (ImGui::IsKeyPressed(m.key, true)) {
                if (pty_.write(m.seq, std::strlen(m.seq)) < 0) {
                    LOG_ERROR("PTY write failed");
                    return;
                }
                scrollToBottom();
                return;
            }
        }

        if (const float wheel = io.MouseWheel; wheel != 0.0f && ImGui::IsWindowHovered()) {
            if (wheel > 0)
                scrollUp(3);
            else
                scrollDown(3);
        }

        if (io.KeyCtrl && io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_V, false)) {
            if (const char* clipboard = ImGui::GetClipboardText()) {
                paste(clipboard);
            }
        }
    }

    void TerminalWidget::drawCell(ImDrawList* dl, int row, int col, const ImVec2& origin) {
        const VTermPos pos = {row, col};
        VTermScreenCell cell;
        vterm_screen_get_cell(screen_, pos, &cell);

        const ImVec2 p0 = {origin.x + col * char_width_, origin.y + row * char_height_};
        const ImVec2 p1 = {p0.x + char_width_, p0.y + char_height_};

        // Selection check
        bool in_selection = false;
        if (selection_start_.row != selection_end_.row || selection_start_.col != selection_end_.col) {
            VTermPos start = selection_start_, end = selection_end_;
            if (start.row > end.row || (start.row == end.row && start.col > end.col)) {
                std::swap(start, end);
            }
            in_selection = (row > start.row || (row == start.row && col >= start.col)) &&
                           (row < end.row || (row == end.row && col <= end.col));
        }

        // Background
        ImU32 bg = vtermColorToImU32(cell.bg);
        if (cell.attrs.reverse)
            bg = vtermColorToImU32(cell.fg);
        if (in_selection)
            bg = SELECTION_COLOR;
        if (bg != IM_COL32(0, 0, 0, 0)) {
            dl->AddRectFilled(p0, p1, bg);
        }

        // Character
        if (cell.chars[0] != 0) {
            char utf8[16];
            size_t total_len = 0;
            for (int i = 0; i < VTERM_MAX_CHARS_PER_CELL && cell.chars[i]; ++i) {
                char tmp[4];
                const size_t char_len = encodeUtf8(cell.chars[i], tmp);
                if (char_len > 0 && total_len + char_len < sizeof(utf8)) {
                    std::memcpy(utf8 + total_len, tmp, char_len);
                    total_len += char_len;
                }
            }
            utf8[total_len] = '\0';

            ImU32 fg = vtermColorToImU32(cell.fg);
            if (cell.attrs.reverse) {
                fg = vtermColorToImU32(cell.bg);
                if (fg == IM_COL32(0, 0, 0, 0))
                    fg = BG_COLOR;
            }
            if (fg == IM_COL32(0, 0, 0, 0))
                fg = DEFAULT_FG;

            dl->AddText(p0, fg, utf8);
        }
    }

    void TerminalWidget::handleResize(int new_cols, int new_rows) {
        if (new_cols == cols_ && new_rows == rows_)
            return;

        cols_ = new_cols;
        rows_ = new_rows;

        {
            std::lock_guard lock(mutex_);
            vterm_set_size(vt_, rows_, cols_);
        }

        pty_.resize(cols_, rows_);
        needs_redraw_ = true;
    }

    ImU32 TerminalWidget::vtermColorToImU32(VTermColor color) const {
        if (VTERM_COLOR_IS_DEFAULT_FG(&color) || VTERM_COLOR_IS_DEFAULT_BG(&color)) {
            return IM_COL32(0, 0, 0, 0);
        }
        if (VTERM_COLOR_IS_INDEXED(&color)) {
            vterm_screen_convert_color_to_rgb(screen_, &color);
        }
        if (VTERM_COLOR_IS_RGB(&color)) {
            return IM_COL32(color.rgb.red, color.rgb.green, color.rgb.blue, 255);
        }
        return DEFAULT_FG;
    }

    void TerminalWidget::scrollUp(int lines) {
        scroll_offset_ = std::min(scroll_offset_ + lines, static_cast<int>(scrollback_.size()));
        needs_redraw_ = true;
    }

    void TerminalWidget::scrollDown(int lines) {
        scroll_offset_ = std::max(0, scroll_offset_ - lines);
        needs_redraw_ = true;
    }

    void TerminalWidget::scrollToBottom() {
        scroll_offset_ = 0;
        needs_redraw_ = true;
    }

    std::string TerminalWidget::getSelection() const {
        if (selection_start_.row == selection_end_.row && selection_start_.col == selection_end_.col) {
            return {};
        }

        VTermPos start = selection_start_, end = selection_end_;
        if (start.row > end.row || (start.row == end.row && start.col > end.col)) {
            std::swap(start, end);
        }

        std::string result;
        for (int row = start.row; row <= end.row; ++row) {
            const int col_start = (row == start.row) ? start.col : 0;
            const int col_end = (row == end.row) ? end.col : cols_ - 1;

            for (int col = col_start; col <= col_end; ++col) {
                VTermScreenCell cell;
                vterm_screen_get_cell(screen_, {row, col}, &cell);
                if (cell.chars[0]) {
                    char utf8[4];
                    const size_t len = encodeUtf8(cell.chars[0], utf8);
                    if (len > 0)
                        result.append(utf8, len);
                } else {
                    result += ' ';
                }
            }
            if (row < end.row)
                result += '\n';
        }
        return result;
    }

    void TerminalWidget::paste(const std::string& text) {
        if (pty_.write("\x1b[200~", 6) < 0 ||
            pty_.write(text.c_str(), text.size()) < 0 ||
            pty_.write("\x1b[201~", 6) < 0) {
            LOG_ERROR("PTY paste failed");
            return;
        }
        scrollToBottom();
    }

    void TerminalWidget::clear() {
        if (pty_.is_running()) {
            if (pty_.write("\x0c", 1) < 0) {
                LOG_ERROR("PTY clear failed");
            }
        } else {
            reset();
        }
    }

    void TerminalWidget::write(const char* data, size_t len) {
        std::lock_guard lock(mutex_);
        vterm_input_write(vt_, data, len);
        has_new_output_ = true;
        needs_redraw_ = true;
    }

    void TerminalWidget::reset() {
        std::lock_guard lock(mutex_);
        vterm_screen_reset(screen_, 1);
        scrollback_.clear();
        scroll_offset_ = 0;
        cursor_pos_ = {0, 0};
        needs_redraw_ = true;
    }

    void TerminalWidget::sendToPty(const std::string& text) {
        if (!pty_.is_running())
            return;
        if (pty_.write(text.c_str(), text.size()) < 0 ||
            pty_.write("\n", 1) < 0) {
            LOG_ERROR("PTY send failed");
            return;
        }
        scrollToBottom();
    }

    void TerminalWidget::interrupt() {
        if (pty_.is_running()) {
            pty_.interrupt();
        }
    }

    int TerminalWidget::onDamage(VTermRect, void* user) {
        static_cast<TerminalWidget*>(user)->needs_redraw_ = true;
        return 0;
    }

    int TerminalWidget::onMoveCursor(VTermPos pos, VTermPos, int visible, void* user) {
        auto* self = static_cast<TerminalWidget*>(user);
        self->cursor_pos_ = pos;
        self->cursor_visible_ = visible != 0;
        self->needs_redraw_ = true;
        return 0;
    }

    int TerminalWidget::onBell(void*) {
        return 0;
    }

    int TerminalWidget::onResize(int rows, int cols, void* user) {
        auto* self = static_cast<TerminalWidget*>(user);
        self->rows_ = rows;
        self->cols_ = cols;
        self->needs_redraw_ = true;
        return 0;
    }

    int TerminalWidget::onPushline(int cols, const VTermScreenCell* cells, void* user) {
        auto* self = static_cast<TerminalWidget*>(user);
        self->scrollback_.push_front({std::vector<VTermScreenCell>(cells, cells + cols)});
        while (self->scrollback_.size() > MAX_SCROLLBACK) {
            self->scrollback_.pop_back();
        }
        return 0;
    }

    int TerminalWidget::onPopline(int cols, VTermScreenCell* cells, void* user) {
        auto* self = static_cast<TerminalWidget*>(user);
        if (self->scrollback_.empty())
            return 0;

        const auto& line = self->scrollback_.front();
        const int copy_cols = std::min(cols, static_cast<int>(line.cells.size()));
        std::memcpy(cells, line.cells.data(), copy_cols * sizeof(VTermScreenCell));

        for (int i = copy_cols; i < cols; ++i) {
            std::memset(&cells[i], 0, sizeof(VTermScreenCell));
        }

        self->scrollback_.pop_front();
        return 1;
    }

} // namespace lfs::vis::terminal
