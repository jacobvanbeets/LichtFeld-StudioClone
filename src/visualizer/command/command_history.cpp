/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "command_history.hpp"

namespace lfs::vis::command {

    void CommandHistory::execute(CommandPtr cmd) {
        if (!cmd)
            return;

        if (current_index_ < history_.size()) {
            history_.resize(current_index_);
        }

        history_.push_back(std::move(cmd));
        current_index_ = history_.size();
    }

    void CommandHistory::undo() {
        if (!canUndo())
            return;
        --current_index_;
        history_[current_index_]->undo();
    }

    void CommandHistory::redo() {
        if (!canRedo())
            return;
        history_[current_index_]->redo();
        ++current_index_;
    }

    void CommandHistory::clear() {
        history_.clear();
        current_index_ = 0;
    }

} // namespace lfs::vis::command
