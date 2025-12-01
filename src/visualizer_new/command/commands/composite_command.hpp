/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command/command.hpp"
#include <vector>

namespace lfs::vis::command {

    class CompositeCommand : public Command {
    public:
        void add(CommandPtr cmd) { commands_.push_back(std::move(cmd)); }

        void undo() override {
            for (auto it = commands_.rbegin(); it != commands_.rend(); ++it) {
                (*it)->undo();
            }
        }

        void redo() override {
            for (auto& cmd : commands_) {
                cmd->redo();
            }
        }

        std::string getName() const override {
            return commands_.empty() ? "Composite" : commands_[0]->getName();
        }

        [[nodiscard]] bool empty() const { return commands_.empty(); }

    private:
        std::vector<CommandPtr> commands_;
    };

} // namespace lfs::vis::command
