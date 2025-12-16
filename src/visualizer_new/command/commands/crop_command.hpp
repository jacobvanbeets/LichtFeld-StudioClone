/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command/command.hpp"
#include "core_new/tensor.hpp"
#include <string>

namespace lfs::vis::command {

    // Undo/redo command for soft crop operations using deletion masks
    // Uses services() to access SceneManager - no stored pointer
    class CropCommand : public Command {
    public:
        CropCommand(std::string node_name,
                    lfs::core::Tensor old_deleted_mask,
                    lfs::core::Tensor new_deleted_mask);

        void undo() override;
        void redo() override;
        [[nodiscard]] std::string getName() const override { return "Crop"; }

    private:
        const std::string node_name_;
        const lfs::core::Tensor old_deleted_mask_;
        const lfs::core::Tensor new_deleted_mask_;
    };

} // namespace lfs::vis::command
