/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_command.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::command {

    SelectionCommand::SelectionCommand(SceneManager* scene_manager,
                                       std::shared_ptr<lfs::core::Tensor> old_selection,
                                       std::shared_ptr<lfs::core::Tensor> new_selection)
        : scene_manager_(scene_manager),
          old_selection_(std::move(old_selection)),
          new_selection_(std::move(new_selection)) {}

    void SelectionCommand::undo() {
        if (!scene_manager_)
            return;

        if (old_selection_ && old_selection_->is_valid()) {
            scene_manager_->getScene().setSelectionMask(old_selection_);
        } else {
            scene_manager_->getScene().clearSelection();
        }
    }

    void SelectionCommand::redo() {
        if (!scene_manager_)
            return;

        if (new_selection_ && new_selection_->is_valid()) {
            scene_manager_->getScene().setSelectionMask(new_selection_);
        } else {
            scene_manager_->getScene().clearSelection();
        }
    }

} // namespace lfs::vis::command
