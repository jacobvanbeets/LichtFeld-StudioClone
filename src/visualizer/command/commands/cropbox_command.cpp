/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "cropbox_command.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::command {

    CropBoxCommand::CropBoxCommand(SceneManager* scene_manager,
                                   const std::string& cropbox_node_name,
                                   const CropBoxState& old_state,
                                   const CropBoxState& new_state)
        : scene_manager_(scene_manager),
          cropbox_node_name_(cropbox_node_name),
          old_state_(old_state),
          new_state_(new_state) {}

    void CropBoxCommand::undo() {
        applyState(old_state_);
    }

    void CropBoxCommand::redo() {
        applyState(new_state_);
    }

    void CropBoxCommand::applyState(const CropBoxState& state) {
        if (!scene_manager_)
            return;

        auto* node = scene_manager_->getScene().getMutableNode(cropbox_node_name_);
        if (!node || !node->cropbox)
            return;

        node->cropbox->min = state.min;
        node->cropbox->max = state.max;
        node->cropbox->inverse = state.inverse;
        node->local_transform = state.local_transform;
        node->transform_dirty = true;

        scene_manager_->getScene().invalidateCache();
        scene_manager_->syncCropBoxToRenderSettings();
    }

} // namespace lfs::vis::command
