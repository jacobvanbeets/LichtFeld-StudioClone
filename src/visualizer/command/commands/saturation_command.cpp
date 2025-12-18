/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "saturation_command.hpp"
#include "core/splat_data.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::command {

    SaturationCommand::SaturationCommand(SceneManager* scene_manager,
                                         std::string node_name,
                                         std::shared_ptr<lfs::core::Tensor> old_sh0,
                                         std::shared_ptr<lfs::core::Tensor> new_sh0)
        : scene_manager_(scene_manager),
          node_name_(std::move(node_name)),
          old_sh0_(std::move(old_sh0)),
          new_sh0_(std::move(new_sh0)) {}

    void SaturationCommand::applySH0(const std::shared_ptr<lfs::core::Tensor>& sh0) {
        if (!scene_manager_ || !sh0 || !sh0->is_valid())
            return;

        auto* node = scene_manager_->getScene().getMutableNode(node_name_);
        if (!node || !node->model)
            return;

        auto& model_sh0 = node->model->sh0();
        if (!model_sh0.is_valid())
            return;

        // Copy the tensor data
        model_sh0.copy_from(*sh0);
    }

    void SaturationCommand::undo() {
        applySH0(old_sh0_);
    }

    void SaturationCommand::redo() {
        applySH0(new_sh0_);
    }

} // namespace lfs::vis::command
