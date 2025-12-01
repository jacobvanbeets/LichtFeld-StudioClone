/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "transform_command.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::command {

    TransformCommand::TransformCommand(SceneManager* scene_manager,
                                        std::string node_name,
                                        const glm::mat4& old_transform,
                                        const glm::mat4& new_transform)
        : scene_manager_(scene_manager)
        , node_name_(std::move(node_name))
        , old_transform_(old_transform)
        , new_transform_(new_transform) {}

    void TransformCommand::undo() {
        if (scene_manager_) {
            scene_manager_->setNodeTransform(node_name_, old_transform_);
        }
    }

    void TransformCommand::redo() {
        if (scene_manager_) {
            scene_manager_->setNodeTransform(node_name_, new_transform_);
        }
    }

} // namespace lfs::vis::command
