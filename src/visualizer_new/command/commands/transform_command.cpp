/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "transform_command.hpp"
#include "core/services.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::command {

    TransformCommand::TransformCommand(std::string node_name,
                                        const glm::mat4& old_transform,
                                        const glm::mat4& new_transform)
        : node_name_(std::move(node_name))
        , old_transform_(old_transform)
        , new_transform_(new_transform) {}

    void TransformCommand::undo() {
        if (auto* sm = services().sceneOrNull()) {
            sm->setNodeTransform(node_name_, old_transform_);
        }
    }

    void TransformCommand::redo() {
        if (auto* sm = services().sceneOrNull()) {
            sm->setNodeTransform(node_name_, new_transform_);
        }
    }

    MultiTransformCommand::MultiTransformCommand(std::vector<std::string> node_names,
                                                  std::vector<glm::mat4> old_transforms,
                                                  std::vector<glm::mat4> new_transforms)
        : node_names_(std::move(node_names))
        , old_transforms_(std::move(old_transforms))
        , new_transforms_(std::move(new_transforms)) {}

    void MultiTransformCommand::undo() {
        auto* sm = services().sceneOrNull();
        if (!sm) return;
        for (size_t i = 0; i < node_names_.size(); ++i) {
            sm->setNodeTransform(node_names_[i], old_transforms_[i]);
        }
    }

    void MultiTransformCommand::redo() {
        auto* sm = services().sceneOrNull();
        if (!sm) return;
        for (size_t i = 0; i < node_names_.size(); ++i) {
            sm->setNodeTransform(node_names_[i], new_transforms_[i]);
        }
    }

} // namespace lfs::vis::command
