/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "crop_command.hpp"
#include "core/services.hpp"
#include "core_new/logger.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::command {

    CropCommand::CropCommand(std::string node_name,
                             lfs::core::Tensor old_deleted_mask,
                             lfs::core::Tensor new_deleted_mask)
        : node_name_(std::move(node_name)),
          old_deleted_mask_(std::move(old_deleted_mask)),
          new_deleted_mask_(std::move(new_deleted_mask)) {
    }

    void CropCommand::undo() {
        auto* scene_manager = services().sceneOrNull();
        if (!scene_manager) return;

        auto& scene = scene_manager->getScene();
        auto* node = scene.getMutableNode(node_name_);
        if (!node || !node->model) {
            LOG_WARN("CropCommand::undo - node '{}' not found", node_name_);
            return;
        }

        node->model->deleted() = old_deleted_mask_.clone();
        scene.markDirty();
    }

    void CropCommand::redo() {
        auto* scene_manager = services().sceneOrNull();
        if (!scene_manager) return;

        auto& scene = scene_manager->getScene();
        auto* node = scene.getMutableNode(node_name_);
        if (!node || !node->model) {
            LOG_WARN("CropCommand::redo - node '{}' not found", node_name_);
            return;
        }

        node->model->deleted() = new_deleted_mask_.clone();
        scene.markDirty();
    }

} // namespace lfs::vis::command
