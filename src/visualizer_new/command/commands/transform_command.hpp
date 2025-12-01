/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command/command.hpp"
#include <glm/glm.hpp>
#include <string>

namespace lfs::vis {
    class SceneManager;
}

namespace lfs::vis::command {

    class TransformCommand : public Command {
    public:
        TransformCommand(SceneManager* scene_manager,
                         std::string node_name,
                         const glm::mat4& old_transform,
                         const glm::mat4& new_transform);

        void undo() override;
        void redo() override;
        std::string getName() const override { return "Transform"; }

    private:
        SceneManager* scene_manager_;
        std::string node_name_;
        glm::mat4 old_transform_;
        glm::mat4 new_transform_;
    };

} // namespace lfs::vis::command
