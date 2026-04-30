/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>
#include <glm/glm.hpp>

#include <optional>
#include <vector>

namespace lfs::vis::gui {

    struct ClipRect {
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
    };

    class LFS_VIS_API LineRenderer {
    public:
        LineRenderer() = default;

        void begin(int screen_w, int screen_h, int fb_w, int fb_h,
                   std::optional<ClipRect> clip_rect = std::nullopt);
        void addLine(glm::vec2 p0, glm::vec2 p1, glm::vec4 color, float thickness = 1.0f);
        void addTriangleFilled(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, glm::vec4 color);
        void addCircleFilled(glm::vec2 center, float radius, glm::vec4 color, int segments = 16);
        void end();

        void destroyResources();

    private:
        enum class CommandType {
            Line,
            Triangle,
            Circle
        };

        struct Command {
            CommandType type = CommandType::Line;
            glm::vec2 p0{0.0f};
            glm::vec2 p1{0.0f};
            glm::vec2 p2{0.0f};
            glm::vec4 color{1.0f};
            float thickness = 1.0f;
            int segments = 16;
        };

        std::vector<Command> commands_;
        std::optional<ClipRect> clip_rect_;
    };

} // namespace lfs::vis::gui
