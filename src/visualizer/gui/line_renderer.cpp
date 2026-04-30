/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/line_renderer.hpp"

#include <imgui.h>

namespace lfs::vis::gui {

    namespace {
        [[nodiscard]] ImVec2 toImVec2(const glm::vec2 v) {
            return ImVec2(v.x, v.y);
        }

        [[nodiscard]] ImU32 toImColor(const glm::vec4 color) {
            return ImGui::ColorConvertFloat4ToU32(ImVec4(color.r, color.g, color.b, color.a));
        }
    } // namespace

    void LineRenderer::begin(const int, const int, const int, const int,
                             const std::optional<ClipRect> clip_rect) {
        clip_rect_ = clip_rect;
        commands_.clear();
    }

    void LineRenderer::addLine(const glm::vec2 p0,
                               const glm::vec2 p1,
                               const glm::vec4 color,
                               const float thickness) {
        commands_.push_back({
            .type = CommandType::Line,
            .p0 = p0,
            .p1 = p1,
            .color = color,
            .thickness = thickness,
        });
    }

    void LineRenderer::addTriangleFilled(const glm::vec2 p0,
                                         const glm::vec2 p1,
                                         const glm::vec2 p2,
                                         const glm::vec4 color) {
        commands_.push_back({
            .type = CommandType::Triangle,
            .p0 = p0,
            .p1 = p1,
            .p2 = p2,
            .color = color,
        });
    }

    void LineRenderer::addCircleFilled(const glm::vec2 center,
                                       const float radius,
                                       const glm::vec4 color,
                                       const int segments) {
        commands_.push_back({
            .type = CommandType::Circle,
            .p0 = center,
            .color = color,
            .thickness = radius,
            .segments = segments,
        });
    }

    void LineRenderer::end() {
        if (commands_.empty()) {
            return;
        }

        ImDrawList* const draw_list = ImGui::GetBackgroundDrawList();
        if (!draw_list) {
            return;
        }

        if (clip_rect_) {
            const auto& clip = *clip_rect_;
            draw_list->PushClipRect(
                ImVec2(static_cast<float>(clip.x), static_cast<float>(clip.y)),
                ImVec2(static_cast<float>(clip.x + clip.width), static_cast<float>(clip.y + clip.height)),
                true);
        }

        for (const auto& command : commands_) {
            const ImU32 color = toImColor(command.color);
            switch (command.type) {
            case CommandType::Line:
                draw_list->AddLine(toImVec2(command.p0), toImVec2(command.p1), color, command.thickness);
                break;
            case CommandType::Triangle:
                draw_list->AddTriangleFilled(
                    toImVec2(command.p0), toImVec2(command.p1), toImVec2(command.p2), color);
                break;
            case CommandType::Circle:
                draw_list->AddCircleFilled(toImVec2(command.p0), command.thickness, color, command.segments);
                break;
            }
        }

        if (clip_rect_) {
            draw_list->PopClipRect();
        }
    }

    void LineRenderer::destroyResources() {
        commands_.clear();
    }

} // namespace lfs::vis::gui
