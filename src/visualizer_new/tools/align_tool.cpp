/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/align_tool.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "internal/viewport.hpp"
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <glm/gtc/matrix_transform.hpp>

namespace lfs::vis::tools {

    AlignTool::AlignTool() = default;

    bool AlignTool::initialize(const ToolContext& ctx) {
        tool_context_ = &ctx;
        return true;
    }

    void AlignTool::shutdown() {
        tool_context_ = nullptr;
        reset();
    }

    void AlignTool::update([[maybe_unused]] const ToolContext& ctx) {}

    static ImVec2 projectToScreen(const glm::vec3& world_pos, const Viewport& viewport) {
        const glm::mat4 view = viewport.getViewMatrix();
        const glm::mat4 proj = viewport.getProjectionMatrix();
        const glm::vec4 clip_pos = proj * view * glm::vec4(world_pos, 1.0f);

        if (clip_pos.w <= 0.0f) return ImVec2(-1000, -1000);

        const glm::vec3 ndc = glm::vec3(clip_pos) / clip_pos.w;
        return ImVec2(
            (ndc.x * 0.5f + 0.5f) * viewport.windowSize.x,
            (1.0f - (ndc.y * 0.5f + 0.5f)) * viewport.windowSize.y
        );
    }

    static float calculateScreenRadius(const glm::vec3& world_pos, const float world_radius, const Viewport& viewport) {
        const glm::mat4 view = viewport.getViewMatrix();
        const glm::mat4 proj = viewport.getProjectionMatrix();
        const glm::vec4 view_pos = view * glm::vec4(world_pos, 1.0f);
        const float depth = -view_pos.z;

        if (depth <= 0.0f) return 0.0f;

        const float screen_radius = (world_radius * proj[1][1] * viewport.windowSize.y) / (2.0f * depth);
        return glm::clamp(screen_radius, 5.0f, 50.0f);
    }

    void AlignTool::renderUI([[maybe_unused]] const lfs::vis::gui::UIContext& ui_ctx,
                              [[maybe_unused]] bool* p_open) {
        if (!isEnabled() || !tool_context_) return;

        ImDrawList* const draw_list = ImGui::GetForegroundDrawList();
        const ImVec2 mouse_pos = ImGui::GetMousePos();
        const auto& viewport = tool_context_->getViewport();
        auto* const rendering_manager = tool_context_->getRenderingManager();
        const bool over_gui = ImGui::GetIO().WantCaptureMouse;

        constexpr float kSphereWorldRadius = 0.05f;
        constexpr ImU32 kSphereColor = IM_COL32(220, 50, 50, 255);
        constexpr ImU32 kSphereOutline = IM_COL32(255, 255, 255, 200);
        constexpr ImU32 kPreviewColor = IM_COL32(220, 50, 50, 150);
        constexpr ImU32 kCrosshairColor = IM_COL32(255, 0, 0, 200);

        // Draw picked points (always visible)
        for (size_t i = 0; i < picked_points_.size(); ++i) {
            const ImVec2 screen_pos = projectToScreen(picked_points_[i], viewport);
            const float screen_radius = calculateScreenRadius(picked_points_[i], kSphereWorldRadius, viewport);

            draw_list->AddCircleFilled(screen_pos, screen_radius, kSphereColor, 32);
            draw_list->AddCircle(screen_pos, screen_radius, kSphereOutline, 32, 1.5f);

            const char label = '1' + static_cast<char>(i);
            draw_list->AddText(ImVec2(screen_pos.x - 4, screen_pos.y - 6), IM_COL32(255, 255, 255, 255), &label, &label + 1);
        }

        // Hide mouse-following elements when over GUI
        if (over_gui) return;

        draw_list->AddCircle(mouse_pos, 5.0f, kCrosshairColor, 16, 2.0f);

        // Live preview at mouse position
        if (picked_points_.size() < 3 && rendering_manager) {
            const float depth = rendering_manager->getDepthAtPixel(
                static_cast<int>(mouse_pos.x), static_cast<int>(mouse_pos.y));

            if (depth > 0.0f && depth < 1e9f) {
                const glm::vec3 preview_point = unprojectScreenPoint(mouse_pos.x, mouse_pos.y, *tool_context_);
                if (preview_point.x > -1e9f) {
                    const ImVec2 screen_pos = projectToScreen(preview_point, viewport);
                    const float screen_radius = calculateScreenRadius(preview_point, kSphereWorldRadius, viewport);

                    draw_list->AddCircleFilled(screen_pos, screen_radius, kPreviewColor, 32);
                    draw_list->AddCircle(screen_pos, screen_radius, IM_COL32(255, 255, 255, 150), 32, 1.5f);

                    const char label = '1' + static_cast<char>(picked_points_.size());
                    draw_list->AddText(ImVec2(screen_pos.x - 4, screen_pos.y - 6), IM_COL32(255, 255, 255, 180), &label, &label + 1);
                }
            }
        }

        // Normal preview when 2 points picked
        if (picked_points_.size() == 2 && rendering_manager) {
            const float depth = rendering_manager->getDepthAtPixel(
                static_cast<int>(mouse_pos.x), static_cast<int>(mouse_pos.y));

            if (depth > 0.0f && depth < 1e9f) {
                const glm::vec3 p2 = unprojectScreenPoint(mouse_pos.x, mouse_pos.y, *tool_context_);
                if (p2.x > -1e9f) {
                    const glm::vec3& p0 = picked_points_[0];
                    const glm::vec3& p1 = picked_points_[1];

                    const glm::vec3 v01 = p1 - p0;
                    const glm::vec3 v02 = p2 - p0;
                    glm::vec3 normal = glm::normalize(glm::cross(v01, v02));
                    if (normal.y > 0.0f) normal = -normal;

                    const glm::vec3 center = (p0 + p1 + p2) / 3.0f;
                    const float line_length = glm::max(glm::length(v01) * 0.5f, 0.1f);
                    const glm::vec3 normal_end = center + normal * line_length;

                    const ImVec2 center_screen = projectToScreen(center, viewport);
                    const ImVec2 normal_screen = projectToScreen(normal_end, viewport);

                    draw_list->AddLine(center_screen, normal_screen, IM_COL32(255, 255, 0, 255), 4.0f);
                    draw_list->AddCircleFilled(normal_screen, 10.0f, IM_COL32(255, 255, 0, 255));
                    draw_list->AddText(ImVec2(normal_screen.x + 12, normal_screen.y - 8), IM_COL32(255, 255, 0, 255), "UP");

                    const ImVec2 p0_screen = projectToScreen(p0, viewport);
                    const ImVec2 p1_screen = projectToScreen(p1, viewport);
                    const ImVec2 p2_screen = projectToScreen(p2, viewport);
                    draw_list->AddLine(p0_screen, p1_screen, IM_COL32(255, 0, 0, 200), 2.0f);
                    draw_list->AddLine(p1_screen, p2_screen, IM_COL32(0, 255, 0, 200), 2.0f);
                    draw_list->AddLine(p2_screen, p0_screen, IM_COL32(0, 0, 255, 200), 2.0f);
                }
            }
        }

        // Instructions
        const char* instruction = nullptr;
        switch (picked_points_.size()) {
            case 0: instruction = "Click 1st point"; break;
            case 1: instruction = "Click 2nd point"; break;
            case 2: instruction = "Click 3rd point"; break;
            default: break;
        }
        if (instruction) {
            draw_list->AddText(ImVec2(mouse_pos.x + 15, mouse_pos.y - 10), kCrosshairColor, instruction);
        }

        char count_text[16];
        snprintf(count_text, sizeof(count_text), "Points: %zu/3", picked_points_.size());
        draw_list->AddText(ImVec2(10, 50), IM_COL32(255, 255, 255, 200), count_text);
    }

    bool AlignTool::handleMouseButton(int button, int action, double x, double y, const ToolContext& ctx) {
        if (!isEnabled()) return false;

        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            const glm::vec3 world_pos = unprojectScreenPoint(x, y, ctx);
            if (world_pos.x > -1e9f) {
                picked_points_.push_back(world_pos);
                ctx.requestRender();

                if (picked_points_.size() == 3) {
                    applyAlignment(ctx);
                    reset();
                }
                return true;
            }
        }

        if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
            reset();
            ctx.requestRender();
            return true;
        }

        return false;
    }

    void AlignTool::onEnabledChanged(bool enabled) {
        if (!enabled) reset();
    }

    void AlignTool::reset() {
        picked_points_.clear();
    }

    glm::vec3 AlignTool::unprojectScreenPoint(double x, double y, const ToolContext& ctx) {
        constexpr glm::vec3 INVALID_POS(-1e10f);

        auto* const rendering_manager = ctx.getRenderingManager();
        if (!rendering_manager) return INVALID_POS;

        const float depth = rendering_manager->getDepthAtPixel(static_cast<int>(x), static_cast<int>(y));
        if (depth < 0.0f) return INVALID_POS;

        const auto& viewport = ctx.getViewport();
        return viewport.unprojectPixel(static_cast<float>(x), static_cast<float>(y), depth,
                                       rendering_manager->getFovDegrees());
    }

    void AlignTool::applyAlignment(const ToolContext& ctx) {
        if (picked_points_.size() != 3) return;

        auto* const scene_manager = ctx.getSceneManager();
        if (!scene_manager) return;

        auto& scene = scene_manager->getScene();

        const glm::vec3& p0 = picked_points_[0];
        const glm::vec3& p1 = picked_points_[1];
        const glm::vec3& p2 = picked_points_[2];

        const glm::vec3 v01 = p1 - p0;
        const glm::vec3 v02 = p2 - p0;
        glm::vec3 normal = glm::normalize(glm::cross(v01, v02));
        const glm::vec3 center = (p0 + p1 + p2) / 3.0f;

        // Ensure normal points downward (-Y)
        if (normal.y > 0.0f) normal = -normal;

        // Rotation to align normal with -Y
        constexpr glm::vec3 kTargetUp(0.0f, -1.0f, 0.0f);
        const glm::vec3 axis = glm::cross(normal, kTargetUp);
        const float axis_len = glm::length(axis);

        glm::mat4 rotation(1.0f);
        if (axis_len > 1e-6f) {
            const float angle = acos(glm::clamp(glm::dot(normal, kTargetUp), -1.0f, 1.0f));
            rotation = glm::rotate(glm::mat4(1.0f), angle, glm::normalize(axis));
        } else if (glm::dot(normal, kTargetUp) < 0.0f) {
            rotation = glm::rotate(glm::mat4(1.0f), glm::pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
        }

        // Transform: translate to origin, rotate, translate to Y=0
        const glm::mat4 to_origin = glm::translate(glm::mat4(1.0f), -center);
        const glm::mat4 from_origin = glm::translate(glm::mat4(1.0f), glm::vec3(center.x, 0.0f, center.z));
        const glm::mat4 transform = from_origin * rotation * to_origin;

        // Apply to all scene nodes
        for (const auto* node : scene.getNodes()) {
            scene_manager->setNodeTransform(node->name, transform * node->local_transform);
        }

        ctx.requestRender();
    }

} // namespace lfs::vis::tools
