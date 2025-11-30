/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/transform_panel.hpp"
#include "gui/utils/crop_box_sync.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer_impl.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <imgui.h>

namespace lfs::vis::gui::panels {

    namespace {
        // Translation step sizes (Ctrl = faster)
        constexpr float TRANSLATE_STEP = 0.01f;
        constexpr float TRANSLATE_STEP_FAST = 0.1f;
        constexpr float TRANSLATE_STEP_CTRL = 0.1f;
        constexpr float TRANSLATE_STEP_CTRL_FAST = 1.0f;

        // Rotation step sizes (fixed, no Ctrl modifier)
        constexpr float ROTATE_STEP = 1.0f;
        constexpr float ROTATE_STEP_FAST = 15.0f;

        // Scale step sizes (fixed, no Ctrl modifier)
        constexpr float SCALE_STEP = 0.01f;
        constexpr float SCALE_STEP_FAST = 0.1f;
        constexpr float MIN_SCALE = 0.001f;

        constexpr float INPUT_WIDTH_PADDING = 40.0f;
    } // namespace

    void DrawTransformControls(const UIContext& ctx, const ToolMode current_tool) {
        const bool is_transform_tool = (current_tool == ToolMode::Translate ||
                                        current_tool == ToolMode::Rotate ||
                                        current_tool == ToolMode::Scale);
        if (!is_transform_tool) return;

        auto* const scene_manager = ctx.viewer->getSceneManager();
        if (!scene_manager) return;

        const std::string node_name = scene_manager->getSelectedNodeName();
        if (node_name.empty()) return;

        const char* header_label = nullptr;
        switch (current_tool) {
            case ToolMode::Translate: header_label = "Translate"; break;
            case ToolMode::Rotate:    header_label = "Rotate"; break;
            case ToolMode::Scale:     header_label = "Scale"; break;
            default: return;
        }

        if (!ImGui::CollapsingHeader(header_label, ImGuiTreeNodeFlags_DefaultOpen)) return;

        const glm::mat4 old_transform = scene_manager->getSelectedNodeTransform();

        // Decompose current transform
        glm::vec3 scale;
        glm::quat rotation;
        glm::vec3 translation;
        glm::vec3 skew;
        glm::vec4 perspective;
        glm::decompose(old_transform, scale, rotation, translation, skew, perspective);

        glm::vec3 euler = glm::degrees(glm::eulerAngles(rotation));

        bool changed = false;
        const bool ctrl_pressed = ImGui::GetIO().KeyCtrl;
        const float translate_step = ctrl_pressed ? TRANSLATE_STEP_CTRL : TRANSLATE_STEP;
        const float translate_step_fast = ctrl_pressed ? TRANSLATE_STEP_CTRL_FAST : TRANSLATE_STEP_FAST;
        const float text_width = ImGui::CalcTextSize("-000.000").x + ImGui::GetStyle().FramePadding.x * 2.0f + INPUT_WIDTH_PADDING;

        ImGui::Text("Node: %s", node_name.c_str());
        ImGui::Separator();

        if (current_tool == ToolMode::Translate) {
            ImGui::Text("Position:");
            ImGui::Text("X:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##PosX", &translation.x, translate_step, translate_step_fast, "%.3f");

            ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##PosY", &translation.y, translate_step, translate_step_fast, "%.3f");

            ImGui::Text("Z:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##PosZ", &translation.z, translate_step, translate_step_fast, "%.3f");
        }

        if (current_tool == ToolMode::Rotate) {
            ImGui::Text("Rotation (degrees):");
            ImGui::Text("X:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##RotX", &euler.x, ROTATE_STEP, ROTATE_STEP_FAST, "%.1f");

            ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##RotY", &euler.y, ROTATE_STEP, ROTATE_STEP_FAST, "%.1f");

            ImGui::Text("Z:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##RotZ", &euler.z, ROTATE_STEP, ROTATE_STEP_FAST, "%.1f");

            if (changed) {
                rotation = glm::quat(glm::radians(euler));
            }
        }

        if (current_tool == ToolMode::Scale) {
            ImGui::Text("Scale:");

            // Uniform scale
            float uniform = (scale.x + scale.y + scale.z) / 3.0f;
            ImGui::Text("U:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            if (ImGui::InputFloat("##ScaleU", &uniform, SCALE_STEP, SCALE_STEP_FAST, "%.3f")) {
                uniform = std::max(uniform, MIN_SCALE);
                scale = glm::vec3(uniform);
                changed = true;
            }

            ImGui::Separator();

            // Per-axis scale
            ImGui::Text("X:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##ScaleX", &scale.x, SCALE_STEP, SCALE_STEP_FAST, "%.3f");

            ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##ScaleY", &scale.y, SCALE_STEP, SCALE_STEP_FAST, "%.3f");

            ImGui::Text("Z:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##ScaleZ", &scale.z, SCALE_STEP, SCALE_STEP_FAST, "%.3f");

            scale = glm::max(scale, glm::vec3(MIN_SCALE));
        }

        if (changed) {
            const glm::mat4 new_transform = glm::translate(glm::mat4(1.0f), translation) *
                                            glm::mat4_cast(rotation) *
                                            glm::scale(glm::mat4(1.0f), scale);

            const glm::mat4 delta_matrix = new_transform * glm::inverse(old_transform);

            scene_manager->setSelectedNodeTransform(new_transform);

            // Sync crop box
            if (auto* const rm = ctx.viewer->getRenderingManager()) {
                auto settings = rm->getSettings();
                utils::applyCropBoxDelta(settings, delta_matrix);
                rm->updateSettings(settings);
            }
        }

        ImGui::Separator();
        if (ImGui::Button("Reset Transform")) {
            const glm::mat4 delta_matrix = glm::inverse(old_transform);
            scene_manager->setSelectedNodeTransform(glm::mat4(1.0f));

            // Sync crop box
            if (auto* const rm = ctx.viewer->getRenderingManager()) {
                auto settings = rm->getSettings();
                utils::applyCropBoxDelta(settings, delta_matrix);
                rm->updateSettings(settings);
            }
        }
    }

} // namespace lfs::vis::gui::panels
