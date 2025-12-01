/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/transform_panel.hpp"
#include "command/command_history.hpp"
#include "command/commands/composite_command.hpp"
#include "command/commands/cropbox_command.hpp"
#include "command/commands/transform_command.hpp"
#include "gui/utils/crop_box_sync.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer_impl.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <imgui.h>

namespace lfs::vis::gui::panels {

namespace {
    constexpr float TRANSLATE_STEP = 0.01f;
    constexpr float TRANSLATE_STEP_FAST = 0.1f;
    constexpr float TRANSLATE_STEP_CTRL = 0.1f;
    constexpr float TRANSLATE_STEP_CTRL_FAST = 1.0f;
    constexpr float ROTATE_STEP = 1.0f;
    constexpr float ROTATE_STEP_FAST = 15.0f;
    constexpr float SCALE_STEP = 0.01f;
    constexpr float SCALE_STEP_FAST = 0.1f;
    constexpr float MIN_SCALE = 0.001f;
    constexpr float INPUT_WIDTH_PADDING = 40.0f;

    struct DecomposedTransform {
        glm::vec3 translation;
        glm::quat rotation;
        glm::vec3 scale;
    };

    DecomposedTransform decompose(const glm::mat4& m) {
        DecomposedTransform d;
        glm::vec3 skew;
        glm::vec4 perspective;
        glm::decompose(m, d.scale, d.rotation, d.translation, skew, perspective);
        return d;
    }

    command::CropBoxState getCropState(const RenderingManager* rm) {
        const auto& s = rm->getSettings();
        return {s.crop_min, s.crop_max, s.crop_transform, s.crop_inverse};
    }
} // namespace

void DrawTransformControls(const UIContext& ctx, const ToolMode current_tool,
                           const TransformSpace transform_space, TransformPanelState& state) {
    if (current_tool != ToolMode::Translate &&
        current_tool != ToolMode::Rotate &&
        current_tool != ToolMode::Scale) {
        return;
    }

    auto* const scene_manager = ctx.viewer->getSceneManager();
    auto* const render_manager = ctx.viewer->getRenderingManager();
    if (!scene_manager || !render_manager) return;

    const std::string& node_name = scene_manager->getSelectedNodeName();
    if (node_name.empty()) return;

    const char* header_label = nullptr;
    switch (current_tool) {
        case ToolMode::Translate: header_label = "Translate"; break;
        case ToolMode::Rotate:    header_label = "Rotate"; break;
        case ToolMode::Scale:     header_label = "Scale"; break;
        default: return;
    }

    if (!ImGui::CollapsingHeader(header_label, ImGuiTreeNodeFlags_DefaultOpen)) return;

    const bool use_world_space = (transform_space == TransformSpace::World);
    const glm::mat4 current_transform = scene_manager->getSelectedNodeTransform();
    auto [translation, rotation, scale] = decompose(current_transform);

    // For world space rotation: use tracked cumulative values
    // For local space: use decomposed euler
    glm::vec3 euler = use_world_space ? state.world_euler
                                      : glm::degrees(glm::eulerAngles(rotation));

    const bool ctrl_pressed = ImGui::GetIO().KeyCtrl;
    const float translate_step = ctrl_pressed ? TRANSLATE_STEP_CTRL : TRANSLATE_STEP;
    const float translate_step_fast = ctrl_pressed ? TRANSLATE_STEP_CTRL_FAST : TRANSLATE_STEP_FAST;
    const float text_width = ImGui::CalcTextSize("-000.000").x +
                             ImGui::GetStyle().FramePadding.x * 2.0f + INPUT_WIDTH_PADDING;

    ImGui::Text("Node: %s", node_name.c_str());
    ImGui::Text("Space: %s", use_world_space ? "World" : "Local");
    ImGui::Separator();

    bool changed = false;
    bool any_active = false;

    if (current_tool == ToolMode::Translate) {
        ImGui::Text("Position:");
        ImGui::Text("X:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(text_width);
        changed |= ImGui::InputFloat("##PosX", &translation.x, translate_step, translate_step_fast, "%.3f");
        any_active |= ImGui::IsItemActive();

        ImGui::Text("Y:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(text_width);
        changed |= ImGui::InputFloat("##PosY", &translation.y, translate_step, translate_step_fast, "%.3f");
        any_active |= ImGui::IsItemActive();

        ImGui::Text("Z:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(text_width);
        changed |= ImGui::InputFloat("##PosZ", &translation.z, translate_step, translate_step_fast, "%.3f");
        any_active |= ImGui::IsItemActive();
    }

    if (current_tool == ToolMode::Rotate) {
        ImGui::Text("Rotation (degrees):");
        ImGui::Text("X:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(text_width);
        changed |= ImGui::InputFloat("##RotX", &euler.x, ROTATE_STEP, ROTATE_STEP_FAST, "%.1f");
        any_active |= ImGui::IsItemActive();

        ImGui::Text("Y:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(text_width);
        changed |= ImGui::InputFloat("##RotY", &euler.y, ROTATE_STEP, ROTATE_STEP_FAST, "%.1f");
        any_active |= ImGui::IsItemActive();

        ImGui::Text("Z:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(text_width);
        changed |= ImGui::InputFloat("##RotZ", &euler.z, ROTATE_STEP, ROTATE_STEP_FAST, "%.1f");
        any_active |= ImGui::IsItemActive();

        if (changed && !use_world_space) {
            rotation = glm::quat(glm::radians(euler));
        }
    }

    if (current_tool == ToolMode::Scale) {
        ImGui::Text("Scale:");

        float uniform = (scale.x + scale.y + scale.z) / 3.0f;
        ImGui::Text("U:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(text_width);
        if (ImGui::InputFloat("##ScaleU", &uniform, SCALE_STEP, SCALE_STEP_FAST, "%.3f")) {
            uniform = std::max(uniform, MIN_SCALE);
            scale = glm::vec3(uniform);
            changed = true;
        }
        any_active |= ImGui::IsItemActive();

        ImGui::Separator();

        ImGui::Text("X:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(text_width);
        changed |= ImGui::InputFloat("##ScaleX", &scale.x, SCALE_STEP, SCALE_STEP_FAST, "%.3f");
        any_active |= ImGui::IsItemActive();

        ImGui::Text("Y:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(text_width);
        changed |= ImGui::InputFloat("##ScaleY", &scale.y, SCALE_STEP, SCALE_STEP_FAST, "%.3f");
        any_active |= ImGui::IsItemActive();

        ImGui::Text("Z:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(text_width);
        changed |= ImGui::InputFloat("##ScaleZ", &scale.z, SCALE_STEP, SCALE_STEP_FAST, "%.3f");
        any_active |= ImGui::IsItemActive();

        scale = glm::max(scale, glm::vec3(MIN_SCALE));
    }

    // Reset world euler tracking when node changes
    if (state.editing_node_name != node_name) {
        state.world_euler = glm::vec3(0.0f);
        state.prev_world_euler = glm::vec3(0.0f);
    }

    // Capture initial state before first change
    if ((any_active || changed) && !state.editing_active) {
        state.editing_active = true;
        state.editing_node_name = node_name;
        state.transform_before_edit = current_transform;
        state.initial_translation = translation;
        state.initial_scale = scale;
        state.prev_world_euler = state.world_euler;  // Continue from current
        state.crop_before_edit = getCropState(render_manager);
    }

    if (changed) {
        glm::mat4 new_transform;

        if (use_world_space) {
            if (current_tool == ToolMode::Translate) {
                new_transform = state.transform_before_edit;
                new_transform[3] = glm::vec4(translation, 1.0f);
            } else if (current_tool == ToolMode::Rotate) {
                // Incremental: apply only delta since last frame
                const glm::vec3 euler_delta = glm::radians(euler - state.prev_world_euler);
                const glm::quat rot_x = glm::angleAxis(euler_delta.x, glm::vec3(1, 0, 0));
                const glm::quat rot_y = glm::angleAxis(euler_delta.y, glm::vec3(0, 1, 0));
                const glm::quat rot_z = glm::angleAxis(euler_delta.z, glm::vec3(0, 0, 1));
                const glm::quat delta_rot = rot_x * rot_y * rot_z;

                // Apply to current rotation
                const glm::quat new_rot = delta_rot * rotation;
                new_transform = glm::translate(glm::mat4(1.0f), translation) *
                                glm::mat4_cast(new_rot) *
                                glm::scale(glm::mat4(1.0f), scale);

                state.prev_world_euler = euler;
                state.world_euler = euler;
            } else {
                const glm::mat3 initial_rs(state.transform_before_edit);
                const glm::vec3 scale_ratio = scale / state.initial_scale;
                new_transform = glm::mat4(glm::mat3(glm::scale(glm::mat4(1.0f), scale_ratio)) * initial_rs);
                new_transform[3] = state.transform_before_edit[3];
            }
        } else {
            new_transform = glm::translate(glm::mat4(1.0f), translation) *
                            glm::mat4_cast(rotation) *
                            glm::scale(glm::mat4(1.0f), scale);
        }

        scene_manager->setSelectedNodeTransform(new_transform);

        const glm::mat4 delta = new_transform * glm::inverse(current_transform);
        auto settings = render_manager->getSettings();
        utils::applyCropBoxDelta(settings, delta);
        render_manager->updateSettings(settings);
    }

    // Commit undo command when editing ends
    if (!any_active && state.editing_active) {
        state.editing_active = false;
        const glm::mat4 final_transform = scene_manager->getSelectedNodeTransform();
        if (state.transform_before_edit != final_transform) {
            auto composite = std::make_unique<command::CompositeCommand>();
            composite->add(std::make_unique<command::TransformCommand>(
                scene_manager, state.editing_node_name,
                state.transform_before_edit, final_transform));
            composite->add(std::make_unique<command::CropBoxCommand>(
                render_manager, state.crop_before_edit, getCropState(render_manager)));
            ctx.viewer->getCommandHistory().execute(std::move(composite));
        }
    }

    ImGui::Separator();
    if (ImGui::Button("Reset Transform")) {
        const command::CropBoxState old_crop = getCropState(render_manager);

        scene_manager->setSelectedNodeTransform(glm::mat4(1.0f));

        const glm::mat4 delta = glm::inverse(current_transform);
        auto settings = render_manager->getSettings();
        utils::applyCropBoxDelta(settings, delta);
        render_manager->updateSettings(settings);

        auto composite = std::make_unique<command::CompositeCommand>();
        composite->add(std::make_unique<command::TransformCommand>(
            scene_manager, node_name, current_transform, glm::mat4(1.0f)));
        composite->add(std::make_unique<command::CropBoxCommand>(
            render_manager, old_crop, getCropState(render_manager)));
        ctx.viewer->getCommandHistory().execute(std::move(composite));

        state.world_euler = glm::vec3(0.0f);
        state.prev_world_euler = glm::vec3(0.0f);
    }
}

} // namespace lfs::vis::gui::panels
