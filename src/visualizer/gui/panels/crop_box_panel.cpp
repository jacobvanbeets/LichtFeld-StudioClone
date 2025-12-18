/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/crop_box_panel.hpp"
#include "command/command_history.hpp"
#include "command/commands/cropbox_command.hpp"
#include "gui/ui_widgets.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer_impl.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <optional>
#include <imgui.h>

namespace lfs::vis::gui::panels {

    using namespace lfs::core::events;

    namespace {
        constexpr float POSITION_STEP = 0.01f;
        constexpr float POSITION_STEP_FAST = 0.1f;
        constexpr float ROTATION_STEP = 1.0f;
        constexpr float ROTATION_STEP_FAST = 15.0f;
        constexpr float MIN_SIZE = 0.001f;
        constexpr float INPUT_WIDTH_PADDING = 40.0f;

        std::optional<command::CropBoxState> s_state_before_edit;
        std::string s_editing_cropbox_name;
        bool s_editing_active = false;

        command::CropBoxState captureState(const SceneNode* node) {
            if (!node || !node->cropbox) {
                return {};
            }
            return {
                .min = node->cropbox->min,
                .max = node->cropbox->max,
                .local_transform = node->local_transform.get(),
                .inverse = node->cropbox->inverse};
        }

        void commitUndoIfChanged(VisualizerImpl* viewer, SceneManager* sm, const std::string& node_name) {
            if (!s_state_before_edit.has_value())
                return;

            const auto* node = sm->getScene().getNode(node_name);
            if (!node)
                return;

            const auto new_state = captureState(node);
            const bool changed = (s_state_before_edit->min != new_state.min ||
                                  s_state_before_edit->max != new_state.max ||
                                  s_state_before_edit->inverse != new_state.inverse ||
                                  s_state_before_edit->local_transform != new_state.local_transform);

            if (changed) {
                auto cmd = std::make_unique<command::CropBoxCommand>(
                    sm, node_name, *s_state_before_edit, new_state);
                viewer->getCommandHistory().execute(std::move(cmd));
            }
            s_state_before_edit.reset();
        }

        glm::vec3 matrixToEulerDegrees(const glm::mat3& rot) {
            float pitch, yaw, roll;
            glm::extractEulerAngleXYZ(glm::mat4(rot), pitch, yaw, roll);
            return glm::degrees(glm::vec3(pitch, yaw, roll));
        }

        glm::mat3 eulerDegreesToMatrix(const glm::vec3& euler) {
            const glm::vec3 rad = glm::radians(euler);
            return glm::mat3(glm::eulerAngleXYZ(rad.x, rad.y, rad.z));
        }
    } // namespace

    void DrawCropBoxControls(const UIContext& ctx) {
        auto* const sm = ctx.viewer->getSceneManager();
        auto* const rm = ctx.viewer->getRenderingManager();
        if (!sm || !rm)
            return;

        if (!ImGui::CollapsingHeader("Crop Box", ImGuiTreeNodeFlags_DefaultOpen))
            return;

        const auto& settings = rm->getSettings();
        if (!settings.show_crop_box) {
            ImGui::TextDisabled("Crop box not visible");
            return;
        }

        // Get selected cropbox node
        const NodeId cropbox_id = sm->getSelectedNodeCropBoxId();
        if (cropbox_id == NULL_NODE) {
            ImGui::TextDisabled("No cropbox selected");
            return;
        }

        auto* node = sm->getScene().getMutableNode(sm->getScene().getNodeById(cropbox_id)->name);
        if (!node || !node->cropbox) {
            ImGui::TextDisabled("Invalid cropbox");
            return;
        }

        bool changed = false;
        bool any_active = false;
        bool any_deactivated = false;

        const float width = ImGui::CalcTextSize("-000.000").x + ImGui::GetStyle().FramePadding.x * 2.0f + INPUT_WIDTH_PADDING;

        // Decompose local_transform
        glm::vec3 translation, scale, skew;
        glm::quat rotation;
        glm::vec4 perspective;
        glm::decompose(node->local_transform.get(), scale, rotation, translation, skew, perspective);
        glm::vec3 euler = matrixToEulerDegrees(glm::mat3_cast(rotation));

        // Position (translation)
        if (ImGui::TreeNodeEx("Position", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("X:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##PosX", &translation.x, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Y:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##PosY", &translation.y, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Z:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##PosZ", &translation.z, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();
            ImGui::TreePop();
        }

        // Rotation (euler angles)
        if (ImGui::TreeNodeEx("Rotation", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("X:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##RotX", &euler.x, ROTATION_STEP, ROTATION_STEP_FAST, "%.1f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Y:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##RotY", &euler.y, ROTATION_STEP, ROTATION_STEP_FAST, "%.1f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Z:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##RotZ", &euler.z, ROTATION_STEP, ROTATION_STEP_FAST, "%.1f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();
            ImGui::TreePop();
        }

        // Size (bounds)
        if (ImGui::TreeNodeEx("Size", ImGuiTreeNodeFlags_DefaultOpen)) {
            glm::vec3 size = node->cropbox->max - node->cropbox->min;
            const glm::vec3 center = (node->cropbox->min + node->cropbox->max) * 0.5f;

            ImGui::Text("X:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##SizeX", &size.x, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Y:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##SizeY", &size.y, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Z:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##SizeZ", &size.z, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            size = glm::max(size, glm::vec3(MIN_SIZE));

            if (changed) {
                node->cropbox->min = center - size * 0.5f;
                node->cropbox->max = center + size * 0.5f;
            }
            ImGui::TreePop();
        }

        // Appearance
        if (ImGui::TreeNode("Appearance")) {
            float color[3] = {node->cropbox->color.x, node->cropbox->color.y, node->cropbox->color.z};
            if (ImGui::ColorEdit3("Color", color)) {
                node->cropbox->color = glm::vec3(color[0], color[1], color[2]);
                changed = true;
            }
            changed |= ImGui::SliderFloat("Line Width", &node->cropbox->line_width, 0.5f, 10.0f);
            ImGui::TreePop();
        }

        // Undo tracking
        if (any_active && !s_editing_active) {
            s_editing_active = true;
            s_editing_cropbox_name = node->name;
            s_state_before_edit = captureState(node);
        }

        if (changed) {
            // Rebuild transform from edited values
            const glm::mat3 rot_mat = eulerDegreesToMatrix(euler);
            node->local_transform = glm::translate(glm::mat4(1.0f), translation) * glm::mat4(rot_mat);
            node->transform_dirty = true;
            sm->getScene().invalidateCache();
            rm->markDirty();
        }

        if (any_deactivated && s_editing_active) {
            s_editing_active = false;
            commitUndoIfChanged(ctx.viewer, sm, s_editing_cropbox_name);
        }

        if (!any_active && s_editing_active) {
            s_editing_active = false;
            commitUndoIfChanged(ctx.viewer, sm, s_editing_cropbox_name);
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled("Enter: Commit crop | Ctrl+C: Copy contents");
    }

    const CropBoxState& getCropBoxState() {
        return CropBoxState::getInstance();
    }

} // namespace lfs::vis::gui::panels
