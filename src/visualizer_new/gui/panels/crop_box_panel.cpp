/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/crop_box_panel.hpp"
#include "command/command_history.hpp"
#include "command/commands/cropbox_command.hpp"
#include "gui/ui_widgets.hpp"
#include "rendering/rendering_manager.hpp"
#include "visualizer_impl.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <imgui.h>
#include <ImGuizmo.h>
#include <optional>

namespace lfs::vis::gui::panels {

    using namespace lfs::core::events;

    namespace {
        // Position/Size step sizes
        constexpr float POSITION_STEP = 0.01f;
        constexpr float POSITION_STEP_FAST = 0.1f;

        // Rotation step sizes
        constexpr float ROTATION_STEP = 1.0f;
        constexpr float ROTATION_STEP_FAST = 15.0f;

        // Size constraints
        constexpr float MIN_SIZE = 0.001f;

        constexpr float INPUT_WIDTH_PADDING = 40.0f;

        // Undo state tracking
        std::optional<command::CropBoxState> s_state_before_edit;
        bool s_editing_active = false;

        command::CropBoxState captureState(const RenderingManager* const rm) {
            const auto settings = rm->getSettings();
            return command::CropBoxState{
                .crop_min = settings.crop_min,
                .crop_max = settings.crop_max,
                .crop_transform = settings.crop_transform,
                .crop_inverse = settings.crop_inverse
            };
        }

        void commitUndoIfChanged(VisualizerImpl* const viewer, RenderingManager* const rm) {
            if (!s_state_before_edit.has_value()) return;

            const auto new_state = captureState(rm);
            const bool changed = (s_state_before_edit->crop_min != new_state.crop_min ||
                                  s_state_before_edit->crop_max != new_state.crop_max ||
                                  s_state_before_edit->crop_inverse != new_state.crop_inverse ||
                                  s_state_before_edit->crop_transform.getTranslation() != new_state.crop_transform.getTranslation() ||
                                  s_state_before_edit->crop_transform.getRotationMat() != new_state.crop_transform.getRotationMat());

            if (changed) {
                auto cmd = std::make_unique<command::CropBoxCommand>(rm, *s_state_before_edit, new_state);
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
        auto* const rm = ctx.viewer->getRenderingManager();
        if (!rm) return;

        if (!ImGui::CollapsingHeader("Crop Box", ImGuiTreeNodeFlags_DefaultOpen)) return;

        auto settings = rm->getSettings();
        if (!settings.show_crop_box) {
            ImGui::TextDisabled("Crop box not visible");
            return;
        }

        bool changed = false;
        bool any_active = false;
        bool any_deactivated = false;

        const float width = ImGui::CalcTextSize("-000.000").x + ImGui::GetStyle().FramePadding.x * 2.0f + INPUT_WIDTH_PADDING;

        // Position (translation)
        if (ImGui::TreeNodeEx("Position", ImGuiTreeNodeFlags_DefaultOpen)) {
            glm::vec3 pos = settings.crop_transform.getTranslation();

            ImGui::Text("X:"); ImGui::SameLine(); ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##PosX", &pos.x, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive(); any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Y:"); ImGui::SameLine(); ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##PosY", &pos.y, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive(); any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Z:"); ImGui::SameLine(); ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##PosZ", &pos.z, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive(); any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            if (changed) {
                settings.crop_transform = lfs::geometry::EuclideanTransform(
                    settings.crop_transform.getRotationMat(), pos);
            }
            ImGui::TreePop();
        }

        // Rotation (euler angles)
        if (ImGui::TreeNodeEx("Rotation", ImGuiTreeNodeFlags_DefaultOpen)) {
            glm::vec3 euler = matrixToEulerDegrees(settings.crop_transform.getRotationMat());

            ImGui::Text("X:"); ImGui::SameLine(); ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##RotX", &euler.x, ROTATION_STEP, ROTATION_STEP_FAST, "%.1f");
            any_active |= ImGui::IsItemActive(); any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Y:"); ImGui::SameLine(); ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##RotY", &euler.y, ROTATION_STEP, ROTATION_STEP_FAST, "%.1f");
            any_active |= ImGui::IsItemActive(); any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Z:"); ImGui::SameLine(); ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##RotZ", &euler.z, ROTATION_STEP, ROTATION_STEP_FAST, "%.1f");
            any_active |= ImGui::IsItemActive(); any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            if (changed) {
                settings.crop_transform = lfs::geometry::EuclideanTransform(
                    eulerDegreesToMatrix(euler), settings.crop_transform.getTranslation());
            }
            ImGui::TreePop();
        }

        // Size (bounds)
        if (ImGui::TreeNodeEx("Size", ImGuiTreeNodeFlags_DefaultOpen)) {
            glm::vec3 size = settings.crop_max - settings.crop_min;
            const glm::vec3 center = (settings.crop_min + settings.crop_max) * 0.5f;

            ImGui::Text("X:"); ImGui::SameLine(); ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##SizeX", &size.x, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive(); any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Y:"); ImGui::SameLine(); ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##SizeY", &size.y, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive(); any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Z:"); ImGui::SameLine(); ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##SizeZ", &size.z, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive(); any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            size = glm::max(size, glm::vec3(MIN_SIZE));

            if (changed) {
                settings.crop_min = center - size * 0.5f;
                settings.crop_max = center + size * 0.5f;
            }
            ImGui::TreePop();
        }

        // Appearance
        if (ImGui::TreeNode("Appearance")) {
            float color[3] = {settings.crop_color.x, settings.crop_color.y, settings.crop_color.z};
            if (ImGui::ColorEdit3("Color", color)) {
                settings.crop_color = glm::vec3(color[0], color[1], color[2]);
                changed = true;
            }
            changed |= ImGui::SliderFloat("Line Width", &settings.crop_line_width, 0.5f, 10.0f);
            ImGui::TreePop();
        }

        // Undo tracking
        if (any_active && !s_editing_active) {
            s_editing_active = true;
            s_state_before_edit = captureState(rm);
        }

        if (changed) {
            rm->updateSettings(settings);
        }

        if (any_deactivated && s_editing_active) {
            s_editing_active = false;
            commitUndoIfChanged(ctx.viewer, rm);
        }

        if (!any_active && s_editing_active) {
            s_editing_active = false;
            commitUndoIfChanged(ctx.viewer, rm);
        }
    }

    const CropBoxState& getCropBoxState() {
        return CropBoxState::getInstance();
    }

} // namespace lfs::vis::gui::panels
