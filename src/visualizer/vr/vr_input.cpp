/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef LFS_VR_ENABLED

#include "vr/vr_input.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

namespace lfs::vis::vr {

    void VRInput::update(const VRManager& vr, float delta_time) {
        handle_thumbstick_locomotion(vr, delta_time);
        handle_grip_drag(vr);
    }

    void VRInput::handle_thumbstick_locomotion(const VRManager& vr, float dt) {
        // Left thumbstick: translate (forward/back, strafe left/right)
        // Right thumbstick: rotate (yaw) and move up/down
        const auto& left = vr.get_left_controller();
        const auto& right = vr.get_right_controller();

        // Dead zone
        constexpr float DEAD_ZONE = 0.15f;

        // Get HMD forward/right directions (projected onto XZ plane for comfort)
        glm::mat4 hmd = vr.get_hmd_pose();
        glm::vec3 hmd_forward = -glm::normalize(glm::vec3(hmd[2][0], 0.0f, hmd[2][2]));
        glm::vec3 hmd_right = glm::normalize(glm::vec3(hmd[0][0], 0.0f, hmd[0][2]));
        glm::vec3 up(0.0f, 1.0f, 0.0f);

        glm::vec3 translation(0.0f);

        // Left thumbstick: movement
        if (left.valid) {
            float lx = left.thumbstick.x;
            float ly = left.thumbstick.y;
            if (std::abs(lx) > DEAD_ZONE)
                translation += hmd_right * lx * move_speed * dt;
            if (std::abs(ly) > DEAD_ZONE)
                translation += hmd_forward * ly * move_speed * dt;
        }

        // Right thumbstick: snap-turn (horizontal) and vertical movement
        if (right.valid) {
            float rx = right.thumbstick.x;
            float ry = right.thumbstick.y;

            // Smooth yaw rotation
            if (std::abs(rx) > DEAD_ZONE) {
                float angle = -rx * rotate_speed * dt;
                glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), angle, up);
                world_transform_ = rotation * world_transform_;
            }

            // Vertical movement
            if (std::abs(ry) > DEAD_ZONE) {
                translation += up * ry * move_speed * dt;
            }
        }

        if (glm::length(translation) > 0.0001f) {
            world_transform_ = glm::translate(glm::mat4(1.0f), translation) * world_transform_;
        }
    }

    void VRInput::handle_grip_drag(const VRManager& vr) {
        // Right grip: grab-the-world navigation
        // When grip is held, controller movement directly moves the scene
        const auto& right = vr.get_right_controller();

        if (!right.valid) {
            grip_active_ = false;
            return;
        }

        if (right.grip_pressed) {
            if (!grip_active_) {
                // Grip just pressed - record starting state
                grip_active_ = true;
                grip_start_controller_ = right.transform;
                grip_start_world_ = world_transform_;
            }

            // Compute delta from grip start to current controller position
            glm::mat4 controller_delta = right.transform * glm::inverse(grip_start_controller_);

            // Apply delta to the world transform
            world_transform_ = controller_delta * grip_start_world_;
        } else {
            grip_active_ = false;
        }
    }

} // namespace lfs::vis::vr

#endif // LFS_VR_ENABLED
