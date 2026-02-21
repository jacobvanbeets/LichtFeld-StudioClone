/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#ifdef LFS_VR_ENABLED

#include "vr/vr_manager.hpp"
#include <glm/glm.hpp>

namespace lfs::vis::vr {

    /// Translates VR controller input into scene navigation transforms.
    /// Maintains a "VR world transform" that positions/orients the Gaussian scene
    /// relative to the VR play space.
    class VRInput {
    public:
        VRInput() = default;

        /// Process controller state and update the world transform.
        /// Call once per frame after VRManager::process_events().
        void update(const VRManager& vr, float delta_time);

        /// The accumulated world transform (scene position/rotation/scale in VR space).
        /// Apply this when computing the view matrix for each eye.
        [[nodiscard]] glm::mat4 get_world_transform() const { return world_transform_; }

        /// Reset navigation to identity (re-center the scene).
        void reset() { world_transform_ = glm::mat4(1.0f); }

        // Navigation settings
        float move_speed = 2.0f;   // meters/second for thumbstick locomotion
        float rotate_speed = 1.5f; // radians/second for thumbstick rotation
        float scale_speed = 2.0f;  // scale factor per second for grip scaling

    private:
        void handle_thumbstick_locomotion(const VRManager& vr, float dt);
        void handle_grip_drag(const VRManager& vr);

        glm::mat4 world_transform_{1.0f};

        // Grip-drag state
        bool grip_active_ = false;
        glm::mat4 grip_start_controller_{1.0f};
        glm::mat4 grip_start_world_{1.0f};
    };

} // namespace lfs::vis::vr

#endif // LFS_VR_ENABLED
