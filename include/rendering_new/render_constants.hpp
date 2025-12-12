/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace lfs::rendering {

    // Camera clipping planes
    constexpr float DEFAULT_NEAR_PLANE = 0.1f;
    constexpr float DEFAULT_FAR_PLANE = 100000.0f;

    // Camera defaults
    constexpr float DEFAULT_FOV = 60.0f;
    constexpr float DEFAULT_ORTHO_SCALE = 100.0f;

    // Coordinate system transform: converts from internal camera space (+Y up, +Z forward)
    // to OpenGL clip space (-Y up, -Z forward). Used by all renderers for consistency.
    inline const glm::mat3 FLIP_YZ{1, 0, 0, 0, -1, 0, 0, 0, -1};

    // Compute view rotation from camera-to-world rotation matrix
    inline glm::mat3 computeViewRotation(const glm::mat3& camera_rotation) {
        return FLIP_YZ * glm::transpose(camera_rotation);
    }

    // Create projection matrix - single source of truth for all renderers
    inline glm::mat4 createProjectionMatrix(const glm::ivec2& viewport_size, const float fov_degrees,
                                            const bool orthographic, const float ortho_scale,
                                            const float near_plane = DEFAULT_NEAR_PLANE,
                                            const float far_plane = DEFAULT_FAR_PLANE) {
        const float aspect = static_cast<float>(viewport_size.x) / viewport_size.y;
        if (orthographic) {
            const float half_width = viewport_size.x / (2.0f * ortho_scale);
            const float half_height = viewport_size.y / (2.0f * ortho_scale);
            return glm::ortho(-half_width, half_width, -half_height, half_height, near_plane, far_plane);
        }
        return glm::perspective(glm::radians(fov_degrees), aspect, near_plane, far_plane);
    }

} // namespace lfs::rendering
