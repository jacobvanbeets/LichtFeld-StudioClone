/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "geometry/euclidean_transform.hpp"
#include "rendering/rendering_manager.hpp"
#include <glm/glm.hpp>

namespace lfs::vis::gui::utils {

    constexpr float SCALE_EPSILON = 1e-5f;

    /**
     * Apply a delta transform to the crop box settings.
     * Handles both scale (applied to bounds) and rotation/translation (via EuclideanTransform).
     */
    inline void applyCropBoxDelta(RenderSettings& settings, const glm::mat4& delta_matrix) {
        const glm::vec3 delta_scale(
            glm::length(glm::vec3(delta_matrix[0])),
            glm::length(glm::vec3(delta_matrix[1])),
            glm::length(glm::vec3(delta_matrix[2])));

        const bool has_scale = glm::abs(delta_scale.x - 1.0f) > SCALE_EPSILON ||
                               glm::abs(delta_scale.y - 1.0f) > SCALE_EPSILON ||
                               glm::abs(delta_scale.z - 1.0f) > SCALE_EPSILON;

        if (has_scale) {
            // Scale crop bounds around their center
            const glm::vec3 center = (settings.crop_min + settings.crop_max) * 0.5f;
            const glm::vec3 half_size = (settings.crop_max - settings.crop_min) * 0.5f;
            const glm::vec3 new_half_size = half_size * delta_scale;
            settings.crop_min = center - new_half_size;
            settings.crop_max = center + new_half_size;

            // Scale the translation component of crop_transform
            const glm::vec3 old_trans = settings.crop_transform.getTranslation();
            settings.crop_transform = lfs::geometry::EuclideanTransform(
                settings.crop_transform.getRotationMat(),
                old_trans * delta_scale);
        } else {
            // Apply rotation/translation via EuclideanTransform
            const lfs::geometry::EuclideanTransform delta(delta_matrix);
            settings.crop_transform = delta * settings.crop_transform;
        }
    }

} // namespace lfs::vis::gui::utils
