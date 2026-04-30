/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::core {
    class Tensor;
}

namespace lfs::vis {
    class VulkanContext;

    struct VulkanViewportOverlayVertex {
        glm::vec2 position{0.0f};
        glm::vec4 color{1.0f};
    };

    struct VulkanViewportPivotOverlay {
        glm::vec2 center_ndc{0.0f};
        glm::vec2 size_ndc{0.0f};
        glm::vec3 color{0.26f, 0.59f, 0.98f};
        float opacity = 1.0f;
    };

    struct VulkanViewportPassParams {
        glm::vec2 viewport_pos{0.0f, 0.0f};
        glm::vec2 viewport_size{0.0f, 0.0f};
        glm::vec2 framebuffer_scale{1.0f, 1.0f};
        glm::vec3 background_color{0.0f, 0.0f, 0.0f};

        std::shared_ptr<const lfs::core::Tensor> scene_image;
        glm::ivec2 scene_image_size{0, 0};
        bool scene_image_flip_y = false;

        bool grid_enabled = false;
        glm::mat4 grid_view{1.0f};
        glm::mat4 grid_projection{1.0f};
        glm::mat4 grid_view_projection{1.0f};
        glm::vec3 grid_view_position{0.0f, 0.0f, 0.0f};
        int grid_plane = 2;
        float grid_opacity = 1.0f;
        bool grid_orthographic = false;

        bool vignette_enabled = false;
        float vignette_intensity = 0.0f;
        float vignette_radius = 0.75f;
        float vignette_softness = 0.5f;

        std::vector<VulkanViewportOverlayVertex> overlay_triangles;
        std::vector<VulkanViewportPivotOverlay> pivot_overlays;
    };

    class LFS_VIS_API VulkanViewportPass {
    public:
        VulkanViewportPass();
        ~VulkanViewportPass();

        VulkanViewportPass(const VulkanViewportPass&) = delete;
        VulkanViewportPass& operator=(const VulkanViewportPass&) = delete;

        [[nodiscard]] bool init(VulkanContext& context);
        void prepare(VulkanContext& context, const VulkanViewportPassParams& params);
        void record(VkCommandBuffer command_buffer,
                    VkExtent2D framebuffer_extent,
                    const VulkanViewportPassParams& params);
        void shutdown();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
