/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.h"
#include "core_new/camera.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"
#include "geometry_new/bounding_box.hpp"
#include "point_cloud_renderer.hpp"
#include "rendering_new/rendering.hpp"  // For SelectionMode
#include "screen_renderer.hpp"
#include <glm/glm.hpp>
#include <optional>

#ifdef CUDA_GL_INTEROP_ENABLED
#include "cuda_gl_interop.hpp"
#include <optional>
#endif

namespace lfs::rendering {

    // Import Tensor from lfs::core
    using lfs::core::Tensor;

    class RenderingPipeline {
    public:
        struct RenderRequest {
            glm::mat3 view_rotation;
            glm::vec3 view_translation;
            glm::ivec2 viewport_size;
            float fov = 60.0f;
            float scaling_modifier = 1.0f;
            bool antialiasing = false;
            int sh_degree = 3;
            RenderMode render_mode = RenderMode::RGB;
            const lfs::geometry::BoundingBox* crop_box = nullptr;
            glm::vec3 background_color = glm::vec3(0.0f, 0.0f, 0.0f);
            bool point_cloud_mode = false;
            float voxel_size = 0.01f;
            bool gut = false;
            bool show_rings = false;
            float ring_width = 0.01f;
            bool show_center_markers = false;
            // Per-node transforms: array of 4x4 matrices and per-Gaussian indices
            std::vector<glm::mat4> model_transforms;          // Array of transforms, one per node
            std::shared_ptr<lfs::core::Tensor> transform_indices;  // Per-Gaussian index [N], nullable
            // Selection mask for highlighting selected Gaussians
            std::shared_ptr<lfs::core::Tensor> selection_mask;
            bool output_screen_positions = false;
            bool brush_active = false;
            float brush_x = 0.0f;
            float brush_y = 0.0f;
            float brush_radius = 0.0f;
            bool brush_add_mode = true;
            lfs::core::Tensor* brush_selection_tensor = nullptr;
            bool brush_saturation_mode = false;
            float brush_saturation_amount = 0.0f;
            bool selection_mode_rings = false;  // Ring mode hover detection
            // Crop box filtering
            const Tensor* crop_box_transform = nullptr;
            const Tensor* crop_box_min = nullptr;
            const Tensor* crop_box_max = nullptr;
            bool crop_inverse = false;
            const Tensor* deleted_mask = nullptr;  // Soft deletion mask [N], true = skip
            // Ring mode hover output
            unsigned long long* hovered_depth_id = nullptr;
            int highlight_gaussian_id = -1;
            float far_plane = 1e10f;
        };

        struct RenderResult {
            Tensor image;
            Tensor depth;
            Tensor screen_positions;  // Optional: screen positions [N, 2] for brush tool
            bool valid = false;
        };

        RenderingPipeline();
        ~RenderingPipeline();

        // Main render function - now returns Result
        Result<RenderResult> render(const lfs::core::SplatData& model, const RenderRequest& request);

        // Static upload function - now returns Result
        static Result<void> uploadToScreen(const RenderResult& result,
                                           ScreenQuadRenderer& renderer,
                                           const glm::ivec2& viewport_size);

    private:
        Result<lfs::core::Camera> createCamera(const RenderRequest& request);
        glm::vec2 computeFov(float fov_degrees, int width, int height);
        Result<RenderResult> renderPointCloud(const lfs::core::SplatData& model, const RenderRequest& request);

        // Ensure persistent FBO is sized correctly (avoids recreation every frame)
        void ensureFBOSize(int width, int height);
        void cleanupFBO();

        // Ensure PBOs are sized correctly (avoids recreation every frame)
        void ensurePBOSize(int width, int height);
        void cleanupPBO();

        Tensor background_;
        std::unique_ptr<PointCloudRenderer> point_cloud_renderer_;

        // Persistent framebuffer objects (reused across frames)
        // Avoids expensive glGenFramebuffers/glDeleteFramebuffers every render
        GLuint persistent_fbo_ = 0;
        GLuint persistent_color_texture_ = 0;
        GLuint persistent_depth_texture_ = 0;
        int persistent_fbo_width_ = 0;
        int persistent_fbo_height_ = 0;

        // Pixel Buffer Objects for async GPU→CPU readback
        // Uses double-buffering to overlap memory transfer with rendering
        GLuint pbo_[2] = {0, 0};
        int pbo_index_ = 0;
        int pbo_width_ = 0;
        int pbo_height_ = 0;

#ifdef CUDA_GL_INTEROP_ENABLED
        // CUDA-GL interop for direct FBO→CUDA texture readback (eliminates CPU round-trip)
        std::optional<CudaGLInteropTexture> fbo_interop_texture_;
        bool use_fbo_interop_ = true;
        int fbo_interop_last_width_ = 0;   // Track FBO size when interop was initialized
        int fbo_interop_last_height_ = 0;  // to detect when we need to reinitialize
#endif
    };

} // namespace lfs::rendering