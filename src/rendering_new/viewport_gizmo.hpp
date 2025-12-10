/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <optional>

namespace lfs::rendering {

    class TextRenderer; // Forward declaration

    // Axis identifiers for click handling
    enum class GizmoAxis { X = 0, Y = 1, Z = 2 };

    class ViewportGizmo {
    public:
        ViewportGizmo();
        ~ViewportGizmo();

        // Initialize OpenGL resources
        Result<void> initialize();

        // Render the gizmo
        Result<void> render(const glm::mat3& camera_rotation,
                            const glm::vec2& viewport_pos,
                            const glm::vec2& viewport_size);

        // Hit-test: returns axis if click position hits a sphere
        [[nodiscard]] std::optional<GizmoAxis> hitTest(const glm::vec2& click_pos,
                                                        const glm::vec2& viewport_pos,
                                                        const glm::vec2& viewport_size) const;

        // Get camera rotation to look along axis (towards negative axis direction)
        [[nodiscard]] static glm::mat3 getAxisViewRotation(GizmoAxis axis, bool negative = false);

        void shutdown();

        void setSize(int size) { size_ = size; }
        void setMargins(int x, int y) { margin_x_ = x; margin_y_ = y; }
        [[nodiscard]] int getSize() const { return size_; }
        [[nodiscard]] int getMarginX() const { return margin_x_; }
        [[nodiscard]] int getMarginY() const { return margin_y_; }

        // Hover state for highlighting
        void setHoveredAxis(std::optional<GizmoAxis> axis) { hovered_axis_ = axis; }
        [[nodiscard]] std::optional<GizmoAxis> getHoveredAxis() const { return hovered_axis_; }

    private:
        Result<void> generateGeometry();
        Result<void> createShaders();

        // OpenGL resources using RAII
        VAO vao_;
        VBO vbo_;
        ManagedShader shader_;

        // Text rendering
        std::unique_ptr<TextRenderer> text_renderer_;

        // Geometry info
        int cylinder_vertex_count_ = 0;
        int sphere_vertex_start_ = 0;
        int sphere_vertex_count_ = 0;
        int ring_vertex_start_ = 0;
        int ring_vertex_count_ = 0;

        int size_ = 95;
        int margin_x_ = 10;
        int margin_y_ = 35;
        bool initialized_ = false;
        std::optional<GizmoAxis> hovered_axis_;

        // Cached sphere positions for hit-testing (updated each render)
        struct SphereHitInfo {
            glm::vec2 screen_pos{0.0f};
            float radius = 0.0f;
            bool visible = false;
        };
        mutable SphereHitInfo sphere_hits_[3];

        // Axis colors: X=Red, Y=Green, Z=Blue
        static constexpr glm::vec3 AXIS_COLORS[3] = {
            {0.89f, 0.15f, 0.21f}, // X - Red
            {0.54f, 0.86f, 0.20f}, // Y - Green
            {0.17f, 0.48f, 0.87f}  // Z - Blue
        };
    };

} // namespace lfs::rendering
