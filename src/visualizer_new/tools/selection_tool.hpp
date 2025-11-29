/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "tool_base.hpp"
#include "core_new/tensor.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace lfs::vis::tools {

    class SelectionTool : public ToolBase {
    public:
        enum class SelectionAction { Add, Remove };

        SelectionTool();
        ~SelectionTool() override = default;

        [[nodiscard]] std::string_view getName() const override { return "Selection Tool"; }
        [[nodiscard]] std::string_view getDescription() const override { return "Paint to select Gaussians"; }

        bool initialize(const ToolContext& ctx) override;
        void shutdown() override;
        void update(const ToolContext& ctx) override;
        void renderUI(const lfs::vis::gui::UIContext& ui_ctx, bool* p_open) override;

        [[nodiscard]] bool isActive() const { return is_painting_; }
        bool handleMouseButton(int button, int action, int mods, double x, double y, const ToolContext& ctx);
        bool handleMouseMove(double x, double y, const ToolContext& ctx);
        bool handleScroll(double x_offset, double y_offset, int mods, const ToolContext& ctx);
        bool handleKeyPress(int key, int mods, const ToolContext& ctx);

        [[nodiscard]] float getBrushRadius() const { return brush_radius_; }
        void setBrushRadius(float radius) { brush_radius_ = std::clamp(radius, 1.0f, 500.0f); }

        [[nodiscard]] bool hasActivePolygon() const { return !polygon_points_.empty(); }
        void clearPolygon();
        void onSelectionModeChanged();

        // Depth filter
        [[nodiscard]] bool isDepthFilterEnabled() const { return depth_filter_enabled_; }
        void resetDepthFilter();

    protected:
        void onEnabledChanged(bool enabled) override;

    private:
        SelectionAction current_action_ = SelectionAction::Add;
        float brush_radius_ = 20.0f;
        bool is_painting_ = false;
        glm::vec2 last_mouse_pos_{0.0f};
        const ToolContext* tool_context_ = nullptr;
        lfs::core::Tensor cumulative_selection_;
        std::shared_ptr<lfs::core::Tensor> selection_before_stroke_;

        // Rectangle selection state
        bool is_rect_dragging_ = false;
        glm::vec2 rect_start_{0.0f};
        glm::vec2 rect_end_{0.0f};

        // Lasso selection state
        bool is_lasso_dragging_ = false;
        std::vector<glm::vec2> lasso_points_;

        // Preview selection
        lfs::core::Tensor preview_selection_;

        // Polygon selection state
        std::vector<glm::vec2> polygon_points_;
        bool polygon_closed_ = false;
        int polygon_dragged_vertex_ = -1;
        static constexpr float POLYGON_VERTEX_RADIUS = 6.0f;
        static constexpr float POLYGON_CLOSE_THRESHOLD = 12.0f;

        void beginStroke(double x, double y, SelectionAction action, bool clear_existing, const ToolContext& ctx);
        void endStroke();
        void updateSelectionAtPoint(double x, double y, const ToolContext& ctx);
        void updateBrushPreview(double x, double y, const ToolContext& ctx);
        void selectInRectangle(const ToolContext& ctx);
        void selectInLasso(const ToolContext& ctx);
        void selectInPolygon(const ToolContext& ctx);
        void resetPolygon();
        void updateRectanglePreview(const ToolContext& ctx);
        void updateLassoPreview(const ToolContext& ctx);
        void clearPreview(const ToolContext& ctx);
        void prepareSelectionState(const ToolContext& ctx, bool add_to_existing);
        void updatePolygonPreview(const ToolContext& ctx);
        int findPolygonVertexAt(float x, float y) const;
        int findPolygonEdgeAt(float x, float y, float& t_out) const;

        // Depth filter state (camera-aligned crop box)
        bool depth_filter_enabled_ = false;
        float depth_far_ = 100.0f;
        float frustum_half_width_ = 50.0f;
        glm::vec3 last_cam_pos_{0.0f};
        glm::vec3 last_cam_fwd_{0.0f, 0.0f, 1.0f};

        static constexpr float DEPTH_MIN = 0.01f;
        static constexpr float DEPTH_MAX = 1000.0f;
        static constexpr float WIDTH_MIN = 0.1f;
        static constexpr float WIDTH_MAX = 10000.0f;
        static constexpr float ADJUST_FACTOR = 1.15f;

        void drawDepthFrustum(const ToolContext& ctx) const;
        void updateSelectionCropBox(const ToolContext& ctx, bool force = false);
        void disableDepthFilter(const ToolContext& ctx);
    };

} // namespace lfs::vis::tools
