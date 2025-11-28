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

        [[nodiscard]] float getBrushRadius() const { return brush_radius_; }
        void setBrushRadius(float radius) { brush_radius_ = std::clamp(radius, 1.0f, 500.0f); }

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

        void beginStroke(double x, double y, SelectionAction action, bool clear_existing, const ToolContext& ctx);
        void endStroke();
        void updateSelectionAtPoint(double x, double y, const ToolContext& ctx);
        void updateBrushPreview(double x, double y, const ToolContext& ctx);
        void selectInRectangle(const ToolContext& ctx);
        void selectInLasso(const ToolContext& ctx);
        static bool pointInPolygon(float px, float py, const std::vector<glm::vec2>& polygon);
    };

} // namespace lfs::vis::tools
