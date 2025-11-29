/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/selection_tool.hpp"
#include "command/command_history.hpp"
#include "command/commands/selection_command.hpp"
#include "internal/viewport.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"
#include "rendering_new/rasterizer/rasterization/include/forward.h"
#include "rendering_new/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>
#include <cmath>

namespace lfs::vis::tools {

    SelectionTool::SelectionTool() = default;

    bool SelectionTool::initialize(const ToolContext& ctx) {
        tool_context_ = &ctx;
        return true;
    }

    void SelectionTool::shutdown() {
        tool_context_ = nullptr;
        is_painting_ = false;
    }

    void SelectionTool::update([[maybe_unused]] const ToolContext& ctx) {
        if (isEnabled()) {
            double mx, my;
            glfwGetCursorPos(ctx.getWindow(), &mx, &my);
            last_mouse_pos_ = glm::vec2(static_cast<float>(mx), static_cast<float>(my));

            // Update depth filter crop box to follow camera when enabled
            if (depth_filter_enabled_) {
                updateSelectionCropBox(ctx);
            }
        }
    }

    void SelectionTool::renderUI([[maybe_unused]] const lfs::vis::gui::UIContext& ui_ctx,
                                  [[maybe_unused]] bool* p_open) {
        if (!isEnabled() || ImGui::GetIO().WantCaptureMouse) return;

        auto sel_mode = lfs::rendering::SelectionMode::Centers;
        if (tool_context_) {
            const auto* const rm = tool_context_->getRenderingManager();
            if (rm) sel_mode = rm->getSelectionMode();
        }

        ImDrawList* const draw_list = ImGui::GetForegroundDrawList();
        const ImVec2 mouse_pos = ImGui::GetMousePos();
        constexpr ImU32 brush_color = IM_COL32(100, 180, 255, 220);

        // Draw rectangle if dragging
        if (is_rect_dragging_) {
            const ImVec2 p1(rect_start_.x, rect_start_.y);
            const ImVec2 p2(rect_end_.x, rect_end_.y);
            draw_list->AddRect(p1, p2, brush_color, 0.0f, 0, 2.0f);
            draw_list->AddRectFilled(p1, p2, IM_COL32(100, 180, 255, 40));
        }

        // Draw lasso if dragging
        if (is_lasso_dragging_ && lasso_points_.size() >= 2) {
            for (size_t i = 1; i < lasso_points_.size(); ++i) {
                draw_list->AddLine(ImVec2(lasso_points_[i - 1].x, lasso_points_[i - 1].y),
                                   ImVec2(lasso_points_[i].x, lasso_points_[i].y), brush_color, 2.0f);
            }
            // Draw closing line to start
            draw_list->AddLine(ImVec2(lasso_points_.back().x, lasso_points_.back().y),
                               ImVec2(lasso_points_.front().x, lasso_points_.front().y),
                               IM_COL32(100, 180, 255, 100), 1.0f);
        }

        // Draw polygon
        if (!polygon_points_.empty()) {
            constexpr ImU32 VERTEX_COLOR = IM_COL32(255, 200, 100, 255);
            constexpr ImU32 VERTEX_HOVER_COLOR = IM_COL32(255, 255, 150, 255);
            constexpr ImU32 CLOSE_HINT_COLOR = IM_COL32(100, 255, 100, 200);
            constexpr ImU32 FILL_COLOR = IM_COL32(100, 180, 255, 40);
            constexpr ImU32 LINE_TO_MOUSE_COLOR = IM_COL32(100, 180, 255, 100);

            for (size_t i = 1; i < polygon_points_.size(); ++i) {
                draw_list->AddLine(ImVec2(polygon_points_[i - 1].x, polygon_points_[i - 1].y),
                                   ImVec2(polygon_points_[i].x, polygon_points_[i].y), brush_color, 2.0f);
            }

            if (polygon_closed_) {
                draw_list->AddLine(ImVec2(polygon_points_.back().x, polygon_points_.back().y),
                                   ImVec2(polygon_points_.front().x, polygon_points_.front().y), brush_color, 2.0f);
                if (polygon_points_.size() >= 3) {
                    std::vector<ImVec2> im_points;
                    im_points.reserve(polygon_points_.size());
                    for (const auto& pt : polygon_points_)
                        im_points.emplace_back(pt.x, pt.y);
                    draw_list->AddConvexPolyFilled(im_points.data(), static_cast<int>(im_points.size()), FILL_COLOR);
                }
            } else {
                draw_list->AddLine(ImVec2(polygon_points_.back().x, polygon_points_.back().y),
                                   mouse_pos, LINE_TO_MOUSE_COLOR, 1.0f);
                if (polygon_points_.size() >= 3) {
                    const glm::vec2 d = glm::vec2(mouse_pos.x, mouse_pos.y) - polygon_points_.front();
                    if (glm::dot(d, d) < POLYGON_CLOSE_THRESHOLD * POLYGON_CLOSE_THRESHOLD) {
                        draw_list->AddCircle(ImVec2(polygon_points_.front().x, polygon_points_.front().y),
                                             POLYGON_VERTEX_RADIUS + 3.0f, CLOSE_HINT_COLOR, 16, 2.0f);
                    }
                }
            }

            const int hovered_idx = findPolygonVertexAt(mouse_pos.x, mouse_pos.y);
            for (size_t i = 0; i < polygon_points_.size(); ++i) {
                const auto& pt = polygon_points_[i];
                const ImU32 color = (static_cast<int>(i) == hovered_idx) ? VERTEX_HOVER_COLOR : VERTEX_COLOR;
                draw_list->AddCircleFilled(ImVec2(pt.x, pt.y), POLYGON_VERTEX_RADIUS, color);
                draw_list->AddCircle(ImVec2(pt.x, pt.y), POLYGON_VERTEX_RADIUS, brush_color, 16, 1.5f);
            }
        }

        const char* info_text = nullptr;
        float text_offset = 15.0f;

        if (sel_mode == lfs::rendering::SelectionMode::Rings) {
            constexpr float cross_size = 8.0f;
            draw_list->AddLine(ImVec2(mouse_pos.x - cross_size, mouse_pos.y),
                               ImVec2(mouse_pos.x + cross_size, mouse_pos.y), brush_color, 2.0f);
            draw_list->AddLine(ImVec2(mouse_pos.x, mouse_pos.y - cross_size),
                               ImVec2(mouse_pos.x, mouse_pos.y + cross_size), brush_color, 2.0f);
            info_text = is_painting_ ? (current_action_ == SelectionAction::Add ? "RING +" : "RING -") : "RING";
        } else if (sel_mode == lfs::rendering::SelectionMode::Rectangle) {
            constexpr float cross_size = 8.0f;
            draw_list->AddLine(ImVec2(mouse_pos.x - cross_size, mouse_pos.y),
                               ImVec2(mouse_pos.x + cross_size, mouse_pos.y), brush_color, 2.0f);
            draw_list->AddLine(ImVec2(mouse_pos.x, mouse_pos.y - cross_size),
                               ImVec2(mouse_pos.x, mouse_pos.y + cross_size), brush_color, 2.0f);
            info_text = is_rect_dragging_ ? (current_action_ == SelectionAction::Add ? "RECT +" : "RECT -") : "RECT";
        } else if (sel_mode == lfs::rendering::SelectionMode::Polygon) {
            constexpr float cross_size = 8.0f;
            draw_list->AddLine(ImVec2(mouse_pos.x - cross_size, mouse_pos.y),
                               ImVec2(mouse_pos.x + cross_size, mouse_pos.y), brush_color, 2.0f);
            draw_list->AddLine(ImVec2(mouse_pos.x, mouse_pos.y - cross_size),
                               ImVec2(mouse_pos.x, mouse_pos.y + cross_size), brush_color, 2.0f);
            if (polygon_closed_) {
                info_text = "POLY [Enter]";
            } else {
                info_text = "POLY";
            }
        } else if (sel_mode == lfs::rendering::SelectionMode::Lasso) {
            constexpr float cross_size = 8.0f;
            draw_list->AddLine(ImVec2(mouse_pos.x - cross_size, mouse_pos.y),
                               ImVec2(mouse_pos.x + cross_size, mouse_pos.y), brush_color, 2.0f);
            draw_list->AddLine(ImVec2(mouse_pos.x, mouse_pos.y - cross_size),
                               ImVec2(mouse_pos.x, mouse_pos.y + cross_size), brush_color, 2.0f);
            info_text = is_lasso_dragging_ ? (current_action_ == SelectionAction::Add ? "LASSO +" : "LASSO -") : "LASSO";
        } else {
            draw_list->AddCircle(mouse_pos, brush_radius_, brush_color, 32, 2.0f);
            draw_list->AddCircleFilled(mouse_pos, 3.0f, brush_color);
            info_text = is_painting_ ? (current_action_ == SelectionAction::Add ? "SEL +" : "SEL -") : "SEL";
            text_offset = brush_radius_ + 10.0f;
        }

        constexpr float font_size = 22.0f;
        const ImVec2 text_pos(mouse_pos.x + text_offset, mouse_pos.y - font_size / 2);
        constexpr ImU32 shadow_color = IM_COL32(0, 0, 0, 180);
        draw_list->AddText(ImGui::GetFont(), font_size, ImVec2(text_pos.x + 1, text_pos.y + 1), shadow_color, info_text);
        draw_list->AddText(ImGui::GetFont(), font_size, text_pos, IM_COL32(255, 255, 255, 255), info_text);

        // Draw depth filter frustum if enabled
        if (depth_filter_enabled_ && tool_context_) {
            drawDepthFrustum(*tool_context_);
        }
    }

    bool SelectionTool::handleMouseButton(const int button, const int action, const int mods,
                                           const double x, const double y, const ToolContext& ctx) {
        if (!isEnabled()) return false;

        const auto* const rm = ctx.getRenderingManager();
        const auto sel_mode = rm ? rm->getSelectionMode() : lfs::rendering::SelectionMode::Centers;

        // Right-click removes last vertex in polygon mode (only when not closed)
        if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
            if (sel_mode == lfs::rendering::SelectionMode::Polygon && !polygon_closed_ && !polygon_points_.empty()) {
                polygon_points_.pop_back();
                ctx.requestRender();
                return true;
            }
            return false;
        }

        if (button != GLFW_MOUSE_BUTTON_LEFT) return false;

        const bool is_rect_mode = (sel_mode == lfs::rendering::SelectionMode::Rectangle);
        const bool is_lasso_mode = (sel_mode == lfs::rendering::SelectionMode::Lasso);
        const bool is_polygon_mode = (sel_mode == lfs::rendering::SelectionMode::Polygon);

        if (action == GLFW_PRESS) {
            const bool ctrl = (mods & GLFW_MOD_CONTROL) != 0;
            const bool shift = (mods & GLFW_MOD_SHIFT) != 0;

            if (is_polygon_mode) {
                const float px = static_cast<float>(x);
                const float py = static_cast<float>(y);
                current_action_ = SelectionAction::Add;

                if (polygon_closed_) {
                    // Drag vertex
                    if (const int vi = findPolygonVertexAt(px, py); vi >= 0) {
                        polygon_dragged_vertex_ = vi;
                        ctx.requestRender();
                        return true;
                    }
                    // Insert vertex on edge
                    float t = 0.0f;
                    if (const int ei = findPolygonEdgeAt(px, py, t); ei >= 0) {
                        const auto& a = polygon_points_[ei];
                        const auto& b = polygon_points_[(ei + 1) % polygon_points_.size()];
                        polygon_points_.insert(polygon_points_.begin() + ei + 1, a + t * (b - a));
                        polygon_dragged_vertex_ = ei + 1;
                        updatePolygonPreview(ctx);
                        ctx.requestRender();
                        return true;
                    }
                    resetPolygon();
                }

                // Close polygon
                if (polygon_points_.size() >= 3 &&
                    glm::distance(glm::vec2(px, py), polygon_points_.front()) < POLYGON_CLOSE_THRESHOLD) {
                    polygon_closed_ = true;
                    prepareSelectionState(ctx, true);
                    updatePolygonPreview(ctx);
                    ctx.requestRender();
                    return true;
                }

                // Drag existing vertex
                const int vertex_idx = findPolygonVertexAt(px, py);
                if (vertex_idx >= 0) {
                    polygon_dragged_vertex_ = vertex_idx;
                    ctx.requestRender();
                    return true;
                }

                polygon_points_.emplace_back(px, py);
                ctx.requestRender();
                return true;
            }

            if (is_rect_mode || is_lasso_mode) {
                // Rectangle/Lasso mode: start dragging
                if (is_rect_mode) {
                    is_rect_dragging_ = true;
                    rect_start_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));
                    rect_end_ = rect_start_;
                } else {
                    is_lasso_dragging_ = true;
                    lasso_points_.clear();
                    lasso_points_.emplace_back(static_cast<float>(x), static_cast<float>(y));
                }
                current_action_ = shift ? SelectionAction::Remove : SelectionAction::Add;

                // Prepare selection state
                auto* const sm = ctx.getSceneManager();
                if (sm) {
                    const size_t num_gaussians = sm->getScene().getTotalGaussianCount();
                    if (num_gaussians > 0) {
                        auto existing = sm->getScene().getSelectionMask();
                        if (existing && existing->is_valid()) {
                            selection_before_stroke_ = std::make_shared<lfs::core::Tensor>(existing->clone());
                        } else {
                            selection_before_stroke_.reset();
                        }
                        if (!ctrl && !shift) {
                            sm->getScene().clearSelection();
                            cumulative_selection_ = lfs::core::Tensor::zeros(
                                {num_gaussians}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
                        } else if (existing && existing->is_valid() && existing->size(0) == num_gaussians) {
                            cumulative_selection_ = existing->to(lfs::core::DataType::Bool);
                        } else {
                            cumulative_selection_ = lfs::core::Tensor::zeros(
                                {num_gaussians}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
                        }
                    }
                }
                return true;
            }

            // Brush/Ring mode
            if (ctrl) {
                beginStroke(x, y, SelectionAction::Add, false, ctx);
            } else if (shift) {
                beginStroke(x, y, SelectionAction::Remove, false, ctx);
            } else {
                beginStroke(x, y, SelectionAction::Add, true, ctx);
            }
            updateSelectionAtPoint(x, y, ctx);
            return true;
        }

        if (action == GLFW_RELEASE) {
            if (polygon_dragged_vertex_ >= 0) {
                polygon_dragged_vertex_ = -1;
                ctx.requestRender();
                return true;
            }
            if (is_rect_dragging_) {
                clearPreview(ctx);
                selectInRectangle(ctx);
                is_rect_dragging_ = false;
                return true;
            }
            if (is_lasso_dragging_) {
                clearPreview(ctx);
                selectInLasso(ctx);
                is_lasso_dragging_ = false;
                lasso_points_.clear();
                return true;
            }
            if (is_painting_) {
                endStroke();
                return true;
            }
        }
        return false;
    }

    bool SelectionTool::handleMouseMove(const double x, const double y, const ToolContext& ctx) {
        if (!isEnabled()) return false;

        last_mouse_pos_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));

        // Polygon vertex dragging
        if (polygon_dragged_vertex_ >= 0 && polygon_dragged_vertex_ < static_cast<int>(polygon_points_.size())) {
            polygon_points_[polygon_dragged_vertex_] = glm::vec2(static_cast<float>(x), static_cast<float>(y));
            if (polygon_closed_) {
                updatePolygonPreview(ctx);
            }
            ctx.requestRender();
            return true;
        }

        if (is_rect_dragging_) {
            rect_end_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));
            updateRectanglePreview(ctx);
            ctx.requestRender();
            return true;
        }

        if (is_lasso_dragging_) {
            const glm::vec2 new_point(static_cast<float>(x), static_cast<float>(y));
            if (lasso_points_.empty() || glm::distance(lasso_points_.back(), new_point) > 3.0f) {
                lasso_points_.push_back(new_point);
                updateLassoPreview(ctx);
            }
            ctx.requestRender();
            return true;
        }

        if (is_painting_) {
            updateSelectionAtPoint(x, y, ctx);
            ctx.requestRender();
            return true;
        }

        updateBrushPreview(x, y, ctx);
        ctx.requestRender();
        return false;
    }

    bool SelectionTool::handleScroll([[maybe_unused]] const double x_offset, const double y_offset,
                                      const int mods, const ToolContext& ctx) {
        if (!isEnabled()) return false;

        const bool ctrl = (mods & GLFW_MOD_CONTROL) != 0;
        const bool shift = (mods & GLFW_MOD_SHIFT) != 0;
        const bool alt = glfwGetKey(ctx.getWindow(), GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                         glfwGetKey(ctx.getWindow(), GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;

        const auto* const rm = ctx.getRenderingManager();
        const auto mode = rm ? rm->getSelectionMode() : lfs::rendering::SelectionMode::Centers;

        // Alt+Scroll: depth filter adjustment
        if (alt && depth_filter_enabled_ && mode != lfs::rendering::SelectionMode::Rings) {
            const float scale = (y_offset > 0) ? ADJUST_FACTOR : (1.0f / ADJUST_FACTOR);

            if (ctrl) {
                frustum_half_width_ = std::clamp(frustum_half_width_ * scale, WIDTH_MIN, WIDTH_MAX);
            } else {
                depth_far_ = std::clamp(depth_far_ * scale, DEPTH_MIN, DEPTH_MAX);
            }

            updateSelectionCropBox(ctx, true);  // Force update - params changed
            ctx.requestRender();
            return true;
        }

        // Brush radius adjustment only for brush/center mode
        if (mode == lfs::rendering::SelectionMode::Rings ||
            mode == lfs::rendering::SelectionMode::Rectangle ||
            mode == lfs::rendering::SelectionMode::Polygon ||
            mode == lfs::rendering::SelectionMode::Lasso) {
            return false;
        }

        if (is_painting_ || ctrl || shift) {
            const float scale = (y_offset > 0) ? 1.1f : 0.9f;
            brush_radius_ = std::clamp(brush_radius_ * scale, 1.0f, 500.0f);
            updateBrushPreview(last_mouse_pos_.x, last_mouse_pos_.y, ctx);
            ctx.requestRender();
            return true;
        }
        return false;
    }

    void SelectionTool::onEnabledChanged(const bool enabled) {
        if (!enabled) {
            is_painting_ = false;
            is_rect_dragging_ = false;
            is_lasso_dragging_ = false;
            lasso_points_.clear();
            resetPolygon();
            preview_selection_ = lfs::core::Tensor();
            cumulative_selection_ = lfs::core::Tensor();
            selection_before_stroke_.reset();
        }

        if (tool_context_) {
            auto* const rm = tool_context_->getRenderingManager();
            if (rm) {
                rm->setOutputScreenPositions(enabled);
                if (!enabled) {
                    rm->clearBrushState();
                    rm->clearPreviewSelection();
                }
                rm->markDirty();
            }
            if (!enabled) {
                if (auto* const sm = tool_context_->getSceneManager()) {
                    sm->getScene().clearSelection();
                }
            }
        }
    }

    void SelectionTool::beginStroke([[maybe_unused]] const double x, [[maybe_unused]] const double y,
                                     const SelectionAction action, const bool clear_existing,
                                     const ToolContext& ctx) {
        is_painting_ = true;
        current_action_ = action;

        auto* const sm = ctx.getSceneManager();
        if (!sm) return;

        const size_t num_gaussians = sm->getScene().getTotalGaussianCount();
        if (num_gaussians == 0) return;

        auto existing = sm->getScene().getSelectionMask();
        if (existing && existing->is_valid()) {
            selection_before_stroke_ = std::make_shared<lfs::core::Tensor>(existing->clone());
        } else {
            selection_before_stroke_.reset();
        }

        if (clear_existing) {
            sm->getScene().clearSelection();
            cumulative_selection_ = lfs::core::Tensor::zeros(
                {num_gaussians}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
        } else {
            if (existing && existing->is_valid() && existing->size(0) == num_gaussians) {
                cumulative_selection_ = existing->to(lfs::core::DataType::Bool);
            } else {
                cumulative_selection_ = lfs::core::Tensor::zeros(
                    {num_gaussians}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
            }
        }
    }

    void SelectionTool::endStroke() {
        is_painting_ = false;

        std::shared_ptr<lfs::core::Tensor> new_selection;

        if (tool_context_) {
            auto* const rm = tool_context_->getRenderingManager();
            if (rm) {
                rm->clearPreviewSelection();
                rm->clearBrushState();
            }

            if (cumulative_selection_.is_valid()) {
                auto* const sm = tool_context_->getSceneManager();
                if (sm) {
                    auto mask = cumulative_selection_.to(lfs::core::DataType::UInt8);
                    new_selection = std::make_shared<lfs::core::Tensor>(mask.clone());
                    sm->getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(mask)));
                }
            }

            if (new_selection && new_selection->is_valid()) {
                auto* const ch = tool_context_->getCommandHistory();
                auto* const sm = tool_context_->getSceneManager();

                if (ch && sm) {
                    auto cmd = std::make_unique<command::SelectionCommand>(
                        sm, selection_before_stroke_, new_selection);
                    ch->execute(std::move(cmd));
                }
            }

            if (auto* const rm2 = tool_context_->getRenderingManager())
                rm2->markDirty();
        }
        selection_before_stroke_.reset();
    }

    void SelectionTool::updateSelectionAtPoint(const double x, const double y, const ToolContext& ctx) {
        auto* const rm = ctx.getRenderingManager();
        if (!rm) return;

        const auto& bounds = ctx.getViewportBounds();
        const auto& viewport = ctx.getViewport();
        const auto& cached = rm->getCachedResult();

        const int render_w = cached.image ? static_cast<int>(cached.image->size(2)) : viewport.windowSize.x;
        const int render_h = cached.image ? static_cast<int>(cached.image->size(1)) : viewport.windowSize.y;

        const float rel_x = static_cast<float>(x) - bounds.x;
        const float rel_y = static_cast<float>(y) - bounds.y;
        const float scale_x = static_cast<float>(render_w) / bounds.width;
        const float scale_y = static_cast<float>(render_h) / bounds.height;

        const float image_x = rel_x * scale_x;
        const float image_y = rel_y * scale_y;
        const bool add_mode = (current_action_ == SelectionAction::Add);

        if (rm->getSelectionMode() == lfs::rendering::SelectionMode::Rings) {
            const int hovered_id = rm->getHoveredGaussianId();
            if (hovered_id >= 0 && cumulative_selection_.is_valid()) {
                lfs::rendering::set_selection_element(cumulative_selection_.ptr<bool>(), hovered_id, add_mode);
            }
            rm->setBrushState(true, image_x, image_y, 0.0f, add_mode, nullptr, false, 0.0f);
            rm->setPreviewSelection(&cumulative_selection_);
        } else {
            const float scaled_radius = brush_radius_ * scale_x;
            rm->setBrushState(true, image_x, image_y, scaled_radius, add_mode, &cumulative_selection_);
            rm->setPreviewSelection(&cumulative_selection_);
        }
    }

    void SelectionTool::updateBrushPreview(const double x, const double y, const ToolContext& ctx) {
        auto* const rm = ctx.getRenderingManager();
        if (!rm) return;

        const auto& bounds = ctx.getViewportBounds();
        const auto& viewport = ctx.getViewport();
        const auto& cached = rm->getCachedResult();

        const int render_w = cached.image ? static_cast<int>(cached.image->size(2)) : viewport.windowSize.x;
        const int render_h = cached.image ? static_cast<int>(cached.image->size(1)) : viewport.windowSize.y;

        const float rel_x = static_cast<float>(x) - bounds.x;
        const float rel_y = static_cast<float>(y) - bounds.y;
        const float scale_x = static_cast<float>(render_w) / bounds.width;
        const float scale_y = static_cast<float>(render_h) / bounds.height;

        const float image_x = rel_x * scale_x;
        const float image_y = rel_y * scale_y;

        const auto sel_mode = rm->getSelectionMode();
        const bool shift_held = glfwGetKey(ctx.getWindow(), GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                                glfwGetKey(ctx.getWindow(), GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        const bool add_mode = !shift_held;

        if (sel_mode == lfs::rendering::SelectionMode::Rings ||
            sel_mode == lfs::rendering::SelectionMode::Rectangle ||
            sel_mode == lfs::rendering::SelectionMode::Polygon ||
            sel_mode == lfs::rendering::SelectionMode::Lasso) {
            rm->setBrushState(true, image_x, image_y, 0.0f, add_mode, nullptr, false, 0.0f);
        } else {
            const float scaled_radius = brush_radius_ * scale_x;
            rm->setBrushState(true, image_x, image_y, scaled_radius, add_mode, nullptr, false, 0.0f);
        }
    }

    void SelectionTool::selectInRectangle(const ToolContext& ctx) {
        auto* const rm = ctx.getRenderingManager();
        auto* const sm = ctx.getSceneManager();
        if (!rm || !sm || !cumulative_selection_.is_valid()) return;

        const auto screen_positions = rm->getScreenPositions();
        if (!screen_positions || !screen_positions->is_valid()) return;

        const auto& bounds = ctx.getViewportBounds();
        const auto& viewport = ctx.getViewport();
        const auto& cached = rm->getCachedResult();

        const int render_w = cached.image ? static_cast<int>(cached.image->size(2)) : viewport.windowSize.x;
        const int render_h = cached.image ? static_cast<int>(cached.image->size(1)) : viewport.windowSize.y;
        const float scale_x = static_cast<float>(render_w) / bounds.width;
        const float scale_y = static_cast<float>(render_h) / bounds.height;

        const float x0 = (std::min(rect_start_.x, rect_end_.x) - bounds.x) * scale_x;
        const float y0 = (std::min(rect_start_.y, rect_end_.y) - bounds.y) * scale_y;
        const float x1 = (std::max(rect_start_.x, rect_end_.x) - bounds.x) * scale_x;
        const float y1 = (std::max(rect_start_.y, rect_end_.y) - bounds.y) * scale_y;

        const bool add_mode = (current_action_ == SelectionAction::Add);
        lfs::rendering::rect_select_mode_tensor(*screen_positions, x0, y0, x1, y1, cumulative_selection_, add_mode);

        auto mask = cumulative_selection_.to(lfs::core::DataType::UInt8);
        auto new_selection = std::make_shared<lfs::core::Tensor>(mask.clone());
        sm->getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(mask)));

        if (auto* const ch = ctx.getCommandHistory()) {
            ch->execute(std::make_unique<command::SelectionCommand>(
                sm, selection_before_stroke_, new_selection));
        }
        selection_before_stroke_.reset();
        rm->clearBrushState();
        rm->markDirty();
    }

    void SelectionTool::selectInLasso(const ToolContext& ctx) {
        auto* const rm = ctx.getRenderingManager();
        auto* const sm = ctx.getSceneManager();
        if (!rm || !sm || !cumulative_selection_.is_valid()) return;
        if (lasso_points_.size() < 3) return;

        const auto screen_positions = rm->getScreenPositions();
        if (!screen_positions || !screen_positions->is_valid()) return;

        const auto& bounds = ctx.getViewportBounds();
        const auto& viewport = ctx.getViewport();
        const auto& cached = rm->getCachedResult();

        const int render_w = cached.image ? static_cast<int>(cached.image->size(2)) : viewport.windowSize.x;
        const int render_h = cached.image ? static_cast<int>(cached.image->size(1)) : viewport.windowSize.y;

        const float scale_x = static_cast<float>(render_w) / bounds.width;
        const float scale_y = static_cast<float>(render_h) / bounds.height;
        const size_t num_verts = lasso_points_.size();

        auto poly_cpu = lfs::core::Tensor::empty({num_verts, 2}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto* const data = poly_cpu.ptr<float>();
        for (size_t i = 0; i < num_verts; ++i) {
            data[i * 2] = (lasso_points_[i].x - bounds.x) * scale_x;
            data[i * 2 + 1] = (lasso_points_[i].y - bounds.y) * scale_y;
        }
        const auto poly_gpu = poly_cpu.cuda();

        auto selection = cumulative_selection_.to(lfs::core::DataType::Bool).cuda();
        const bool add_mode = (current_action_ == SelectionAction::Add);
        lfs::rendering::polygon_select_mode_tensor(*screen_positions, poly_gpu, selection, add_mode);

        auto new_selection = std::make_shared<lfs::core::Tensor>(selection.clone());
        sm->getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(selection)));

        if (auto* const ch = ctx.getCommandHistory()) {
            ch->execute(std::make_unique<command::SelectionCommand>(
                sm, selection_before_stroke_, new_selection));
        }
        selection_before_stroke_.reset();
        rm->clearBrushState();
        rm->markDirty();
    }

    void SelectionTool::selectInPolygon(const ToolContext& ctx) {
        if (!polygon_closed_ || polygon_points_.size() < 3) return;

        auto* const rm = ctx.getRenderingManager();
        auto* const sm = ctx.getSceneManager();
        if (!rm || !sm || !cumulative_selection_.is_valid()) return;

        const auto positions = rm->getScreenPositions();
        if (!positions || !positions->is_valid()) return;

        const auto& bounds = ctx.getViewportBounds();
        const auto& viewport = ctx.getViewport();
        const auto& cached = rm->getCachedResult();
        const int render_w = cached.image ? static_cast<int>(cached.image->size(2)) : viewport.windowSize.x;
        const int render_h = cached.image ? static_cast<int>(cached.image->size(1)) : viewport.windowSize.y;
        const float scale_x = static_cast<float>(render_w) / bounds.width;
        const float scale_y = static_cast<float>(render_h) / bounds.height;
        const size_t num_verts = polygon_points_.size();

        auto poly_cpu = lfs::core::Tensor::empty({num_verts, 2}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto* const data = poly_cpu.ptr<float>();
        for (size_t i = 0; i < num_verts; ++i) {
            data[i * 2] = (polygon_points_[i].x - bounds.x) * scale_x;
            data[i * 2 + 1] = (polygon_points_[i].y - bounds.y) * scale_y;
        }
        const auto poly_gpu = poly_cpu.cuda();

        auto selection = cumulative_selection_.to(lfs::core::DataType::Bool).cuda();
        const bool add_mode = (current_action_ == SelectionAction::Add);
        lfs::rendering::polygon_select_mode_tensor(*positions, poly_gpu, selection, add_mode);

        auto new_selection = std::make_shared<lfs::core::Tensor>(selection.clone());
        sm->getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(selection)));

        if (auto* const ch = ctx.getCommandHistory()) {
            ch->execute(std::make_unique<command::SelectionCommand>(
                sm, selection_before_stroke_, new_selection));
        }
        selection_before_stroke_.reset();
        rm->clearBrushState();
        rm->markDirty();
    }

    void SelectionTool::resetPolygon() {
        polygon_points_.clear();
        polygon_closed_ = false;
        polygon_dragged_vertex_ = -1;
    }

    void SelectionTool::clearPolygon() {
        if (polygon_points_.empty()) return;

        if (tool_context_) {
            if (auto* const rm = tool_context_->getRenderingManager()) {
                rm->clearPreviewSelection();
                rm->markDirty();
            }
        }
        preview_selection_ = lfs::core::Tensor();
        resetPolygon();
        selection_before_stroke_.reset();
    }

    void SelectionTool::onSelectionModeChanged() {
        clearPolygon();
        is_rect_dragging_ = false;
        is_lasso_dragging_ = false;
        lasso_points_.clear();
        is_painting_ = false;
        cumulative_selection_ = lfs::core::Tensor();
        preview_selection_ = lfs::core::Tensor();
        selection_before_stroke_.reset();
    }

    void SelectionTool::clearPreview(const ToolContext& ctx) {
        preview_selection_ = lfs::core::Tensor();
        if (auto* const rm = ctx.getRenderingManager())
            rm->clearPreviewSelection();
    }

    void SelectionTool::updateRectanglePreview(const ToolContext& ctx) {
        auto* const rm = ctx.getRenderingManager();
        if (!rm) return;

        const auto screen_positions = rm->getScreenPositions();
        if (!screen_positions || !screen_positions->is_valid()) return;

        const auto& bounds = ctx.getViewportBounds();
        const auto& viewport = ctx.getViewport();
        const auto& cached = rm->getCachedResult();
        const int render_w = cached.image ? static_cast<int>(cached.image->size(2)) : viewport.windowSize.x;
        const int render_h = cached.image ? static_cast<int>(cached.image->size(1)) : viewport.windowSize.y;
        const float scale_x = static_cast<float>(render_w) / bounds.width;
        const float scale_y = static_cast<float>(render_h) / bounds.height;

        const float x0 = (std::min(rect_start_.x, rect_end_.x) - bounds.x) * scale_x;
        const float y0 = (std::min(rect_start_.y, rect_end_.y) - bounds.y) * scale_y;
        const float x1 = (std::max(rect_start_.x, rect_end_.x) - bounds.x) * scale_x;
        const float y1 = (std::max(rect_start_.y, rect_end_.y) - bounds.y) * scale_y;

        const size_t n = screen_positions->size(0);
        if (!preview_selection_.is_valid() || preview_selection_.size(0) != n) {
            preview_selection_ = lfs::core::Tensor::zeros({n}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
        } else {
            preview_selection_.fill_(false);
        }

        lfs::rendering::rect_select_tensor(*screen_positions, x0, y0, x1, y1, preview_selection_);
        rm->setPreviewSelection(&preview_selection_);
    }

    void SelectionTool::updateLassoPreview(const ToolContext& ctx) {
        if (lasso_points_.size() < 3) return;

        auto* const rm = ctx.getRenderingManager();
        if (!rm) return;

        const auto screen_positions = rm->getScreenPositions();
        if (!screen_positions || !screen_positions->is_valid()) return;

        const auto& bounds = ctx.getViewportBounds();
        const auto& viewport = ctx.getViewport();
        const auto& cached = rm->getCachedResult();
        const int render_w = cached.image ? static_cast<int>(cached.image->size(2)) : viewport.windowSize.x;
        const int render_h = cached.image ? static_cast<int>(cached.image->size(1)) : viewport.windowSize.y;
        const float scale_x = static_cast<float>(render_w) / bounds.width;
        const float scale_y = static_cast<float>(render_h) / bounds.height;

        const size_t num_verts = lasso_points_.size();
        auto poly_cpu = lfs::core::Tensor::empty({num_verts, 2}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto* const data = poly_cpu.ptr<float>();
        for (size_t i = 0; i < num_verts; ++i) {
            data[i * 2] = (lasso_points_[i].x - bounds.x) * scale_x;
            data[i * 2 + 1] = (lasso_points_[i].y - bounds.y) * scale_y;
        }
        const auto poly_gpu = poly_cpu.cuda();

        const size_t n = screen_positions->size(0);
        if (!preview_selection_.is_valid() || preview_selection_.size(0) != n) {
            preview_selection_ = lfs::core::Tensor::zeros({n}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
        } else {
            preview_selection_.fill_(false);
        }

        lfs::rendering::polygon_select_tensor(*screen_positions, poly_gpu, preview_selection_);
        rm->setPreviewSelection(&preview_selection_);
    }

    void SelectionTool::prepareSelectionState(const ToolContext& ctx, const bool add_to_existing) {
        auto* const sm = ctx.getSceneManager();
        if (!sm) return;

        const size_t num_gaussians = sm->getScene().getTotalGaussianCount();
        if (num_gaussians == 0) return;

        auto existing = sm->getScene().getSelectionMask();
        selection_before_stroke_ = (existing && existing->is_valid())
            ? std::make_shared<lfs::core::Tensor>(existing->clone())
            : nullptr;

        if (add_to_existing && existing && existing->is_valid() && existing->size(0) == num_gaussians) {
            cumulative_selection_ = existing->to(lfs::core::DataType::Bool);
        } else {
            cumulative_selection_ = lfs::core::Tensor::zeros(
                {num_gaussians}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
        }
    }

    void SelectionTool::updatePolygonPreview(const ToolContext& ctx) {
        if (!polygon_closed_ || polygon_points_.size() < 3) return;

        auto* const rm = ctx.getRenderingManager();
        if (!rm) return;

        const auto positions = rm->getScreenPositions();
        if (!positions || !positions->is_valid()) return;

        const auto& bounds = ctx.getViewportBounds();
        const auto& viewport = ctx.getViewport();
        const auto& cached = rm->getCachedResult();
        const int render_w = cached.image ? static_cast<int>(cached.image->size(2)) : viewport.windowSize.x;
        const int render_h = cached.image ? static_cast<int>(cached.image->size(1)) : viewport.windowSize.y;
        const float scale_x = static_cast<float>(render_w) / bounds.width;
        const float scale_y = static_cast<float>(render_h) / bounds.height;

        const size_t n = positions->size(0);
        const size_t num_verts = polygon_points_.size();

        auto poly_cpu = lfs::core::Tensor::empty({num_verts, 2}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto* const data = poly_cpu.ptr<float>();
        for (size_t i = 0; i < num_verts; ++i) {
            data[i * 2] = (polygon_points_[i].x - bounds.x) * scale_x;
            data[i * 2 + 1] = (polygon_points_[i].y - bounds.y) * scale_y;
        }
        const auto poly_gpu = poly_cpu.cuda();

        if (!preview_selection_.is_valid() || preview_selection_.size(0) != n) {
            preview_selection_ = lfs::core::Tensor::zeros({n}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
        } else {
            preview_selection_.fill_(false);
        }

        lfs::rendering::polygon_select_tensor(*positions, poly_gpu, preview_selection_);
        rm->setPreviewSelection(&preview_selection_);
    }

    int SelectionTool::findPolygonVertexAt(const float x, const float y) const {
        constexpr float RADIUS_SQ = POLYGON_VERTEX_RADIUS * POLYGON_VERTEX_RADIUS;
        const glm::vec2 p(x, y);
        for (size_t i = 0; i < polygon_points_.size(); ++i) {
            const glm::vec2 d = p - polygon_points_[i];
            if (glm::dot(d, d) <= RADIUS_SQ)
                return static_cast<int>(i);
        }
        return -1;
    }

    int SelectionTool::findPolygonEdgeAt(const float x, const float y, float& t_out) const {
        if (polygon_points_.size() < 2) return -1;

        constexpr float EDGE_THRESHOLD_SQ = 8.0f * 8.0f;
        const glm::vec2 p(x, y);
        const size_t n = polygon_points_.size();

        for (size_t i = 0; i < n; ++i) {
            const glm::vec2& a = polygon_points_[i];
            const glm::vec2& b = polygon_points_[(i + 1) % n];
            const glm::vec2 ab = b - a;
            const float len_sq = glm::dot(ab, ab);
            if (len_sq < 1e-6f) continue;

            const float t = glm::clamp(glm::dot(p - a, ab) / len_sq, 0.0f, 1.0f);
            const glm::vec2 d = p - (a + t * ab);
            if (glm::dot(d, d) <= EDGE_THRESHOLD_SQ) {
                t_out = t;
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    bool SelectionTool::handleKeyPress(const int key, const int mods, const ToolContext& ctx) {
        if (!isEnabled()) return false;

        const auto* const rm = ctx.getRenderingManager();
        const auto sel_mode = rm ? rm->getSelectionMode() : lfs::rendering::SelectionMode::Centers;

        // Ctrl+F toggles depth filter
        if (key == GLFW_KEY_F && (mods & GLFW_MOD_CONTROL) && sel_mode != lfs::rendering::SelectionMode::Rings) {
            if (depth_filter_enabled_) {
                disableDepthFilter(ctx);
            } else {
                depth_filter_enabled_ = true;
                updateSelectionCropBox(ctx, true);  // Force initial update
            }
            ctx.requestRender();
            return true;
        }

        // Escape disables depth filter
        if (key == GLFW_KEY_ESCAPE && depth_filter_enabled_) {
            disableDepthFilter(ctx);
            ctx.requestRender();
            return true;
        }

        // Polygon-specific key handling
        if (sel_mode != lfs::rendering::SelectionMode::Polygon) return false;

        // Confirm polygon selection
        if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER) {
            if (polygon_closed_ && polygon_points_.size() >= 3) {
                const bool ctrl = (mods & GLFW_MOD_CONTROL) != 0;
                const bool shift = (mods & GLFW_MOD_SHIFT) != 0;
                clearPreview(ctx);
                prepareSelectionState(ctx, ctrl || shift);
                current_action_ = shift ? SelectionAction::Remove : SelectionAction::Add;
                selectInPolygon(ctx);
                resetPolygon();
                ctx.requestRender();
                return true;
            }
        }

        // Cancel and restore
        if (key == GLFW_KEY_ESCAPE && !polygon_points_.empty()) {
            clearPreview(ctx);
            resetPolygon();
            selection_before_stroke_.reset();
            ctx.requestRender();
            return true;
        }

        return false;
    }

    void SelectionTool::resetDepthFilter() {
        depth_filter_enabled_ = false;
        depth_far_ = 100.0f;
        frustum_half_width_ = 50.0f;
    }

    void SelectionTool::updateSelectionCropBox(const ToolContext& ctx, const bool force) {
        auto* const rm = ctx.getRenderingManager();
        if (!rm) return;

        const auto& viewport = ctx.getViewport();
        const glm::vec3& cam_pos = viewport.camera.t;
        const glm::vec3 cam_fwd = viewport.camera.R[2];

        // Skip update if camera hasn't moved (unless forced)
        constexpr float EPSILON = 1e-5f;
        if (!force &&
            glm::dot(cam_pos - last_cam_pos_, cam_pos - last_cam_pos_) < EPSILON &&
            glm::dot(cam_fwd - last_cam_fwd_, cam_fwd - last_cam_fwd_) < EPSILON) {
            return;
        }
        last_cam_pos_ = cam_pos;
        last_cam_fwd_ = cam_fwd;

        // Build camera-aligned crop box transform
        const glm::quat cam_quat = glm::quat_cast(viewport.camera.R);
        const lfs::geometry::EuclideanTransform crop_transform(cam_quat, cam_pos);

        // Crop box in camera local space: X bounded by width, Y unbounded, Z from 0 to depth_far
        constexpr float Y_BOUND = 10000.0f;
        const glm::vec3 crop_min(-frustum_half_width_, -Y_BOUND, 0.0f);
        const glm::vec3 crop_max(frustum_half_width_, Y_BOUND, depth_far_);

        auto settings = rm->getSettings();
        settings.crop_transform = crop_transform;
        settings.crop_min = crop_min;
        settings.crop_max = crop_max;
        settings.use_crop_box = true;
        settings.show_crop_box = false;
        settings.crop_inverse = false;
        rm->updateSettings(settings);
    }

    void SelectionTool::disableDepthFilter(const ToolContext& ctx) {
        depth_filter_enabled_ = false;

        auto* const rm = ctx.getRenderingManager();
        if (rm) {
            auto settings = rm->getSettings();
            settings.use_crop_box = false;
            rm->updateSettings(settings);
        }
    }

    void SelectionTool::drawDepthFrustum(const ToolContext& ctx) const {
        constexpr float BAR_HEIGHT = 8.0f;
        constexpr float BAR_WIDTH = 200.0f;
        constexpr float FONT_SIZE = 16.0f;
        constexpr float HINT_FONT_SIZE = 12.0f;
        constexpr ImU32 BAR_BG_COLOR = IM_COL32(50, 50, 50, 180);
        constexpr ImU32 BAR_FILL_COLOR = IM_COL32(255, 180, 50, 200);
        constexpr ImU32 MARKER_COLOR = IM_COL32(255, 100, 100, 200);
        constexpr ImU32 TEXT_COLOR = IM_COL32(255, 255, 255, 255);
        constexpr ImU32 SHADOW_COLOR = IM_COL32(0, 0, 0, 180);
        constexpr ImU32 HINT_COLOR = IM_COL32(180, 180, 180, 200);

        const auto& bounds = ctx.getViewportBounds();
        const float bar_x = bounds.x + 10.0f;
        const float bar_y = bounds.y + bounds.height - 45.0f;

        ImDrawList* const draw_list = ImGui::GetForegroundDrawList();

        // Background bar
        draw_list->AddRectFilled({bar_x, bar_y}, {bar_x + BAR_WIDTH, bar_y + BAR_HEIGHT}, BAR_BG_COLOR);

        // Map depth to bar position (log scale)
        const float log_range = std::log10(DEPTH_MAX) - std::log10(DEPTH_MIN);
        const float far_pos = bar_x + (std::log10(depth_far_) - std::log10(DEPTH_MIN)) / log_range * BAR_WIDTH;

        // Fill and marker
        draw_list->AddRectFilled({bar_x, bar_y}, {far_pos, bar_y + BAR_HEIGHT}, BAR_FILL_COLOR);
        draw_list->AddLine({far_pos, bar_y - 3}, {far_pos, bar_y + BAR_HEIGHT + 3}, MARKER_COLOR, 2.0f);

        // Info text with shadow
        char info_text[64];
        if (frustum_half_width_ < WIDTH_MAX - 1.0f) {
            snprintf(info_text, sizeof(info_text), "Depth: %.1f  Width: %.1f", depth_far_, frustum_half_width_ * 2.0f);
        } else {
            snprintf(info_text, sizeof(info_text), "Depth: %.1f", depth_far_);
        }
        const ImVec2 text_pos(bar_x, bar_y - 20.0f);
        draw_list->AddText(ImGui::GetFont(), FONT_SIZE, {text_pos.x + 1, text_pos.y + 1}, SHADOW_COLOR, info_text);
        draw_list->AddText(ImGui::GetFont(), FONT_SIZE, text_pos, TEXT_COLOR, info_text);

        // Hint
        draw_list->AddText(ImGui::GetFont(), HINT_FONT_SIZE, {bar_x, bar_y + BAR_HEIGHT + 5.0f}, HINT_COLOR,
                           "Alt+Scroll: depth | Ctrl+Alt+Scroll: width | Esc: off");
    }

} // namespace lfs::vis::tools
