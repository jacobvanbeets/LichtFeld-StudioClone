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
#include <imgui.h>

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
    }

    bool SelectionTool::handleMouseButton(const int button, const int action, const int mods,
                                           const double x, const double y, const ToolContext& ctx) {
        if (!isEnabled() || button != GLFW_MOUSE_BUTTON_LEFT) return false;

        const auto* const rm = ctx.getRenderingManager();
        const auto sel_mode = rm ? rm->getSelectionMode() : lfs::rendering::SelectionMode::Centers;
        const bool is_rect_mode = (sel_mode == lfs::rendering::SelectionMode::Rectangle);
        const bool is_lasso_mode = (sel_mode == lfs::rendering::SelectionMode::Lasso);

        if (action == GLFW_PRESS) {
            const bool ctrl = (mods & GLFW_MOD_CONTROL) != 0;
            const bool shift = (mods & GLFW_MOD_SHIFT) != 0;

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
            if (is_rect_dragging_) {
                selectInRectangle(ctx);
                is_rect_dragging_ = false;
                return true;
            }
            if (is_lasso_dragging_) {
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

        if (is_rect_dragging_) {
            rect_end_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));
            ctx.requestRender();
            return true;
        }

        if (is_lasso_dragging_) {
            const glm::vec2 new_point(static_cast<float>(x), static_cast<float>(y));
            // Only add point if moved enough (reduces point count)
            if (lasso_points_.empty() || glm::distance(lasso_points_.back(), new_point) > 3.0f) {
                lasso_points_.push_back(new_point);
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

        const auto* const rm = ctx.getRenderingManager();
        if (rm) {
            const auto mode = rm->getSelectionMode();
            if (mode == lfs::rendering::SelectionMode::Rings ||
                mode == lfs::rendering::SelectionMode::Rectangle ||
                mode == lfs::rendering::SelectionMode::Lasso) {
                return false;
            }
        }

        const bool ctrl = (mods & GLFW_MOD_CONTROL) != 0;
        const bool shift = (mods & GLFW_MOD_SHIFT) != 0;
        const bool alt = glfwGetKey(ctx.getWindow(), GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                         glfwGetKey(ctx.getWindow(), GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;

        if (is_painting_ || ctrl || shift || alt) {
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
        }

        if (tool_context_) {
            auto* const rm = tool_context_->getRenderingManager();
            if (rm) {
                rm->setOutputScreenPositions(enabled);
                if (!enabled) {
                    rm->clearBrushState();
                }
                rm->markDirty();
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

        if (tool_context_ && cumulative_selection_.is_valid()) {
            auto* const sm = tool_context_->getSceneManager();
            if (sm) {
                auto mask = cumulative_selection_.to(lfs::core::DataType::UInt8);
                new_selection = std::make_shared<lfs::core::Tensor>(mask.clone());
                sm->getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(mask)));
            }
        }

        if (tool_context_ && new_selection && new_selection->is_valid()) {
            auto* const ch = tool_context_->getCommandHistory();
            auto* const sm = tool_context_->getSceneManager();

            if (ch && sm) {
                auto cmd = std::make_unique<command::SelectionCommand>(
                    sm, selection_before_stroke_, new_selection);
                ch->execute(std::move(cmd));
            }
        }
        selection_before_stroke_.reset();

        if (tool_context_) {
            auto* const rm = tool_context_->getRenderingManager();
            if (rm) {
                rm->clearBrushState();
                rm->markDirty();
            }
        }
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
                auto cpu_sel = cumulative_selection_.cpu();
                cpu_sel.ptr<bool>()[hovered_id] = add_mode;
                cumulative_selection_ = cpu_sel.cuda();

                auto* const sm = ctx.getSceneManager();
                if (sm) {
                    auto mask = cumulative_selection_.to(lfs::core::DataType::UInt8);
                    sm->getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(mask)));
                }
            }
            rm->setBrushState(true, image_x, image_y, 0.0f, add_mode, nullptr, false, 0.0f);
        } else {
            const float scaled_radius = brush_radius_ * scale_x;
            rm->setBrushState(true, image_x, image_y, scaled_radius, add_mode, &cumulative_selection_);

            auto* const sm = ctx.getSceneManager();
            if (sm && cumulative_selection_.is_valid()) {
                auto mask = cumulative_selection_.to(lfs::core::DataType::UInt8);
                sm->getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(mask)));
            }
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

        // Convert screen rect to image coords
        const float img_x1 = (std::min(rect_start_.x, rect_end_.x) - bounds.x) * scale_x;
        const float img_y1 = (std::min(rect_start_.y, rect_end_.y) - bounds.y) * scale_y;
        const float img_x2 = (std::max(rect_start_.x, rect_end_.x) - bounds.x) * scale_x;
        const float img_y2 = (std::max(rect_start_.y, rect_end_.y) - bounds.y) * scale_y;

        // Get screen positions on CPU
        auto positions_cpu = screen_positions->cpu();
        const auto* pos_data = positions_cpu.ptr<float>();
        const size_t num_gaussians = static_cast<size_t>(positions_cpu.size(0));

        auto sel_cpu = cumulative_selection_.cpu();
        auto* const sel_data = sel_cpu.ptr<bool>();
        const bool add_mode = (current_action_ == SelectionAction::Add);

        for (size_t i = 0; i < num_gaussians; ++i) {
            const float px = pos_data[i * 2];
            const float py = pos_data[i * 2 + 1];

            if (px >= img_x1 && px <= img_x2 && py >= img_y1 && py <= img_y2) {
                sel_data[i] = add_mode;
            }
        }

        cumulative_selection_ = sel_cpu.cuda();

        // Apply selection
        auto mask = cumulative_selection_.to(lfs::core::DataType::UInt8);
        auto new_selection = std::make_shared<lfs::core::Tensor>(mask.clone());
        sm->getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(mask)));

        // Create undo command
        auto* const ch = ctx.getCommandHistory();
        if (ch && new_selection && new_selection->is_valid()) {
            auto cmd = std::make_unique<command::SelectionCommand>(
                sm, selection_before_stroke_, new_selection);
            ch->execute(std::move(cmd));
        }
        selection_before_stroke_.reset();

        rm->clearBrushState();
        rm->markDirty();
    }

    bool SelectionTool::pointInPolygon(const float px, const float py, const std::vector<glm::vec2>& polygon) {
        if (polygon.size() < 3) return false;

        // Ray casting algorithm
        bool inside = false;
        const size_t n = polygon.size();
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            const float xi = polygon[i].x, yi = polygon[i].y;
            const float xj = polygon[j].x, yj = polygon[j].y;

            if (((yi > py) != (yj > py)) &&
                (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
                inside = !inside;
            }
        }
        return inside;
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

        // Convert lasso points to image coords
        std::vector<glm::vec2> img_lasso;
        img_lasso.reserve(lasso_points_.size());
        for (const auto& pt : lasso_points_) {
            img_lasso.emplace_back((pt.x - bounds.x) * scale_x, (pt.y - bounds.y) * scale_y);
        }

        // Get screen positions on CPU
        auto positions_cpu = screen_positions->cpu();
        const auto* pos_data = positions_cpu.ptr<float>();
        const size_t num_gaussians = static_cast<size_t>(positions_cpu.size(0));

        auto sel_cpu = cumulative_selection_.cpu();
        auto* const sel_data = sel_cpu.ptr<bool>();
        const bool add_mode = (current_action_ == SelectionAction::Add);

        for (size_t i = 0; i < num_gaussians; ++i) {
            const float px = pos_data[i * 2];
            const float py = pos_data[i * 2 + 1];

            if (pointInPolygon(px, py, img_lasso)) {
                sel_data[i] = add_mode;
            }
        }

        cumulative_selection_ = sel_cpu.cuda();

        // Apply selection
        auto mask = cumulative_selection_.to(lfs::core::DataType::UInt8);
        auto new_selection = std::make_shared<lfs::core::Tensor>(mask.clone());
        sm->getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(mask)));

        // Create undo command
        auto* const ch = ctx.getCommandHistory();
        if (ch && new_selection && new_selection->is_valid()) {
            auto cmd = std::make_unique<command::SelectionCommand>(
                sm, selection_before_stroke_, new_selection);
            ch->execute(std::move(cmd));
        }
        selection_before_stroke_.reset();

        rm->clearBrushState();
        rm->markDirty();
    }

} // namespace lfs::vis::tools
