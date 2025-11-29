/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/brush_tool.hpp"
#include "command/command_history.hpp"
#include "command/commands/saturation_command.hpp"
#include "command/commands/selection_command.hpp"
#include "internal/viewport.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"
#include <imgui.h>

namespace lfs::vis::tools {

    BrushTool::BrushTool() = default;

    bool BrushTool::initialize(const ToolContext& ctx) {
        tool_context_ = &ctx;
        return true;
    }

    void BrushTool::shutdown() {
        tool_context_ = nullptr;
        is_painting_ = false;
    }

    void BrushTool::update([[maybe_unused]] const ToolContext& ctx) {
        if (isEnabled()) {
            double mx, my;
            glfwGetCursorPos(ctx.getWindow(), &mx, &my);
            last_mouse_pos_ = glm::vec2(static_cast<float>(mx), static_cast<float>(my));
        }
    }

    void BrushTool::renderUI([[maybe_unused]] const lfs::vis::gui::UIContext& ui_ctx,
                              [[maybe_unused]] bool* p_open) {
        if (!isEnabled() || ImGui::GetIO().WantCaptureMouse) return;

        ImDrawList* const draw_list = ImGui::GetForegroundDrawList();
        const ImVec2 mouse_pos(last_mouse_pos_.x, last_mouse_pos_.y);

        const ImU32 brush_color = (current_mode_ == BrushMode::Select)
            ? IM_COL32(100, 180, 255, 220)
            : IM_COL32(255, 200, 100, 220);

        draw_list->AddCircle(mouse_pos, brush_radius_, brush_color, 32, 2.0f);
        draw_list->AddCircleFilled(mouse_pos, 3.0f, brush_color);

        // Show mode and value next to circle
        static char info_text[32];
        if (current_mode_ == BrushMode::Select) {
            if (is_painting_) {
                snprintf(info_text, sizeof(info_text), "SEL %s",
                         (current_action_ == BrushAction::Add) ? "+" : "-");
            } else {
                snprintf(info_text, sizeof(info_text), "SEL");
            }
        } else {
            // Always show saturation amount
            snprintf(info_text, sizeof(info_text), "SAT %+.0f%%", saturation_amount_ * 100.0f);
        }

        // Larger font with background for better readability
        constexpr float font_size = 22.0f;
        const ImVec2 text_pos(mouse_pos.x + brush_radius_ + 10, mouse_pos.y - font_size / 2);

        // Draw text shadow/outline for contrast
        const ImU32 shadow_color = IM_COL32(0, 0, 0, 180);
        draw_list->AddText(ImGui::GetFont(), font_size, ImVec2(text_pos.x + 1, text_pos.y + 1), shadow_color, info_text);
        draw_list->AddText(ImGui::GetFont(), font_size, text_pos, IM_COL32(255, 255, 255, 255), info_text);
    }

    bool BrushTool::handleMouseButton(int button, int action, int mods, double x, double y,
                                       const ToolContext& ctx) {
        if (!isEnabled() || button != GLFW_MOUSE_BUTTON_LEFT) return false;

        if (action == GLFW_PRESS) {
            const bool ctrl = (mods & GLFW_MOD_CONTROL) != 0;
            const bool shift = (mods & GLFW_MOD_SHIFT) != 0;

            if (current_mode_ == BrushMode::Select) {
                if (ctrl) {
                    beginStroke(x, y, BrushAction::Add, false, ctx);
                } else if (shift) {
                    beginStroke(x, y, BrushAction::Remove, false, ctx);
                } else {
                    beginStroke(x, y, BrushAction::Add, true, ctx);
                }
                updateSelectionAtPoint(x, y, ctx);
            } else if (current_mode_ == BrushMode::Saturation) {
                beginSaturationStroke(x, y, ctx);
                updateSaturationAtPoint(x, y, ctx);
            }
            return true;
        }

        if (action == GLFW_RELEASE && is_painting_) {
            if (current_mode_ == BrushMode::Select) {
                endStroke();
            } else if (current_mode_ == BrushMode::Saturation) {
                endSaturationStroke();
            }
            return true;
        }
        return false;
    }

    bool BrushTool::handleMouseMove(double x, double y, const ToolContext& ctx) {
        if (!isEnabled()) return false;

        last_mouse_pos_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));

        if (is_painting_) {
            if (current_mode_ == BrushMode::Select) {
                updateSelectionAtPoint(x, y, ctx);
            } else if (current_mode_ == BrushMode::Saturation) {
                updateSaturationAtPoint(x, y, ctx);
            }
            ctx.requestRender();
            return true;
        }

        // Update brush preview state for highlighting (without selection tensor)
        updateBrushPreview(x, y, ctx);
        ctx.requestRender();
        return false;
    }

    bool BrushTool::handleScroll([[maybe_unused]] double x_offset, double y_offset,
                                  int mods, const ToolContext& ctx) {
        if (!isEnabled()) return false;

        const bool ctrl = (mods & GLFW_MOD_CONTROL) != 0;
        const bool shift = (mods & GLFW_MOD_SHIFT) != 0;
        const bool alt = glfwGetKey(ctx.getWindow(), GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                         glfwGetKey(ctx.getWindow(), GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;

        // Alt+scroll: only handle in saturation mode, otherwise let selection tool handle it
        if (alt) {
            if (current_mode_ == BrushMode::Saturation) {
                const float delta = (y_offset > 0) ? 0.1f : -0.1f;
                saturation_amount_ = std::clamp(saturation_amount_ + delta, -1.0f, 1.0f);
                updateBrushPreview(last_mouse_pos_.x, last_mouse_pos_.y, ctx);
                ctx.requestRender();
                return true;
            }
            return false;  // Let other handlers deal with Alt+scroll
        }

        // Ctrl or Shift or painting: adjust brush radius
        if (is_painting_ || ctrl || shift) {
            const float scale = (y_offset > 0) ? 1.1f : 0.9f;
            brush_radius_ = std::clamp(brush_radius_ * scale, 1.0f, 500.0f);
            updateBrushPreview(last_mouse_pos_.x, last_mouse_pos_.y, ctx);
            ctx.requestRender();
            return true;
        }
        return false;
    }

    bool BrushTool::handleKeyPress(int key, [[maybe_unused]] int mods, [[maybe_unused]] const ToolContext& ctx) {
        if (!isEnabled()) return false;

        // B key cycles through brush modes
        if (key == GLFW_KEY_B) {
            if (current_mode_ == BrushMode::Select) {
                current_mode_ = BrushMode::Saturation;
            } else {
                current_mode_ = BrushMode::Select;
            }
            return true;
        }
        return false;
    }

    void BrushTool::onEnabledChanged(bool enabled) {
        if (!enabled) {
            is_painting_ = false;
        }

        if (tool_context_) {
            auto* const rm = tool_context_->getRenderingManager();
            if (rm) {
                rm->setOutputScreenPositions(enabled);
                if (enabled) rm->markDirty();
            }
        }
    }

    void BrushTool::beginStroke([[maybe_unused]] double x, [[maybe_unused]] double y,
                                 BrushAction action, bool clear_existing, const ToolContext& ctx) {
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

    void BrushTool::endStroke() {
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

    void BrushTool::beginSaturationStroke([[maybe_unused]] double x, [[maybe_unused]] double y,
                                           const ToolContext& ctx) {
        is_painting_ = true;

        auto* const sm = ctx.getSceneManager();
        if (!sm) return;

        // Get the first visible node and store its SH0 for undo
        auto visible_nodes = sm->getScene().getVisibleNodes();
        if (visible_nodes.empty()) return;

        saturation_node_name_ = visible_nodes[0]->name;
        auto* mutable_node = sm->getScene().getMutableNode(saturation_node_name_);
        if (!mutable_node || !mutable_node->model) return;

        const auto& sh0 = mutable_node->model->sh0();
        if (sh0.is_valid()) {
            sh0_before_stroke_ = std::make_shared<lfs::core::Tensor>(sh0.clone());
        } else {
            sh0_before_stroke_.reset();
        }
    }

    void BrushTool::endSaturationStroke() {
        is_painting_ = false;

        // Create undo command for saturation changes
        if (tool_context_ && sh0_before_stroke_ && sh0_before_stroke_->is_valid()) {
            auto* const ch = tool_context_->getCommandHistory();
            auto* const sm = tool_context_->getSceneManager();

            if (ch && sm && !saturation_node_name_.empty()) {
                auto* mutable_node = sm->getScene().getMutableNode(saturation_node_name_);
                if (mutable_node && mutable_node->model) {
                    const auto& sh0 = mutable_node->model->sh0();
                    if (sh0.is_valid()) {
                        auto new_sh0 = std::make_shared<lfs::core::Tensor>(sh0.clone());
                        auto cmd = std::make_unique<command::SaturationCommand>(
                            sm, saturation_node_name_, sh0_before_stroke_, new_sh0);
                        ch->execute(std::move(cmd));
                    }
                }
            }
        }

        sh0_before_stroke_.reset();
        saturation_node_name_.clear();

        if (tool_context_) {
            auto* const rm = tool_context_->getRenderingManager();
            if (rm) {
                rm->clearBrushState();
                rm->markDirty();
            }
        }
    }

    void BrushTool::updateSaturationAtPoint(double x, double y, const ToolContext& ctx) {
        auto* const rm = ctx.getRenderingManager();
        auto* const sm = ctx.getSceneManager();
        if (!rm || !sm) return;

        // Get the first visible node's SplatData (for now, single model support)
        auto visible_nodes = sm->getScene().getVisibleNodes();
        if (visible_nodes.empty()) return;

        auto* mutable_node = sm->getScene().getMutableNode(visible_nodes[0]->name);
        if (!mutable_node || !mutable_node->model) return;

        auto& sh0 = mutable_node->model->sh0();
        if (!sh0.is_valid()) return;

        // Convert screen coords to image coords
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
        const float scaled_radius = brush_radius_ * scale_x;

        // Reshape SH0 from [N, 1, 3] to [N, 3] for the kernel
        auto sh0_reshaped = sh0.reshape({static_cast<int>(sh0.size(0)), 3});

        // Call the saturation adjustment (same amount as preview shows)
        rm->adjustSaturation(image_x, image_y, scaled_radius, saturation_amount_, sh0_reshaped);

        // Also set brush state for preview (in saturation mode)
        rm->setBrushState(true, image_x, image_y, scaled_radius, true, nullptr, true, saturation_amount_);
    }

    void BrushTool::clearSelection(const ToolContext& ctx) {
        auto* const sm = ctx.getSceneManager();
        if (sm) sm->getScene().clearSelection();

        cumulative_selection_ = lfs::core::Tensor();
        ctx.requestRender();
    }

    void BrushTool::updateSelectionAtPoint(double x, double y, const ToolContext& ctx) {
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
        const float scaled_radius = brush_radius_ * scale_x;
        const bool add_mode = (current_action_ == BrushAction::Add);

        rm->setBrushState(true, image_x, image_y, scaled_radius, add_mode, &cumulative_selection_);
    }

    void BrushTool::updateBrushPreview(double x, double y, const ToolContext& ctx) {
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
        const float scaled_radius = brush_radius_ * scale_x;

        // Set brush state for preview (pass saturation mode and amount for saturation preview)
        const bool sat_mode = (current_mode_ == BrushMode::Saturation);
        rm->setBrushState(true, image_x, image_y, scaled_radius, true, nullptr, sat_mode, sat_mode ? saturation_amount_ : 0.0f);
    }

} // namespace lfs::vis::tools
