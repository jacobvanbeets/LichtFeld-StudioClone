/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"
#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "model_renderability.hpp"
#include "rendering/rasterizer/rasterization/include/rasterization_config.h"
#include "scene/scene_manager.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "viewport_region_utils.hpp"
#include "viewport_request_builder.hpp"
#include <algorithm>
#include <cmath>
#include <shared_mutex>

namespace lfs::vis {

    namespace {
        [[nodiscard]] std::optional<std::shared_lock<std::shared_mutex>> acquireLiveModelRenderLock(
            const SceneManager* const scene_manager) {
            std::optional<std::shared_lock<std::shared_mutex>> lock;
            if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
                if (const auto* trainer = tm->getTrainer()) {
                    lock.emplace(trainer->getRenderMutex());
                }
            }
            return lock;
        }
    } // namespace

    RenderingManager::VulkanFrameResult RenderingManager::renderVulkanFrame(const RenderContext& context) {
        SceneManager* const scene_manager = context.scene_manager;

        if (!engine_) {
            engine_ = lfs::rendering::RenderingEngine::create();
        }
        if (!raster_initialized_) {
            if (auto init_result = engine_->initializeRasterOnly(); !init_result) {
                LOG_ERROR("Failed to initialize Vulkan raster path: {}", init_result.error());
                return {.image = vulkan_viewport_image_,
                        .size = vulkan_viewport_image_size_,
                        .flip_y = vulkan_viewport_image_flip_y_};
            }
            raster_initialized_ = true;
        }

        if (scene_manager && (dirty_mask_.load(std::memory_order_relaxed) & DirtyFlag::SELECTION)) {
            for (const auto& group : scene_manager->getScene().getSelectionGroups()) {
                lfs::rendering::config::setSelectionGroupColor(
                    group.id, make_float3(group.color.x, group.color.y, group.color.z));
            }
        }

        const auto framebuffer_region =
            resolveFramebufferViewportRegion(context.viewport, context.logical_screen_size, context.viewport_region);
        glm::ivec2 current_size = context.viewport.frameBufferSize;
        if (context.viewport_region) {
            current_size = framebuffer_region.size;
        }
        if (current_size.x <= 0 || current_size.y <= 0) {
            return {.image = vulkan_viewport_image_,
                    .size = vulkan_viewport_image_size_,
                    .flip_y = vulkan_viewport_image_flip_y_};
        }

        const auto resize_result = frame_lifecycle_service_.handleViewportResize(current_size);
        if (resize_result.dirty) {
            markDirty(resize_result.dirty);
        }

        auto render_lock = acquireLiveModelRenderLock(scene_manager);

        const lfs::core::SplatData* const model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
        const bool has_renderable_model = hasRenderableGaussians(model);
        const size_t model_ptr = reinterpret_cast<size_t>(model);

        if (const auto model_change = frame_lifecycle_service_.handleModelChange(model_ptr, viewport_artifact_service_);
            model_change.changed) {
            vulkan_viewport_image_.reset();
            vulkan_viewport_image_size_ = {0, 0};
            vulkan_viewport_image_flip_y_ = false;
            markDirty(DirtyFlag::ALL);
        }

        const bool is_training = scene_manager && scene_manager->hasDataset() &&
                                 scene_manager->getTrainerManager() &&
                                 scene_manager->getTrainerManager()->isRunning();
        if (const DirtyMask training_dirty = frame_lifecycle_service_.handleTrainingRefresh(
                is_training,
                framerate_controller_.getSettings().training_frame_refresh_time_sec);
            training_dirty) {
            markDirty(training_dirty);
        }

        if (const DirtyMask required_dirty = frame_lifecycle_service_.requiredDirtyMask(
                vulkan_viewport_image_ != nullptr,
                has_renderable_model,
                settings_.split_view_mode);
            required_dirty) {
            dirty_mask_.fetch_or(required_dirty, std::memory_order_relaxed);
        }

        DirtyMask frame_dirty = dirty_mask_.exchange(0);
        if (!has_renderable_model) {
            vulkan_viewport_image_.reset();
            vulkan_viewport_image_size_ = {0, 0};
            vulkan_viewport_image_flip_y_ = false;
            render_lock.reset();
            return {};
        }

        if (frame_dirty == 0 && vulkan_viewport_image_) {
            render_lock.reset();
            return {.image = vulkan_viewport_image_,
                    .size = vulkan_viewport_image_size_,
                    .flip_y = vulkan_viewport_image_flip_y_};
        }

        const float scale = std::clamp(settings_.render_scale, 0.25f, 1.0f);
        glm::ivec2 render_size(
            std::max(static_cast<int>(std::lround(static_cast<float>(current_size.x) * scale)), 1),
            std::max(static_cast<int>(std::lround(static_cast<float>(current_size.y) * scale)), 1));

        SceneRenderState scene_state;
        if (scene_manager) {
            scene_state = scene_manager->buildRenderState();
        }

        const FrameContext frame_ctx{
            .viewport = context.viewport,
            .viewport_region = context.viewport_region,
            .render_lock_held = render_lock.has_value(),
            .scene_manager = scene_manager,
            .model = model,
            .scene_state = std::move(scene_state),
            .settings = settings_,
            .render_size = render_size,
            .viewport_pos = {0, 0},
            .frame_dirty = frame_dirty,
            .cursor_preview = viewport_overlay_service_.cursorPreview(),
            .gizmo = viewport_overlay_service_.makeFrameGizmoState(),
            .hovered_camera_id = camera_interaction_service_.hoveredCameraId(),
            .current_camera_id = camera_interaction_service_.currentCameraId(),
            .hovered_gaussian_id = viewport_overlay_service_.hoveredGaussianId(),
            .selection_flash_intensity = getSelectionFlashIntensity(),
            .view_panels = {}};

        auto request = buildViewportRenderRequest(frame_ctx, render_size);
        auto render_result = engine_->renderGaussiansImage(*model, request);
        render_lock.reset();

        if (!render_result || !render_result->image) {
            LOG_ERROR("Failed to render Vulkan viewport image: {}",
                      render_result ? "missing image payload" : render_result.error());
            vulkan_viewport_image_.reset();
            vulkan_viewport_image_size_ = {0, 0};
            vulkan_viewport_image_flip_y_ = false;
            return {};
        }

        vulkan_viewport_image_ = std::move(render_result->image);
        vulkan_viewport_image_size_ = render_size;
        vulkan_viewport_image_flip_y_ = !render_result->metadata.flip_y;

        if (resize_result.completed) {
            frame_lifecycle_service_.noteResizeCompleted();
            lfs::core::Tensor::trim_memory_pool();
        }

        queueCameraMetricsRefreshIfStale(scene_manager);
        viewport_interaction_context_.scene_manager = scene_manager;

        return {.image = vulkan_viewport_image_,
                .size = vulkan_viewport_image_size_,
                .flip_y = vulkan_viewport_image_flip_y_};
    }

} // namespace lfs::vis
