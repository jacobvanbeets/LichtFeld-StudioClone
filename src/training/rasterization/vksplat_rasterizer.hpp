/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "optimizer/render_output.hpp"
#include <expected>
#include <string>
#include <utility>

namespace lfs::training {

    struct VKSplatRasterizeContext {
        int width = 0;
        int height = 0;
        bool has_visible_splats = false;
        lfs::core::Tensor bg_color;
        lfs::core::Tensor bg_image;
    };

    [[nodiscard]] bool vksplat_backend_available();

    std::expected<std::pair<RenderOutput, VKSplatRasterizeContext>, std::string> vksplat_rasterize_forward(
        const lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        const lfs::core::param::OptimizationParameters& params,
        int tile_x_offset = 0,
        int tile_y_offset = 0,
        int tile_width = 0,
        int tile_height = 0,
        const lfs::core::Tensor& bg_image = {});

    void vksplat_rasterize_backward(
        VKSplatRasterizeContext& ctx,
        const lfs::core::Tensor& grad_image,
        lfs::core::SplatData& gaussian_model,
        const lfs::core::param::OptimizationParameters& params,
        const lfs::core::Tensor& grad_alpha_extra = {},
        int iteration = 0);

    std::expected<bool, std::string> vksplat_flush_model(
        lfs::core::SplatData& gaussian_model);

    [[nodiscard]] size_t vksplat_resident_splat_count();

    RenderOutput vksplat_rasterize(
        const lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        const lfs::core::param::OptimizationParameters& params,
        const lfs::core::Tensor& bg_image = {});

} // namespace lfs::training
