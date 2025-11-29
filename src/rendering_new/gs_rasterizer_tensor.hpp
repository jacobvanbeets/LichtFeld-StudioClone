/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/camera.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"
#include <tuple>

namespace lfs::rendering {

    using lfs::core::Tensor;

    std::tuple<Tensor, Tensor> rasterize_tensor(
        const lfs::core::Camera& viewpoint_camera,
        const lfs::core::SplatData& gaussian_model,
        const Tensor& bg_color,
        bool show_rings = false,
        float ring_width = 0.01f,
        const Tensor* model_transforms = nullptr,
        const Tensor* transform_indices = nullptr,
        const Tensor* selection_mask = nullptr,
        Tensor* screen_positions_out = nullptr,
        bool brush_active = false,
        float brush_x = 0.0f,
        float brush_y = 0.0f,
        float brush_radius = 0.0f,
        bool brush_add_mode = true,
        Tensor* brush_selection_out = nullptr,
        bool brush_saturation_mode = false,
        float brush_saturation_amount = 0.0f,
        bool selection_mode_rings = false,
        bool show_center_markers = false,
        const Tensor* crop_box_transform = nullptr,
        const Tensor* crop_box_min = nullptr,
        const Tensor* crop_box_max = nullptr,
        bool crop_inverse = false,
        const Tensor* deleted_mask = nullptr,
        unsigned long long* hovered_depth_id = nullptr,
        int highlight_gaussian_id = -1,
        float far_plane = 1e10f);

} // namespace lfs::rendering
