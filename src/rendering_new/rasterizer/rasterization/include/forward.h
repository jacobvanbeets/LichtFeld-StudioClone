/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"
#include <cstdint>
#include <functional>

namespace lfs::rendering {

    // Brush selection: mark Gaussians within radius of mouse position
    // screen_positions: [N, 2] from previous render
    // mouse_x, mouse_y: mouse position in image coords (0 to width, 0 to height)
    // radius: brush radius in pixels
    // selection_out: [N] uint8_t, 1 = selected
    void brush_select(
        const float2* screen_positions,
        float mouse_x,
        float mouse_y,
        float radius,
        uint8_t* selection_out,
        int n_primitives);

    void forward(
        std::function<char*(size_t)> per_primitive_buffers_func,
        std::function<char*(size_t)> per_tile_buffers_func,
        std::function<char*(size_t)> per_instance_buffers_func,
        const float3* means, const float3* scales_raw,
        const float4* rotations_raw,
        const float* opacities_raw,
        const float3* sh_coefficients_0,
        const float3* sh_coefficients_rest,
        const float4* w2c,
        const float3* cam_position,
        float* image,
        float* alpha,
        float* depth,
        const int n_primitives,
        const int active_sh_bases,
        const int total_bases_sh_rest,
        const int width,
        const int height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float near,
        const float far,
        const bool show_rings = false,
        const float ring_width = 0.002f,
        const float* model_transforms = nullptr,    // Array of 4x4 transforms (row-major), one per node
        const int* transform_indices = nullptr,     // Per-Gaussian index into transforms array [N]
        const int num_transforms = 0,               // Number of transforms in array
        const uint8_t* selection_mask = nullptr,    // Per-Gaussian selection mask [N], 1=selected (yellow)
        float2* screen_positions_out = nullptr,
        bool brush_active = false,
        float brush_x = 0.0f,
        float brush_y = 0.0f,
        float brush_radius = 0.0f,
        bool brush_add_mode = true,
        bool* brush_selection_out = nullptr,
        bool brush_saturation_mode = false,
        float brush_saturation_amount = 0.0f,
        // Crop box visualization (highlight outside gaussians)
        const float* crop_box_transform = nullptr,  // 4x4 world-to-box transform (row-major)
        const float3* crop_box_min = nullptr,       // Box min bounds in local space
        const float3* crop_box_max = nullptr);      // Box max bounds in local space

}