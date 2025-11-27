/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/tensor.hpp"
#include <tuple>

namespace lfs::rendering {

    // Import Tensor from lfs::core
    using lfs::core::Tensor;

    /**
     * @brief Forward rasterization with custom Tensor types (libtorch-free)
     *
     * @param means Gaussian means [N, 3]
     * @param scales_raw Gaussian scales in log-space [N, 3]
     * @param rotations_raw Gaussian rotations (unnormalized quaternions) [N, 4]
     * @param opacities_raw Gaussian opacities in logit-space [N, 1]
     * @param sh_coefficients_0 SH coefficients degree 0 [N, 1, 3]
     * @param sh_coefficients_rest SH coefficients degree 1+ [N, (degree+1)²-1, 3]
     * @param w2c World-to-camera transform matrix [4, 4]
     * @param cam_position Camera position in world space [3]
     * @param active_sh_bases Number of active SH bases (degree+1)²
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param focal_x Focal length in x direction
     * @param focal_y Focal length in y direction
     * @param center_x Principal point x coordinate
     * @param center_y Principal point y coordinate
     * @param near_plane Near clipping plane
     * @param far_plane Far clipping plane
     * @param show_rings Enable ring mode visualization
     * @param ring_width Width of the ring band
     * @param model_transforms Array of 4x4 transforms [num_transforms, 4, 4]
     * @param transform_indices Per-Gaussian transform index [N]
     * @param selection_mask Per-Gaussian selection mask [N] (uint8, 1=selected/yellow)
     * @param screen_positions_out Optional output: screen positions [N, 2] for brush tool
     *
     * @return Tuple of (rendered_image [3, H, W], alpha_map [1, H, W], depth_map [1, H, W])
     */
    std::tuple<Tensor, Tensor, Tensor>
    forward_wrapper_tensor(
        const Tensor& means,
        const Tensor& scales_raw,
        const Tensor& rotations_raw,
        const Tensor& opacities_raw,
        const Tensor& sh_coefficients_0,
        const Tensor& sh_coefficients_rest,
        const Tensor& w2c,
        const Tensor& cam_position,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool show_rings = false,
        const float ring_width = 0.002f,
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
        const Tensor* crop_box_transform = nullptr,
        const Tensor* crop_box_min = nullptr,
        const Tensor* crop_box_max = nullptr);

    /**
     * @brief Select Gaussians within brush radius using GPU
     *
     * @param screen_positions Screen positions [N, 2] from render
     * @param mouse_x Mouse X in image coords
     * @param mouse_y Mouse Y in image coords
     * @param radius Brush radius in pixels
     * @param selection_out Output selection mask [N] uint8
     */
    void brush_select_tensor(
        const Tensor& screen_positions,
        float mouse_x,
        float mouse_y,
        float radius,
        Tensor& selection_out);

} // namespace lfs::rendering
