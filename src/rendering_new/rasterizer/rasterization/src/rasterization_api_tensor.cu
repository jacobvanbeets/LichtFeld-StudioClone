/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "forward.h"
#include "rasterization_api_tensor.h"
#include "rasterization_config.h"
#include "core_new/cuda/memory_arena.hpp"
#include <functional>
#include <stdexcept>
#include <tuple>

namespace lfs::rendering {

    inline std::function<char*(size_t)> resize_function_wrapper_tensor(Tensor& t) {
        return [&t](size_t N) -> char* {
            if (N == 0) {
                t = Tensor::empty({0}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                return nullptr;
            }
            t = Tensor::empty({N}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
            return reinterpret_cast<char*>(t.ptr<uint8_t>());
        };
    }

    inline void check_tensor_input(bool debug, const Tensor& tensor, const char* name) {
        if (debug) {
            if (!tensor.is_valid() || tensor.device() != lfs::core::Device::CUDA ||
                tensor.dtype() != lfs::core::DataType::Float32 || !tensor.is_contiguous()) {
                throw std::runtime_error("Input tensor '" + std::string(name) +
                                         "' must be a contiguous CUDA float tensor.");
            }
        }
    }

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
        const bool show_rings,
        const float ring_width,
        const Tensor* model_transforms,
        const Tensor* transform_indices,
        const Tensor* selection_mask,
        Tensor* screen_positions_out,
        bool brush_active,
        float brush_x,
        float brush_y,
        float brush_radius,
        bool brush_add_mode,
        Tensor* brush_selection_out,
        bool brush_saturation_mode,
        float brush_saturation_amount,
        const Tensor* crop_box_transform,
        const Tensor* crop_box_min,
        const Tensor* crop_box_max) {

        check_tensor_input(config::debug, means, "means");
        check_tensor_input(config::debug, scales_raw, "scales_raw");
        check_tensor_input(config::debug, rotations_raw, "rotations_raw");
        check_tensor_input(config::debug, opacities_raw, "opacities_raw");
        check_tensor_input(config::debug, sh_coefficients_0, "sh_coefficients_0");
        check_tensor_input(config::debug, sh_coefficients_rest, "sh_coefficients_rest");

        const int n_primitives = static_cast<int>(means.size(0));
        const int total_bases_sh_rest = static_cast<int>(sh_coefficients_rest.size(1));

        Tensor image = Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)},
                                     lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        Tensor alpha = Tensor::empty({1, static_cast<size_t>(height), static_cast<size_t>(width)},
                                     lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        Tensor depth = Tensor::empty({1, static_cast<size_t>(height), static_cast<size_t>(width)},
                                     lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        // Coordinate with training: wait for training, use shared arena
        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
        arena.set_rendering_active(true);
        arena.wait_for_training();
        uint64_t frame_id = arena.begin_frame(true);  // true = from_rendering
        auto arena_allocator = arena.get_allocator(frame_id);

        const std::function<char*(size_t)> per_primitive_buffers_func = arena_allocator;
        const std::function<char*(size_t)> per_tile_buffers_func = arena_allocator;
        const std::function<char*(size_t)> per_instance_buffers_func = arena_allocator;

        Tensor w2c_contig = w2c.is_contiguous() ? w2c : w2c.contiguous();
        Tensor cam_pos_contig = cam_position.is_contiguous() ? cam_position : cam_position.contiguous();

        // Prepare model transforms array pointer
        const float* model_transforms_ptr = nullptr;
        Tensor model_transforms_contig;
        int num_transforms = 0;
        if (model_transforms != nullptr && model_transforms->is_valid() && model_transforms->numel() > 0) {
            model_transforms_contig = model_transforms->is_contiguous() ? *model_transforms : model_transforms->contiguous();
            model_transforms_ptr = model_transforms_contig.ptr<float>();
            // Transforms are stored as [num_transforms, 4, 4] or flat [num_transforms * 16]
            num_transforms = static_cast<int>(model_transforms_contig.numel() / 16);
        }

        // Prepare transform indices pointer
        const int* transform_indices_ptr = nullptr;
        Tensor transform_indices_contig;
        if (transform_indices != nullptr && transform_indices->is_valid() && transform_indices->numel() > 0) {
            transform_indices_contig = transform_indices->is_contiguous() ? *transform_indices : transform_indices->contiguous();
            transform_indices_ptr = transform_indices_contig.ptr<int>();
        }

        // Prepare selection mask pointer
        const uint8_t* selection_mask_ptr = nullptr;
        Tensor selection_mask_contig;
        if (selection_mask != nullptr && selection_mask->is_valid() && selection_mask->numel() > 0) {
            selection_mask_contig = selection_mask->is_contiguous() ? *selection_mask : selection_mask->contiguous();
            selection_mask_ptr = selection_mask_contig.ptr<uint8_t>();
        }

        // Prepare screen positions output buffer if requested
        float2* screen_positions_ptr = nullptr;
        if (screen_positions_out != nullptr) {
            *screen_positions_out = Tensor::empty({static_cast<size_t>(n_primitives), 2},
                                                   lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            screen_positions_ptr = reinterpret_cast<float2*>(screen_positions_out->ptr<float>());
        }

        // Get brush selection buffer pointer (tensor owned by caller)
        bool* brush_selection_ptr = nullptr;
        if (brush_active && brush_selection_out != nullptr && brush_selection_out->is_valid()) {
            brush_selection_ptr = brush_selection_out->ptr<bool>();
        }

        // Prepare crop box parameters
        const float* crop_box_transform_ptr = nullptr;
        const float3* crop_box_min_ptr = nullptr;
        const float3* crop_box_max_ptr = nullptr;
        Tensor crop_box_transform_contig, crop_box_min_contig, crop_box_max_contig;
        if (crop_box_transform != nullptr && crop_box_transform->is_valid() &&
            crop_box_min != nullptr && crop_box_min->is_valid() &&
            crop_box_max != nullptr && crop_box_max->is_valid()) {
            crop_box_transform_contig = crop_box_transform->is_contiguous() ? *crop_box_transform : crop_box_transform->contiguous();
            crop_box_min_contig = crop_box_min->is_contiguous() ? *crop_box_min : crop_box_min->contiguous();
            crop_box_max_contig = crop_box_max->is_contiguous() ? *crop_box_max : crop_box_max->contiguous();
            crop_box_transform_ptr = crop_box_transform_contig.ptr<float>();
            crop_box_min_ptr = reinterpret_cast<const float3*>(crop_box_min_contig.ptr<float>());
            crop_box_max_ptr = reinterpret_cast<const float3*>(crop_box_max_contig.ptr<float>());
        }

        forward(
            per_primitive_buffers_func,
            per_tile_buffers_func,
            per_instance_buffers_func,
            reinterpret_cast<const float3*>(means.ptr<float>()),
            reinterpret_cast<const float3*>(scales_raw.ptr<float>()),
            reinterpret_cast<const float4*>(rotations_raw.ptr<float>()),
            opacities_raw.ptr<float>(),
            reinterpret_cast<const float3*>(sh_coefficients_0.ptr<float>()),
            reinterpret_cast<const float3*>(sh_coefficients_rest.ptr<float>()),
            reinterpret_cast<const float4*>(w2c_contig.ptr<float>()),
            reinterpret_cast<const float3*>(cam_pos_contig.ptr<float>()),
            image.ptr<float>(),
            alpha.ptr<float>(),
            depth.ptr<float>(),
            n_primitives,
            active_sh_bases,
            total_bases_sh_rest,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            near_plane,
            far_plane,
            show_rings,
            ring_width,
            model_transforms_ptr,
            transform_indices_ptr,
            num_transforms,
            selection_mask_ptr,
            screen_positions_ptr,
            brush_active,
            brush_x,
            brush_y,
            brush_radius,
            brush_add_mode,
            brush_selection_ptr,
            brush_saturation_mode,
            brush_saturation_amount,
            crop_box_transform_ptr,
            crop_box_min_ptr,
            crop_box_max_ptr);

        arena.end_frame(frame_id, true);  // true = from_rendering
        arena.set_rendering_active(false);

        return {std::move(image), std::move(alpha), std::move(depth)};
    }

    void brush_select_tensor(
        const Tensor& screen_positions,
        float mouse_x,
        float mouse_y,
        float radius,
        Tensor& selection_out) {

        if (!screen_positions.is_valid() || screen_positions.size(0) == 0) return;

        int n_primitives = static_cast<int>(screen_positions.size(0));

        brush_select(
            reinterpret_cast<const float2*>(screen_positions.ptr<float>()),
            mouse_x,
            mouse_y,
            radius,
            selection_out.ptr<uint8_t>(),
            n_primitives);
    }

} // namespace lfs::rendering
