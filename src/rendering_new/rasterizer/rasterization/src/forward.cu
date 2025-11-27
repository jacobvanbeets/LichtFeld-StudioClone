/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "buffer_utils.h"
#include "forward.h"
#include "helper_math.h"
#include "kernels_forward.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <functional>

// Initialize mean2d buffer with invalid marker values
__global__ void init_mean2d_kernel(float2* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = make_float2(-10000.0f, -10000.0f);
    }
}

// Copy mean2d to screen positions, flipping Y to match window coordinates
// The rasterizer's mean2d has Y increasing upward (OpenGL convention),
// but window coordinates have Y increasing downward
__global__ void copy_screen_positions_kernel(
    const float2* __restrict__ mean2d,
    float2* __restrict__ screen_positions_out,
    float height,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float2 pos = mean2d[idx];
    // Flip Y: window_y = height - rasterizer_y
    // Keep invalid markers as-is (they have large negative values)
    if (pos.y > -1000.0f) {
        pos.y = height - pos.y;
    }
    screen_positions_out[idx] = pos;
}

// Simple kernel to select Gaussians within brush radius
__global__ void brush_select_kernel(
    const float2* __restrict__ screen_positions,
    float mouse_x,
    float mouse_y,
    float radius_sq,
    uint8_t* __restrict__ selection_out,
    int n_primitives) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_primitives) return;

    float2 pos = screen_positions[idx];

    // Skip invalid/off-screen positions (marked with large negative values)
    if (pos.x < -1000.0f || pos.y < -1000.0f) return;

    float dx = pos.x - mouse_x;
    float dy = pos.y - mouse_y;
    float dist_sq = dx * dx + dy * dy;

    if (dist_sq <= radius_sq) {
        selection_out[idx] = 1;
    }
}

void lfs::rendering::brush_select(
    const float2* screen_positions,
    float mouse_x,
    float mouse_y,
    float radius,
    uint8_t* selection_out,
    int n_primitives) {

    if (n_primitives <= 0) return;

    constexpr int block_size = 256;
    int grid_size = (n_primitives + block_size - 1) / block_size;

    brush_select_kernel<<<grid_size, block_size>>>(
        screen_positions,
        mouse_x,
        mouse_y,
        radius * radius,  // Pass squared radius to avoid sqrt in kernel
        selection_out,
        n_primitives);
}

// sorting is done separately for depth and tile as proposed in https://github.com/m-schuetz/Splatshop
void lfs::rendering::forward(
    std::function<char*(size_t)> per_primitive_buffers_func,
    std::function<char*(size_t)> per_tile_buffers_func,
    std::function<char*(size_t)> per_instance_buffers_func,
    const float3* means,
    const float3* scales_raw,
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
    const float near_, // near and far are macros in windowns
    const float far_,
    const bool show_rings,
    const float ring_width,
    const float* model_transforms,
    const int* transform_indices,
    const int num_transforms,
    const uint8_t* selection_mask,
    float2* screen_positions_out,
    bool brush_active,
    float brush_x,
    float brush_y,
    float brush_radius,
    bool brush_add_mode,
    bool* brush_selection_out,
    bool brush_saturation_mode,
    float brush_saturation_amount,
    const float* crop_box_transform,
    const float3* crop_box_min,
    const float3* crop_box_max) {

    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;

    char* per_tile_buffers_blob = per_tile_buffers_func(required<PerTileBuffers>(n_tiles));
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);

    static cudaStream_t memset_stream = 0;
    if constexpr (!config::debug) {
        static bool memset_stream_initialized = false;
        if (!memset_stream_initialized) {
            cudaStreamCreate(&memset_stream);
            memset_stream_initialized = true;
        }
        cudaMemsetAsync(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, memset_stream);
    } else
        cudaMemset(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles);

    char* per_primitive_buffers_blob = per_primitive_buffers_func(required<PerPrimitiveBuffers>(n_primitives));
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);

    cudaMemset(per_primitive_buffers.n_visible_primitives, 0, sizeof(uint));
    cudaMemset(per_primitive_buffers.n_instances, 0, sizeof(uint));

    // Initialize mean2d with invalid marker values for brush selection
    // Only visible Gaussians will have their mean2d updated by preprocess kernel
    if (screen_positions_out != nullptr) {
        constexpr int init_block = 256;
        int init_grid = (n_primitives + init_block - 1) / init_block;
        init_mean2d_kernel<<<init_grid, init_block>>>(per_primitive_buffers.mean2d, n_primitives);
    }

    kernels::forward::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess>>>(
        means,
        scales_raw,
        rotations_raw,
        opacities_raw,
        sh_coefficients_0,
        sh_coefficients_rest,
        w2c,
        cam_position,
        per_primitive_buffers.depth_keys.Current(),
        per_primitive_buffers.primitive_indices.Current(),
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.screen_bounds,
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_primitive_buffers.color,
        per_primitive_buffers.depth,
        per_primitive_buffers.n_visible_primitives,
        per_primitive_buffers.n_instances,
        n_primitives,
        grid.x,
        grid.y,
        active_sh_bases,
        total_bases_sh_rest,
        static_cast<float>(width),
        static_cast<float>(height),
        fx,
        fy,
        cx,
        cy,
        near_,
        far_,
        model_transforms,
        transform_indices,
        num_transforms,
        selection_mask,
        brush_active,
        brush_x,
        brush_y,
        brush_radius * brush_radius,  // Pass squared radius for efficient comparison
        brush_add_mode,
        brush_selection_out,
        brush_saturation_mode,
        brush_saturation_amount,
        crop_box_transform,
        crop_box_min,
        crop_box_max);
    CHECK_CUDA(config::debug, "preprocess")


    // Copy screen positions if requested (for brush tool selection)
    if (screen_positions_out != nullptr) {
        cudaMemcpy(screen_positions_out, per_primitive_buffers.mean2d,
                   sizeof(float2) * n_primitives, cudaMemcpyDeviceToDevice);
    }

    int n_visible_primitives;
    cudaMemcpy(&n_visible_primitives, per_primitive_buffers.n_visible_primitives, sizeof(uint), cudaMemcpyDeviceToHost);
    int n_instances;
    cudaMemcpy(&n_instances, per_primitive_buffers.n_instances, sizeof(uint), cudaMemcpyDeviceToHost);

    cub::DeviceRadixSort::SortPairs(
        per_primitive_buffers.cub_workspace,
        per_primitive_buffers.cub_workspace_size,
        per_primitive_buffers.depth_keys,
        per_primitive_buffers.primitive_indices,
        n_visible_primitives);
    CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs (Depth)")

    kernels::forward::apply_depth_ordering_cu<<<div_round_up(n_visible_primitives, config::block_size_apply_depth_ordering), config::block_size_apply_depth_ordering>>>(
        per_primitive_buffers.primitive_indices.Current(),
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.offset,
        n_visible_primitives);
    CHECK_CUDA(config::debug, "apply_depth_ordering")

    cub::DeviceScan::ExclusiveSum(
        per_primitive_buffers.cub_workspace,
        per_primitive_buffers.cub_workspace_size,
        per_primitive_buffers.offset,
        per_primitive_buffers.offset,
        n_visible_primitives);
    CHECK_CUDA(config::debug, "cub::DeviceScan::ExclusiveSum (Primitive Offsets)")

    char* per_instance_buffers_blob = per_instance_buffers_func(required<PerInstanceBuffers>(n_instances));
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, n_instances);

    kernels::forward::create_instances_cu<<<div_round_up(n_visible_primitives, config::block_size_create_instances), config::block_size_create_instances>>>(
        per_primitive_buffers.primitive_indices.Current(),
        per_primitive_buffers.offset,
        per_primitive_buffers.screen_bounds,
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_instance_buffers.keys.Current(),
        per_instance_buffers.primitive_indices.Current(),
        grid.x,
        n_visible_primitives);
    CHECK_CUDA(config::debug, "create_instances")

    cub::DeviceRadixSort::SortPairs(
        per_instance_buffers.cub_workspace,
        per_instance_buffers.cub_workspace_size,
        per_instance_buffers.keys,
        per_instance_buffers.primitive_indices,
        n_instances);
    CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs (Tile)")

    if constexpr (!config::debug)
        cudaStreamSynchronize(memset_stream);

    if (n_instances > 0) {
        kernels::forward::extract_instance_ranges_cu<<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges>>>(
            per_instance_buffers.keys.Current(),
            per_tile_buffers.instance_ranges,
            n_instances);
        CHECK_CUDA(config::debug, "extract_instance_ranges")
    }

    kernels::forward::blend_cu<<<grid, block>>>(
        per_tile_buffers.instance_ranges,
        per_instance_buffers.primitive_indices.Current(),
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_primitive_buffers.color,
        per_primitive_buffers.depth,
        image,
        alpha,
        depth,
        width,
        height,
        grid.x,
        show_rings,
        ring_width);
    CHECK_CUDA(config::debug, "blend")
}