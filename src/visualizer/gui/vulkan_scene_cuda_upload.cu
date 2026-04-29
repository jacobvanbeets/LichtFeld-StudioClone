/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/vulkan_scene_cuda_upload.hpp"

#include <cuda_runtime.h>

namespace lfs::vis::gui {
    namespace {
        __global__ void floatImageToRgba8Kernel(const float* __restrict__ input,
                                                uchar4* __restrict__ output,
                                                const int width,
                                                const int height,
                                                const int channels,
                                                const bool is_chw) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= width || y >= height) {
                return;
            }

            float r = 0.0f;
            float g = 0.0f;
            float b = 0.0f;
            float a = 1.0f;
            if (is_chw) {
                const int hw = width * height;
                const int idx = y * width + x;
                r = input[idx];
                g = input[hw + idx];
                b = input[2 * hw + idx];
                if (channels == 4) {
                    a = input[3 * hw + idx];
                }
            } else {
                const int base = (y * width + x) * channels;
                r = input[base];
                g = input[base + 1];
                b = input[base + 2];
                if (channels == 4) {
                    a = input[base + 3];
                }
            }

            uchar4 pixel{};
            pixel.x = static_cast<unsigned char>(fminf(fmaxf(r, 0.0f), 1.0f) * 255.0f + 0.5f);
            pixel.y = static_cast<unsigned char>(fminf(fmaxf(g, 0.0f), 1.0f) * 255.0f + 0.5f);
            pixel.z = static_cast<unsigned char>(fminf(fmaxf(b, 0.0f), 1.0f) * 255.0f + 0.5f);
            pixel.w = static_cast<unsigned char>(fminf(fmaxf(a, 0.0f), 1.0f) * 255.0f + 0.5f);
            output[y * width + x] = pixel;
        }
    } // namespace

    cudaError_t uploadFloatImageToCudaArray(const float* input,
                                            void* rgba_buffer,
                                            cudaArray_t output,
                                            const int width,
                                            const int height,
                                            const int channels,
                                            const bool is_chw,
                                            const cudaStream_t stream) {
        if (!input || !rgba_buffer || !output || width <= 0 || height <= 0 ||
            (channels != 3 && channels != 4)) {
            return cudaErrorInvalidValue;
        }

        const dim3 block(16, 16);
        const dim3 grid((width + block.x - 1) / block.x,
                        (height + block.y - 1) / block.y);
        auto* const rgba = static_cast<uchar4*>(rgba_buffer);
        floatImageToRgba8Kernel<<<grid, block, 0, stream>>>(
            input, rgba, width, height, channels, is_chw);
        if (const cudaError_t launch_status = cudaGetLastError(); launch_status != cudaSuccess) {
            return launch_status;
        }

        return cudaMemcpy2DToArrayAsync(output,
                                        0,
                                        0,
                                        rgba,
                                        static_cast<size_t>(width) * sizeof(uchar4),
                                        static_cast<size_t>(width) * sizeof(uchar4),
                                        static_cast<size_t>(height),
                                        cudaMemcpyDeviceToDevice,
                                        stream);
    }

} // namespace lfs::vis::gui
