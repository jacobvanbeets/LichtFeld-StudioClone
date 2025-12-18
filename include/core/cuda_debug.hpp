/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include <cuda_runtime.h>

namespace lfs::core::debug {

    inline void check_cuda_error(const cudaError_t err, const char* file, const int line, const char* expr) {
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error at {}:{} - {}: {} ({})",
                      file, line, cudaGetErrorName(err), cudaGetErrorString(err), expr);
        }
    }

    inline void check_kernel_sync(const char* file, const int line, const char* kernel_name) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG_ERROR("Kernel '{}' launch failed at {}:{} - {}: {}",
                      kernel_name, file, line, cudaGetErrorName(err), cudaGetErrorString(err));
            return;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG_ERROR("Kernel '{}' execution failed at {}:{} - {}: {}",
                      kernel_name, file, line, cudaGetErrorName(err), cudaGetErrorString(err));
        }
    }

    inline void check_kernel_async(const char* file, const int line, const char* kernel_name) {
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG_ERROR("Kernel '{}' launch failed at {}:{} - {}: {}",
                      kernel_name, file, line, cudaGetErrorName(err), cudaGetErrorString(err));
        }
    }

} // namespace lfs::core::debug

#define CHECK_CUDA(call) \
    lfs::core::debug::check_cuda_error((call), __FILE__, __LINE__, #call)

#define CHECK_CUDA_RETURN(call)                                                  \
    do {                                                                         \
        const cudaError_t _err = (call);                                         \
        if (_err != cudaSuccess) {                                               \
            lfs::core::debug::check_cuda_error(_err, __FILE__, __LINE__, #call); \
            return;                                                              \
        }                                                                        \
    } while (0)

#define CHECK_CUDA_RETURN_VAL(call, val)                                         \
    do {                                                                         \
        const cudaError_t _err = (call);                                         \
        if (_err != cudaSuccess) {                                               \
            lfs::core::debug::check_cuda_error(_err, __FILE__, __LINE__, #call); \
            return (val);                                                        \
        }                                                                        \
    } while (0)

#ifdef CUDA_DEBUG_SYNC
#define CUDA_KERNEL_CHECK(name) \
    lfs::core::debug::check_kernel_sync(__FILE__, __LINE__, name)

#define CUDA_KERNEL_LAUNCH(kernel, grid, block, shared, stream, ...) \
    do {                                                             \
        kernel<<<grid, block, shared, stream>>>(__VA_ARGS__);        \
        CUDA_KERNEL_CHECK(#kernel);                                  \
    } while (0)
#elif defined(DEBUG_BUILD)
#define CUDA_KERNEL_CHECK(name) \
    lfs::core::debug::check_kernel_async(__FILE__, __LINE__, name)

#define CUDA_KERNEL_LAUNCH(kernel, grid, block, shared, stream, ...) \
    do {                                                             \
        kernel<<<grid, block, shared, stream>>>(__VA_ARGS__);        \
        CUDA_KERNEL_CHECK(#kernel);                                  \
    } while (0)
#else
#define CUDA_KERNEL_CHECK(name) ((void)0)

#define CUDA_KERNEL_LAUNCH(kernel, grid, block, shared, stream, ...) \
    kernel<<<grid, block, shared, stream>>>(__VA_ARGS__)
#endif

#define CUDA_KERNEL(kernel, grid, block, ...) \
    CUDA_KERNEL_LAUNCH(kernel, grid, block, 0, 0, __VA_ARGS__)
