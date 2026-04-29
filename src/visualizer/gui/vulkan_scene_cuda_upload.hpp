/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::vis::gui {

    cudaError_t uploadFloatImageToCudaArray(const float* input,
                                            void* rgba_buffer,
                                            cudaArray_t output,
                                            int width,
                                            int height,
                                            int channels,
                                            bool is_chw,
                                            cudaStream_t stream);

} // namespace lfs::vis::gui
