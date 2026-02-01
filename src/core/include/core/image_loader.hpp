/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <filesystem>
#include <functional>

namespace lfs::core {

    struct ImageLoadParams {
        std::filesystem::path path;
        int resize_factor = 1;
        int max_width = 0;
        void* stream = nullptr;
    };

    using ImageLoadFunc = std::function<Tensor(const ImageLoadParams&)>;

    void set_image_loader(ImageLoadFunc fn);
    Tensor load_image_cached(const ImageLoadParams& params);

} // namespace lfs::core
