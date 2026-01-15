/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <array>
#include <functional>
#include <memory>
#include <optional>

namespace lfs::core {
    class Tensor;
}

namespace lfs::vis {

    struct ViewInfo {
        std::array<float, 9> rotation;
        std::array<float, 3> translation;
        int width;
        int height;
        float fov;
    };

    struct ViewportRender {
        std::shared_ptr<lfs::core::Tensor> image;
        std::shared_ptr<lfs::core::Tensor> screen_positions;
    };

    using GetViewCallback = std::function<std::optional<ViewInfo>()>;
    using GetViewportRenderCallback = std::function<std::optional<ViewportRender>()>;

    void set_view_callback(GetViewCallback callback);
    void set_viewport_render_callback(GetViewportRenderCallback callback);
    std::optional<ViewInfo> get_current_view_info();
    std::optional<ViewportRender> get_viewport_render();

} // namespace lfs::vis
