/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "py_tensor.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

namespace nb = nanobind;

namespace lfs::vis {
    class Scene;
}

namespace lfs::python {

    struct PyViewInfo {
        PyTensor rotation;
        PyTensor translation;
        int width;
        int height;
        float fov_x;
        float fov_y;
    };

    struct PyViewportRender {
        PyTensor image;
        std::optional<PyTensor> screen_positions;
    };

    [[nodiscard]] std::optional<PyViewportRender> get_viewport_render();

    [[nodiscard]] std::optional<PyViewportRender> capture_viewport();

    [[nodiscard]] std::optional<PyTensor> render_view(const PyTensor& rotation, const PyTensor& translation, int width,
                                                      int height, float fov_degrees = 60.0f,
                                                      const PyTensor* bg_color = nullptr);

    [[nodiscard]] std::optional<PyTensor> compute_screen_positions(const PyTensor& rotation, const PyTensor& translation,
                                                                   int width, int height, float fov_degrees = 60.0f);

    [[nodiscard]] std::optional<PyViewInfo> get_current_view();

    void register_rendering(nb::module_& m);

    void set_render_scene_context(vis::Scene* scene);
    [[nodiscard]] vis::Scene* get_render_scene();

} // namespace lfs::python
