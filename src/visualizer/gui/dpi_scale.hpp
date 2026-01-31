/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::python {
    void  set_shared_dpi_scale(float scale);
    float get_shared_dpi_scale();
} // namespace lfs::python

namespace lfs::vis::gui {

    inline float getDpiScale() { return lfs::python::get_shared_dpi_scale(); }
    inline void setDpiScale(float scale) { lfs::python::set_shared_dpi_scale(scale); }

} // namespace lfs::vis::gui
