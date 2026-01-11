/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "core/property_registry.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace lfs::python {

    class PyOptimizationParams {
    public:
        PyOptimizationParams();

        [[nodiscard]] nb::object get(const std::string& prop_id) const;
        void set(const std::string& prop_id, nb::object value);
        [[nodiscard]] nb::dict prop_info(const std::string& prop_id) const;
        void reset(const std::string& prop_id);
        [[nodiscard]] nb::list properties() const;

        core::param::OptimizationParameters& params();
        [[nodiscard]] const core::param::OptimizationParameters& params() const;
        void refresh();

    private:
        core::param::OptimizationParameters params_;
        bool has_active_trainer_ = false;
    };

    void register_optimization_properties();
    void register_params(nb::module_& m);

} // namespace lfs::python
