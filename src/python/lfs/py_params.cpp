/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_params.hpp"

#include "control/command_api.hpp"
#include "core/logger.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace lfs::python {

    using namespace lfs::core::param;
    using namespace lfs::core::prop;
    using lfs::training::CommandCenter;

    void register_optimization_properties() {
        PropertyGroupBuilder<OptimizationParameters>("optimization", "Optimization")
            // Training control
            .size_prop(&OptimizationParameters::iterations,
                       "iterations", "Max Iterations", 30000, 1, 1000000,
                       "Maximum number of training iterations")
            .int_prop(&OptimizationParameters::sh_degree,
                      "sh_degree", "SH Degree", 3, 0, 3,
                      "Spherical harmonics degree (0-3)")
            .size_prop(&OptimizationParameters::sh_degree_interval,
                       "sh_degree_interval", "SH Interval", 1000, 100, 10000,
                       "Iterations between SH degree increases")
            .int_prop(&OptimizationParameters::max_cap,
                      "max_cap", "Max Gaussians", 1000000, 1000, 10000000,
                      "Maximum number of gaussians")

            // Learning rates
            .float_prop(&OptimizationParameters::means_lr,
                        "means_lr", "Position LR", 0.000016f, 0.0f, 0.001f,
                        "Learning rate for gaussian positions")
            .flags(PROP_LIVE_UPDATE)
            .category("learning_rates")
            .float_prop(&OptimizationParameters::shs_lr,
                        "shs_lr", "SH LR", 0.0025f, 0.0f, 0.1f,
                        "Learning rate for spherical harmonics")
            .flags(PROP_LIVE_UPDATE)
            .category("learning_rates")
            .float_prop(&OptimizationParameters::opacity_lr,
                        "opacity_lr", "Opacity LR", 0.05f, 0.0f, 1.0f,
                        "Learning rate for opacity")
            .flags(PROP_LIVE_UPDATE)
            .category("learning_rates")
            .float_prop(&OptimizationParameters::scaling_lr,
                        "scaling_lr", "Scale LR", 0.005f, 0.0f, 0.1f,
                        "Learning rate for gaussian scales")
            .flags(PROP_LIVE_UPDATE)
            .category("learning_rates")
            .float_prop(&OptimizationParameters::rotation_lr,
                        "rotation_lr", "Rotation LR", 0.001f, 0.0f, 0.1f,
                        "Learning rate for rotations")
            .flags(PROP_LIVE_UPDATE)
            .category("learning_rates")

            // Loss parameters
            .float_prop(&OptimizationParameters::lambda_dssim,
                        "lambda_dssim", "DSSIM Weight", 0.2f, 0.0f, 1.0f,
                        "Weight for structural similarity loss")
            .category("loss")
            .float_prop(&OptimizationParameters::opacity_reg,
                        "opacity_reg", "Opacity Reg", 0.01f, 0.0f, 1.0f,
                        "Opacity regularization weight")
            .category("loss")
            .float_prop(&OptimizationParameters::scale_reg,
                        "scale_reg", "Scale Reg", 0.01f, 0.0f, 1.0f,
                        "Scale regularization weight")
            .category("loss")

            // Refinement
            .size_prop(&OptimizationParameters::refine_every,
                       "refine_every", "Refine Every", 100, 1, 1000,
                       "Interval for adaptive density control")
            .category("refinement")
            .size_prop(&OptimizationParameters::start_refine,
                       "start_refine", "Start Refine", 500, 0, 10000,
                       "Iteration to start refinement")
            .category("refinement")
            .size_prop(&OptimizationParameters::stop_refine,
                       "stop_refine", "Stop Refine", 25000, 0, 100000,
                       "Iteration to stop refinement")
            .category("refinement")
            .float_prop(&OptimizationParameters::grad_threshold,
                        "grad_threshold", "Grad Threshold", 0.0002f, 0.0f, 0.01f,
                        "Gradient threshold for densification")
            .category("refinement")
            .float_prop(&OptimizationParameters::min_opacity,
                        "min_opacity", "Min Opacity", 0.005f, 0.0f, 0.1f,
                        "Minimum opacity for pruning")
            .category("refinement")
            .float_prop(&OptimizationParameters::init_opacity,
                        "init_opacity", "Init Opacity", 0.5f, 0.0f, 1.0f,
                        "Initial opacity for new gaussians")
            .category("refinement")
            .float_prop(&OptimizationParameters::init_scaling,
                        "init_scaling", "Init Scale", 0.1f, 0.0f, 1.0f,
                        "Initial scale for new gaussians")
            .category("refinement")

            // Mask parameters
            .enum_prop(&OptimizationParameters::mask_mode,
                       "mask_mode", "Mask Mode", MaskMode::None,
                       {{"None", MaskMode::None},
                        {"Segment", MaskMode::Segment},
                        {"Ignore", MaskMode::Ignore},
                        {"AlphaConsistent", MaskMode::AlphaConsistent}},
                       "Attention mask behavior during training")
            .category("mask")
            .bool_prop(&OptimizationParameters::invert_masks,
                       "invert_masks", "Invert Masks", false,
                       "Swap object and background in masks")
            .category("mask")
            .float_prop(&OptimizationParameters::mask_threshold,
                        "mask_threshold", "Mask Threshold", 0.5f, 0.0f, 1.0f,
                        "Threshold for mask binarization")
            .category("mask")
            .float_prop(&OptimizationParameters::mask_opacity_penalty_weight,
                        "mask_opacity_penalty_weight", "Penalty Weight", 1.0f, 0.0f, 10.0f,
                        "Opacity penalty weight for segment mode")
            .category("mask")

            // Bilateral grid
            .bool_prop(&OptimizationParameters::use_bilateral_grid,
                       "use_bilateral_grid", "Bilateral Grid", false,
                       "Enable bilateral grid color correction")
            .flags(PROP_NEEDS_RESTART)
            .category("bilateral_grid")
            .int_prop(&OptimizationParameters::bilateral_grid_X,
                      "bilateral_grid_x", "Grid X", 16, 4, 64,
                      "Bilateral grid X resolution")
            .category("bilateral_grid")
            .int_prop(&OptimizationParameters::bilateral_grid_Y,
                      "bilateral_grid_y", "Grid Y", 16, 4, 64,
                      "Bilateral grid Y resolution")
            .category("bilateral_grid")
            .int_prop(&OptimizationParameters::bilateral_grid_W,
                      "bilateral_grid_w", "Grid W", 8, 2, 32,
                      "Bilateral grid intensity bins")
            .category("bilateral_grid")
            .float_prop(&OptimizationParameters::bilateral_grid_lr,
                        "bilateral_grid_lr", "Grid LR", 0.002f, 0.0f, 0.1f,
                        "Bilateral grid learning rate")
            .category("bilateral_grid")
            .float_prop(&OptimizationParameters::tv_loss_weight,
                        "tv_loss_weight", "TV Loss Weight", 10.0f, 0.0f, 100.0f,
                        "Total variation loss weight")
            .category("bilateral_grid")

            // Strategy
            .string_prop(&OptimizationParameters::strategy,
                         "strategy", "Strategy", "mcmc",
                         "Optimization strategy: mcmc or adc")
            .flags(PROP_NEEDS_RESTART)

            // ADC strategy parameters
            .float_prop(&OptimizationParameters::prune_opacity,
                        "prune_opacity", "Prune Opacity", 0.005f, 0.0f, 0.1f,
                        "Opacity threshold for pruning (ADC)")
            .category("adc")
            .float_prop(&OptimizationParameters::grow_scale3d,
                        "grow_scale3d", "Grow Scale 3D", 0.01f, 0.0f, 0.1f,
                        "3D scale threshold for growing (ADC)")
            .category("adc")
            .float_prop(&OptimizationParameters::grow_scale2d,
                        "grow_scale2d", "Grow Scale 2D", 0.05f, 0.0f, 0.2f,
                        "2D scale threshold for growing (ADC)")
            .category("adc")
            .size_prop(&OptimizationParameters::reset_every,
                       "reset_every", "Reset Every", 3000, 100, 10000,
                       "Iteration interval for opacity reset (ADC)")
            .category("adc")

            // Flags
            .bool_prop(&OptimizationParameters::mip_filter,
                       "mip_filter", "Mip Filter", false,
                       "Enable mip filtering (anti-aliasing)")
            .bool_prop(&OptimizationParameters::bg_modulation,
                       "bg_modulation", "BG Modulation", false,
                       "Enable sinusoidal background modulation")
            .bool_prop(&OptimizationParameters::headless,
                       "headless", "Headless", false,
                       "Run without visualization")
            .flags(PROP_READONLY)
            .bool_prop(&OptimizationParameters::enable_eval,
                       "enable_eval", "Enable Eval", false,
                       "Run evaluation at specified steps")

            // Random initialization
            .bool_prop(&OptimizationParameters::random,
                       "random", "Random Init", false,
                       "Use random initialization instead of SfM")
            .flags(PROP_NEEDS_RESTART)
            .category("random_init")
            .int_prop(&OptimizationParameters::init_num_pts,
                      "init_num_pts", "Init Points", 100000, 1000, 1000000,
                      "Number of random points to initialize")
            .category("random_init")
            .float_prop(&OptimizationParameters::init_extent,
                        "init_extent", "Init Extent", 3.0f, 0.1f, 10.0f,
                        "Extent of random point cloud")
            .category("random_init")

            // Sparsity
            .bool_prop(&OptimizationParameters::enable_sparsity,
                       "enable_sparsity", "Enable Sparsity", false,
                       "Enable sparsity optimization")
            .category("sparsity")
            .int_prop(&OptimizationParameters::sparsify_steps,
                      "sparsify_steps", "Sparsify Steps", 15000, 1000, 50000,
                      "Iteration to run sparsification")
            .category("sparsity")
            .float_prop(&OptimizationParameters::prune_ratio,
                        "prune_ratio", "Prune Ratio", 0.6f, 0.0f, 1.0f,
                        "Target pruning ratio for sparsification")
            .category("sparsity")

            .build();
    }

    PyOptimizationParams::PyOptimizationParams() {
        refresh();
    }

    void PyOptimizationParams::refresh() {
        auto snap = CommandCenter::instance().snapshot();
        if (snap.trainer) {
            params_ = snap.trainer->getParams().optimization;
            has_active_trainer_ = true;
        } else {
            params_ = OptimizationParameters{};
            has_active_trainer_ = false;
        }
    }

    core::param::OptimizationParameters& PyOptimizationParams::params() {
        return params_;
    }

    const core::param::OptimizationParameters& PyOptimizationParams::params() const {
        return params_;
    }

    nb::object PyOptimizationParams::get(const std::string& prop_id) const {
        auto* meta = PropertyRegistry::instance().get_property("optimization", prop_id);
        if (!meta) {
            throw std::runtime_error("Unknown property: " + prop_id);
        }

        std::any value = meta->getter(&params_);

        switch (meta->type) {
        case PropType::Bool:
            return nb::cast(std::any_cast<bool>(value));
        case PropType::Int:
            return nb::cast(std::any_cast<int>(value));
        case PropType::Float:
            return nb::cast(std::any_cast<float>(value));
        case PropType::String:
            return nb::cast(std::any_cast<std::string>(value));
        case PropType::SizeT:
            return nb::cast(std::any_cast<size_t>(value));
        case PropType::Enum:
            return nb::cast(std::any_cast<int>(value));
        default:
            throw std::runtime_error("Unsupported property type");
        }
    }

    void PyOptimizationParams::set(const std::string& prop_id, nb::object value) {
        auto* meta = PropertyRegistry::instance().get_property("optimization", prop_id);
        if (!meta) {
            throw std::runtime_error("Unknown property: " + prop_id);
        }

        if (meta->is_readonly()) {
            throw std::runtime_error("Property is read-only: " + prop_id);
        }

        std::any old_value = meta->getter(&params_);
        std::any new_value;

        switch (meta->type) {
        case PropType::Bool:
            new_value = nb::cast<bool>(value);
            break;
        case PropType::Int:
            new_value = nb::cast<int>(value);
            break;
        case PropType::Float:
            new_value = nb::cast<float>(value);
            break;
        case PropType::String:
            new_value = nb::cast<std::string>(value);
            break;
        case PropType::SizeT:
            new_value = static_cast<size_t>(nb::cast<int64_t>(value));
            break;
        case PropType::Enum:
            new_value = nb::cast<int>(value);
            break;
        default:
            throw std::runtime_error("Unsupported property type");
        }

        meta->setter(&params_, new_value);
        PropertyRegistry::instance().notify("optimization", prop_id, old_value, new_value);
    }

    nb::dict PyOptimizationParams::prop_info(const std::string& prop_id) const {
        auto* meta = PropertyRegistry::instance().get_property("optimization", prop_id);
        if (!meta) {
            throw std::runtime_error("Unknown property: " + prop_id);
        }

        nb::dict info;
        info["id"] = meta->id;
        info["name"] = meta->name;
        info["description"] = meta->description;
        info["group"] = meta->group;
        info["readonly"] = meta->is_readonly();
        info["live_update"] = meta->is_live_update();
        info["needs_restart"] = meta->needs_restart();

        switch (meta->type) {
        case PropType::Float:
            info["type"] = "float";
            info["min"] = meta->min_value;
            info["max"] = meta->max_value;
            info["default"] = meta->default_value;
            break;
        case PropType::Int:
            info["type"] = "int";
            info["min"] = static_cast<int>(meta->min_value);
            info["max"] = static_cast<int>(meta->max_value);
            info["default"] = static_cast<int>(meta->default_value);
            break;
        case PropType::SizeT:
            info["type"] = "int";
            info["min"] = static_cast<int64_t>(meta->min_value);
            info["max"] = static_cast<int64_t>(meta->max_value);
            info["default"] = static_cast<int64_t>(meta->default_value);
            break;
        case PropType::Bool:
            info["type"] = "bool";
            info["default"] = meta->default_value > 0.5;
            break;
        case PropType::String:
            info["type"] = "string";
            info["default"] = meta->default_string;
            break;
        case PropType::Enum:
            info["type"] = "enum";
            info["default"] = meta->default_enum;
            {
                nb::list items;
                for (const auto& ei : meta->enum_items) {
                    nb::dict item;
                    item["name"] = ei.name;
                    item["value"] = ei.value;
                    items.append(item);
                }
                info["items"] = items;
            }
            break;
        }

        return info;
    }

    void PyOptimizationParams::reset(const std::string& prop_id) {
        auto* meta = PropertyRegistry::instance().get_property("optimization", prop_id);
        if (!meta) {
            throw std::runtime_error("Unknown property: " + prop_id);
        }

        std::any default_val;
        switch (meta->type) {
        case PropType::Float:
            default_val = static_cast<float>(meta->default_value);
            break;
        case PropType::Int:
            default_val = static_cast<int>(meta->default_value);
            break;
        case PropType::SizeT:
            default_val = static_cast<size_t>(meta->default_value);
            break;
        case PropType::Bool:
            default_val = meta->default_value > 0.5;
            break;
        case PropType::String:
            default_val = meta->default_string;
            break;
        case PropType::Enum:
            default_val = meta->default_enum;
            break;
        }

        std::any old_value = meta->getter(&params_);
        meta->setter(&params_, default_val);
        PropertyRegistry::instance().notify("optimization", prop_id, old_value, default_val);
    }

    nb::list PyOptimizationParams::properties() const {
        auto* group = PropertyRegistry::instance().get_group("optimization");
        if (!group) {
            return nb::list();
        }

        nb::list result;
        for (const auto& prop : group->properties) {
            nb::dict item;
            item["id"] = prop.id;
            item["name"] = prop.name;
            item["group"] = prop.group;
            item["value"] = get(prop.id);
            result.append(item);
        }
        return result;
    }

    void register_params(nb::module_& m) {
        register_optimization_properties();

        // MaskMode enum
        nb::enum_<MaskMode>(m, "MaskMode")
            .value("NONE", MaskMode::None)
            .value("SEGMENT", MaskMode::Segment)
            .value("IGNORE", MaskMode::Ignore)
            .value("ALPHA_CONSISTENT", MaskMode::AlphaConsistent);

        // PyOptimizationParams class
        nb::class_<PyOptimizationParams>(m, "OptimizationParams")
            .def(nb::init<>())
            .def("__getattr__", &PyOptimizationParams::get, nb::arg("name"))
            .def("__setattr__", &PyOptimizationParams::set, nb::arg("name"), nb::arg("value"))
            .def("prop_info", &PyOptimizationParams::prop_info, nb::arg("prop_id"),
                 "Get metadata for a property")
            .def("reset", &PyOptimizationParams::reset, nb::arg("prop_id"),
                 "Reset property to default value")
            .def("properties", &PyOptimizationParams::properties,
                 "List all properties with their current values")
            .def("refresh", &PyOptimizationParams::refresh,
                 "Refresh from current training state")

            // Direct property access for common properties (IDE autocomplete)
            .def_prop_rw(
                "iterations",
                [](PyOptimizationParams& self) { return self.params().iterations; },
                [](PyOptimizationParams& self, size_t v) { self.params().iterations = v; })
            .def_prop_rw(
                "means_lr",
                [](PyOptimizationParams& self) { return self.params().means_lr; },
                [](PyOptimizationParams& self, float v) { self.params().means_lr = v; })
            .def_prop_rw(
                "shs_lr",
                [](PyOptimizationParams& self) { return self.params().shs_lr; },
                [](PyOptimizationParams& self, float v) { self.params().shs_lr = v; })
            .def_prop_rw(
                "opacity_lr",
                [](PyOptimizationParams& self) { return self.params().opacity_lr; },
                [](PyOptimizationParams& self, float v) { self.params().opacity_lr = v; })
            .def_prop_rw(
                "scaling_lr",
                [](PyOptimizationParams& self) { return self.params().scaling_lr; },
                [](PyOptimizationParams& self, float v) { self.params().scaling_lr = v; })
            .def_prop_rw(
                "rotation_lr",
                [](PyOptimizationParams& self) { return self.params().rotation_lr; },
                [](PyOptimizationParams& self, float v) { self.params().rotation_lr = v; })
            .def_prop_rw(
                "lambda_dssim",
                [](PyOptimizationParams& self) { return self.params().lambda_dssim; },
                [](PyOptimizationParams& self, float v) { self.params().lambda_dssim = v; })
            .def_prop_rw(
                "sh_degree",
                [](PyOptimizationParams& self) { return self.params().sh_degree; },
                [](PyOptimizationParams& self, int v) { self.params().sh_degree = v; })
            .def_prop_rw(
                "max_cap",
                [](PyOptimizationParams& self) { return self.params().max_cap; },
                [](PyOptimizationParams& self, int v) { self.params().max_cap = v; })
            .def_prop_ro("strategy", [](PyOptimizationParams& self) { return self.params().strategy; })
            .def_prop_ro("headless", [](PyOptimizationParams& self) { return self.params().headless; });

        // Factory function
        m.def(
            "optimization_params", []() { return PyOptimizationParams{}; },
            "Get current optimization parameters");

        // Property change callback registration
        m.def(
            "on_property_change",
            [](const std::string& property_path, nb::callable callback) {
                // Parse property_path like "optimization.means_lr"
                auto dot_pos = property_path.find('.');
                if (dot_pos == std::string::npos) {
                    throw std::runtime_error("Invalid property path. Use 'group.property' format");
                }
                std::string group_id = property_path.substr(0, dot_pos);
                std::string prop_id = property_path.substr(dot_pos + 1);

                // Wrap Python callback
                nb::object cb_obj = nb::cast<nb::object>(callback);
                auto cpp_callback = [cb_obj](const std::string& /*group*/,
                                             const std::string& /*prop*/,
                                             const std::any& old_val,
                                             const std::any& new_val) {
                    nb::gil_scoped_acquire gil;
                    try {
                        // Convert std::any to Python objects
                        nb::object py_old, py_new;
                        if (old_val.type() == typeid(float)) {
                            py_old = nb::cast(std::any_cast<float>(old_val));
                            py_new = nb::cast(std::any_cast<float>(new_val));
                        } else if (old_val.type() == typeid(int)) {
                            py_old = nb::cast(std::any_cast<int>(old_val));
                            py_new = nb::cast(std::any_cast<int>(new_val));
                        } else if (old_val.type() == typeid(bool)) {
                            py_old = nb::cast(std::any_cast<bool>(old_val));
                            py_new = nb::cast(std::any_cast<bool>(new_val));
                        } else if (old_val.type() == typeid(size_t)) {
                            py_old = nb::cast(std::any_cast<size_t>(old_val));
                            py_new = nb::cast(std::any_cast<size_t>(new_val));
                        } else if (old_val.type() == typeid(std::string)) {
                            py_old = nb::cast(std::any_cast<std::string>(old_val));
                            py_new = nb::cast(std::any_cast<std::string>(new_val));
                        } else {
                            py_old = nb::none();
                            py_new = nb::none();
                        }
                        cb_obj(py_old, py_new);
                    } catch (const std::exception& e) {
                        LOG_ERROR("Property change callback error: {}", e.what());
                    }
                };

                size_t sub_id = PropertyRegistry::instance().subscribe(group_id, prop_id, cpp_callback);
                return sub_id;
            },
            nb::arg("property_path"), nb::arg("callback"),
            "Register a callback for property changes. Returns subscription ID.\n"
            "Usage: lf.on_property_change('optimization.means_lr', lambda old, new: print(f'{old} -> {new}'))");

        m.def(
            "unsubscribe_property_change",
            [](size_t subscription_id) {
                PropertyRegistry::instance().unsubscribe(subscription_id);
            },
            nb::arg("subscription_id"),
            "Unsubscribe from property change notifications");

        // Decorator-style callback registration
        m.def(
            "property_callback",
            [](const std::string& property_path) {
                return nb::cpp_function([property_path](nb::object func) {
                    auto dot_pos = property_path.find('.');
                    if (dot_pos == std::string::npos) {
                        throw std::runtime_error("Invalid property path. Use 'group.property' format");
                    }
                    std::string group_id = property_path.substr(0, dot_pos);
                    std::string prop_id = property_path.substr(dot_pos + 1);

                    nb::object cb_obj = func;
                    auto cpp_callback = [cb_obj](const std::string&, const std::string&,
                                                 const std::any& old_val, const std::any& new_val) {
                        nb::gil_scoped_acquire gil;
                        try {
                            nb::object py_old, py_new;
                            if (old_val.type() == typeid(float)) {
                                py_old = nb::cast(std::any_cast<float>(old_val));
                                py_new = nb::cast(std::any_cast<float>(new_val));
                            } else if (old_val.type() == typeid(int)) {
                                py_old = nb::cast(std::any_cast<int>(old_val));
                                py_new = nb::cast(std::any_cast<int>(new_val));
                            } else if (old_val.type() == typeid(bool)) {
                                py_old = nb::cast(std::any_cast<bool>(old_val));
                                py_new = nb::cast(std::any_cast<bool>(new_val));
                            } else if (old_val.type() == typeid(size_t)) {
                                py_old = nb::cast(std::any_cast<size_t>(old_val));
                                py_new = nb::cast(std::any_cast<size_t>(new_val));
                            } else if (old_val.type() == typeid(std::string)) {
                                py_old = nb::cast(std::any_cast<std::string>(old_val));
                                py_new = nb::cast(std::any_cast<std::string>(new_val));
                            } else {
                                py_old = nb::none();
                                py_new = nb::none();
                            }
                            cb_obj(py_old, py_new);
                        } catch (const std::exception& e) {
                            LOG_ERROR("Property change callback error: {}", e.what());
                        }
                    };

                    PropertyRegistry::instance().subscribe(group_id, prop_id, cpp_callback);
                    return func;
                });
            },
            nb::arg("property_path"),
            "Decorator for property change handlers.\n"
            "Usage: @lf.property_callback('optimization.means_lr')\n"
            "       def on_lr_change(old_val, new_val): ...");
    }

} // namespace lfs::python
