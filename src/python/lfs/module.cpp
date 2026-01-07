/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "py_cameras.hpp"
#include "py_scene.hpp"
#include "py_splat_data.hpp"
#include "py_tensor.hpp"

#include "control/command_api.hpp"
#include "control/control_boundary.hpp"
#include "core/logger.hpp"
#include "training/strategies/istrategy.hpp"
#include "training/trainer.hpp"
#include "visualizer/core/services.hpp"
#include "visualizer/scene/scene_manager.hpp"

#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

    using lfs::training::Command;
    using lfs::training::CommandCenter;
    using lfs::training::CommandTarget;
    using lfs::training::ControlBoundary;
    using lfs::training::ControlHook;
    using lfs::training::HookContext;
    using lfs::training::SelectionKind;
    using lfs::training::TrainingPhase;

    // RAII session that registers Python callbacks to the control boundary
    class PyControlSession {
    public:
        PyControlSession() = default;

        ~PyControlSession() { clear(); }

        void clear() {
            for (const auto& reg : registrations_) {
                ControlBoundary::instance().unregister_callback(reg.hook, reg.id);
            }
            registrations_.clear();
        }

        void on_training_start(nb::callable fn) { add(ControlHook::TrainingStart, std::move(fn)); }
        void on_iteration_start(nb::callable fn) { add(ControlHook::IterationStart, std::move(fn)); }
        void on_pre_optimizer_step(nb::callable fn) { add(ControlHook::PreOptimizerStep, std::move(fn)); }
        void on_post_step(nb::callable fn) { add(ControlHook::PostStep, std::move(fn)); }
        void on_training_end(nb::callable fn) { add(ControlHook::TrainingEnd, std::move(fn)); }

    private:
        struct RegistrationHandle {
            ControlHook hook;
            std::size_t id;
        };

        void add(ControlHook hook, nb::callable fn) {
            nb::object fn_obj = std::move(fn);
            auto cb = [fn_obj](const HookContext& ctx) {
                nb::gil_scoped_acquire gil;
                fn_obj(ctx.iteration, ctx.loss, ctx.num_gaussians, ctx.is_refining);
            };

            const auto id = ControlBoundary::instance().register_callback(hook, std::move(cb));
            registrations_.push_back({hook, id});
            owned_callbacks_.push_back(std::move(fn_obj));
        }

        std::vector<RegistrationHandle> registrations_;
        std::vector<nb::object> owned_callbacks_;
    };

    // Simple context view for reading training state
    struct PyContextView {
        int iteration() const { return CommandCenter::instance().snapshot().iteration; }
        int max_iterations() const { return CommandCenter::instance().snapshot().max_iterations; }
        float loss() const { return CommandCenter::instance().snapshot().loss; }
        std::size_t num_gaussians() const { return CommandCenter::instance().snapshot().num_gaussians; }
        bool is_refining() const { return CommandCenter::instance().snapshot().is_refining; }
        bool is_training() const { return CommandCenter::instance().snapshot().is_running; }
        bool is_paused() const { return CommandCenter::instance().snapshot().is_paused; }

        std::string phase() const {
            auto p = CommandCenter::instance().snapshot().phase;
            switch (p) {
            case TrainingPhase::Idle: return "idle";
            case TrainingPhase::IterationStart: return "iteration_start";
            case TrainingPhase::Forward: return "forward";
            case TrainingPhase::Backward: return "backward";
            case TrainingPhase::OptimizerStep: return "optimizer_step";
            case TrainingPhase::SafeControl: return "safe_control";
            default: return "unknown";
            }
        }

        std::string strategy() const {
            const auto snap = CommandCenter::instance().snapshot();
            if (!snap.trainer)
                return "none";
            return snap.trainer->getParams().optimization.strategy;
        }
    };

    // Gaussians info view
    struct PyGaussiansView {
        std::size_t count() const {
            const auto snap = CommandCenter::instance().snapshot();
            if (!snap.trainer)
                return 0;
            return snap.trainer->get_strategy_mutable().get_model().size();
        }

        int sh_degree() const {
            const auto snap = CommandCenter::instance().snapshot();
            if (!snap.trainer)
                return 0;
            return snap.trainer->get_strategy_mutable().get_model().get_active_sh_degree();
        }

        int max_sh_degree() const {
            const auto snap = CommandCenter::instance().snapshot();
            if (!snap.trainer)
                return 0;
            return snap.trainer->get_strategy_mutable().get_model().get_max_sh_degree();
        }
    };

    // Optimizer view for Python - wraps CommandCenter operations
    struct PyOptimizerView {
        void scale_lr(float factor) {
            Command cmd;
            cmd.op = "scale_lr";
            cmd.target = CommandTarget::Optimizer;
            cmd.selection = {SelectionKind::All};
            cmd.args["factor"] = static_cast<double>(factor);
            const auto result = CommandCenter::instance().execute(cmd);
            if (!result) {
                LOG_ERROR("scale_lr failed: {}", result.error());
            }
        }

        void set_lr(float value) {
            Command cmd;
            cmd.op = "set_lr";
            cmd.target = CommandTarget::Optimizer;
            cmd.selection = {SelectionKind::All};
            cmd.args["value"] = static_cast<double>(value);
            const auto result = CommandCenter::instance().execute(cmd);
            if (!result) {
                LOG_ERROR("set_lr failed: {}", result.error());
            }
        }

        float get_lr() const {
            const auto snap = CommandCenter::instance().snapshot();
            if (!snap.trainer)
                return 0.0f;
            return snap.trainer->get_strategy_mutable().get_optimizer().get_lr();
        }
    };

    // Model view for Python - wraps CommandCenter operations
    struct PyModelView {
        void clamp(const std::string& attr, std::optional<float> min_val, std::optional<float> max_val) {
            Command cmd;
            cmd.op = "clamp_attribute";
            cmd.target = CommandTarget::Model;
            cmd.selection = {SelectionKind::All};
            cmd.args["attribute"] = attr;
            if (min_val)
                cmd.args["min"] = static_cast<double>(*min_val);
            if (max_val)
                cmd.args["max"] = static_cast<double>(*max_val);
            const auto result = CommandCenter::instance().execute(cmd);
            if (!result) {
                LOG_ERROR("clamp failed: {}", result.error());
            }
        }

        void scale(const std::string& attr, float factor) {
            Command cmd;
            cmd.op = "scale_attribute";
            cmd.target = CommandTarget::Model;
            cmd.selection = {SelectionKind::All};
            cmd.args["attribute"] = attr;
            cmd.args["factor"] = static_cast<double>(factor);
            const auto result = CommandCenter::instance().execute(cmd);
            if (!result) {
                LOG_ERROR("scale failed: {}", result.error());
            }
        }

        void set(const std::string& attr, float value) {
            Command cmd;
            cmd.op = "set_attribute";
            cmd.target = CommandTarget::Model;
            cmd.selection = {SelectionKind::All};
            cmd.args["attribute"] = attr;
            cmd.args["value"] = static_cast<double>(value);
            const auto result = CommandCenter::instance().execute(cmd);
            if (!result) {
                LOG_ERROR("set failed: {}", result.error());
            }
        }
    };

    // Session view for Python - provides access to optimizer and model operations
    struct PySession {
        PyOptimizerView optimizer() { return {}; }
        PyModelView model() { return {}; }

        void pause() {
            Command cmd;
            cmd.op = "pause";
            cmd.target = CommandTarget::Session;
            cmd.selection = {SelectionKind::All};
            const auto result = CommandCenter::instance().execute(cmd);
            if (!result) {
                LOG_ERROR("pause failed: {}", result.error());
            }
        }

        void resume() {
            Command cmd;
            cmd.op = "resume";
            cmd.target = CommandTarget::Session;
            cmd.selection = {SelectionKind::All};
            const auto result = CommandCenter::instance().execute(cmd);
            if (!result) {
                LOG_ERROR("resume failed: {}", result.error());
            }
        }

        void request_stop() {
            Command cmd;
            cmd.op = "request_stop";
            cmd.target = CommandTarget::Session;
            cmd.selection = {SelectionKind::All};
            const auto result = CommandCenter::instance().execute(cmd);
            if (!result) {
                LOG_ERROR("request_stop failed: {}", result.error());
            }
        }
    };

    // Thread-local Trainer pointer for get_scene() to access Scene during hooks
    thread_local lfs::training::Trainer* g_current_trainer = nullptr;

    // RAII guard to set/clear current trainer
    struct TrainerGuard {
        TrainerGuard(lfs::training::Trainer* t) { g_current_trainer = t; }
        ~TrainerGuard() { g_current_trainer = nullptr; }
    };

    // Hook registration helper
    std::size_t register_hook(ControlHook hook, nb::callable cb) {
        if (!cb)
            return 0;
        const nb::object ocb = nb::cast<nb::object>(cb);
        LOG_INFO("Python hook registered for hook {}", static_cast<int>(hook));
        return ControlBoundary::instance().register_callback(hook, [ocb, hook](const HookContext& ctx) {
            nb::gil_scoped_acquire guard;
            TrainerGuard trainer_guard(ctx.trainer);
            LOG_DEBUG("Python hook invoke hook={} iter={}", static_cast<int>(hook), ctx.iteration);
            try {
                nb::dict d;
                d["iter"] = ctx.iteration;
                d["loss"] = ctx.loss;
                d["num_splats"] = ctx.num_gaussians;
                d["is_refining"] = ctx.is_refining;
                ocb(d);
            } catch (const std::exception& e) {
                LOG_ERROR("Python hook threw: {}", e.what());
            }
        });
    }

    // Get Scene from current trainer (for use in hooks) or services (for GUI)
    lfs::vis::Scene* get_scene_internal() {
        // First try the current trainer (headless mode)
        if (g_current_trainer) {
            return g_current_trainer->getScene();
        }
        // Fall back to services (GUI mode)
        auto* const scene_manager = lfs::vis::services().sceneOrNull();
        if (scene_manager) {
            return &scene_manager->getScene();
        }
        return nullptr;
    }

} // namespace

NB_MODULE(lichtfeld, m) {
    m.doc() = "LichtFeld Python control module for Gaussian splatting";

    // Enum for hook types
    nb::enum_<ControlHook>(m, "Hook")
        .value("training_start", ControlHook::TrainingStart)
        .value("iteration_start", ControlHook::IterationStart)
        .value("pre_optimizer_step", ControlHook::PreOptimizerStep)
        .value("post_step", ControlHook::PostStep)
        .value("training_end", ControlHook::TrainingEnd);

    // Session class for managing callbacks
    nb::class_<PyControlSession>(m, "Session")
        .def(nb::init<>())
        .def("on_training_start", &PyControlSession::on_training_start, "Register training start callback")
        .def("on_iteration_start", &PyControlSession::on_iteration_start, "Register iteration start callback")
        .def("on_pre_optimizer_step", &PyControlSession::on_pre_optimizer_step, "Register pre-optimizer callback")
        .def("on_post_step", &PyControlSession::on_post_step, "Register post-step callback")
        .def("on_training_end", &PyControlSession::on_training_end, "Register training end callback")
        .def("clear", &PyControlSession::clear, "Unregister all callbacks");

    // Context view class
    nb::class_<PyContextView>(m, "Context")
        .def(nb::init<>())
        .def_prop_ro("iteration", &PyContextView::iteration)
        .def_prop_ro("max_iterations", &PyContextView::max_iterations)
        .def_prop_ro("loss", &PyContextView::loss)
        .def_prop_ro("num_gaussians", &PyContextView::num_gaussians)
        .def_prop_ro("is_refining", &PyContextView::is_refining)
        .def_prop_ro("is_training", &PyContextView::is_training)
        .def_prop_ro("is_paused", &PyContextView::is_paused)
        .def_prop_ro("phase", &PyContextView::phase)
        .def_prop_ro("strategy", &PyContextView::strategy);

    // Gaussians view class
    nb::class_<PyGaussiansView>(m, "Gaussians")
        .def(nb::init<>())
        .def_prop_ro("count", &PyGaussiansView::count)
        .def_prop_ro("sh_degree", &PyGaussiansView::sh_degree)
        .def_prop_ro("max_sh_degree", &PyGaussiansView::max_sh_degree);

    // Optimizer view class
    nb::class_<PyOptimizerView>(m, "Optimizer")
        .def(nb::init<>())
        .def("scale_lr", &PyOptimizerView::scale_lr, nb::arg("factor"), "Scale learning rate by factor")
        .def("set_lr", &PyOptimizerView::set_lr, nb::arg("value"), "Set learning rate")
        .def("get_lr", &PyOptimizerView::get_lr, "Get current learning rate");

    // Model view class
    nb::class_<PyModelView>(m, "Model")
        .def(nb::init<>())
        .def("clamp", &PyModelView::clamp, nb::arg("attr"), nb::arg("min") = nb::none(), nb::arg("max") = nb::none(),
             "Clamp attribute values")
        .def("scale", &PyModelView::scale, nb::arg("attr"), nb::arg("factor"), "Scale attribute by factor")
        .def("set", &PyModelView::set, nb::arg("attr"), nb::arg("value"), "Set attribute value");

    // Session class
    nb::class_<PySession>(m, "Session")
        .def(nb::init<>())
        .def("optimizer", &PySession::optimizer, "Get optimizer view")
        .def("model", &PySession::model, "Get model view")
        .def("pause", &PySession::pause, "Pause training")
        .def("resume", &PySession::resume, "Resume training")
        .def("request_stop", &PySession::request_stop, "Request training stop");

    // Convenience functions
    m.def(
        "context", []() { return PyContextView{}; }, "Get current training context");
    m.def(
        "gaussians", []() { return PyGaussiansView{}; }, "Get Gaussians info");
    m.def(
        "session", []() { return PySession{}; }, "Get training session");

    // Hook registration functions (decorator-style)
    m.def(
        "on_training_start", [](nb::callable cb) {
            register_hook(ControlHook::TrainingStart, cb);
            return cb;
        },
        nb::arg("callback"), "Decorator for training start handler");

    m.def(
        "on_iteration_start", [](nb::callable cb) {
            register_hook(ControlHook::IterationStart, cb);
            return cb;
        },
        nb::arg("callback"), "Decorator for iteration start handler");

    m.def(
        "on_post_step", [](nb::callable cb) {
            register_hook(ControlHook::PostStep, cb);
            return cb;
        },
        nb::arg("callback"), "Decorator for post-step handler");

    m.def(
        "on_pre_optimizer_step", [](nb::callable cb) {
            register_hook(ControlHook::PreOptimizerStep, cb);
            return cb;
        },
        nb::arg("callback"), "Decorator for pre-optimizer handler");

    m.def(
        "on_training_end", [](nb::callable cb) {
            register_hook(ControlHook::TrainingEnd, cb);
            return cb;
        },
        nb::arg("callback"), "Decorator for training end handler");

    // Submodule: lf.handlers for organized access
    auto handlers = m.def_submodule("handlers", "Event handlers");
    handlers.def("on_training_start", [](nb::callable cb) {
        register_hook(ControlHook::TrainingStart, cb);
        return cb;
    });
    handlers.def("on_iteration_start", [](nb::callable cb) {
        register_hook(ControlHook::IterationStart, cb);
        return cb;
    });
    handlers.def("on_iteration_post", [](nb::callable cb) {
        register_hook(ControlHook::PostStep, cb);
        return cb;
    });
    handlers.def("on_post_step", [](nb::callable cb) {
        register_hook(ControlHook::PostStep, cb);
        return cb;
    });
    handlers.def("on_pre_optimizer_step", [](nb::callable cb) {
        register_hook(ControlHook::PreOptimizerStep, cb);
        return cb;
    });
    handlers.def("on_training_end", [](nb::callable cb) {
        register_hook(ControlHook::TrainingEnd, cb);
        return cb;
    });

    // Register Tensor class
    lfs::python::register_tensor(m);

    // Scene submodule
    auto scene_module = m.def_submodule("scene", "Scene graph API");
    lfs::python::register_splat_data(scene_module);
    lfs::python::register_scene(scene_module);
    lfs::python::register_cameras(scene_module);

    // Get scene function - works in both headless (during hooks) and GUI mode
    m.def(
        "get_scene", []() -> std::optional<lfs::python::PyScene> {
            auto* scene = get_scene_internal();
            if (!scene) {
                return std::nullopt;
            }
            return lfs::python::PyScene(scene);
        },
        "Get the current scene (None if not available)");

    // Get train/val cameras from scene
    m.def(
        "train_cameras", []() -> std::optional<lfs::python::PyCameraDataset> {
            auto* scene = get_scene_internal();
            if (!scene) {
                return std::nullopt;
            }
            auto dataset = scene->getTrainCameras();
            if (!dataset) {
                return std::nullopt;
            }
            return lfs::python::PyCameraDataset(dataset);
        },
        "Get training cameras (None if not available)");

    m.def(
        "val_cameras", []() -> std::optional<lfs::python::PyCameraDataset> {
            auto* scene = get_scene_internal();
            if (!scene) {
                return std::nullopt;
            }
            auto dataset = scene->getValCameras();
            if (!dataset) {
                return std::nullopt;
            }
            return lfs::python::PyCameraDataset(dataset);
        },
        "Get validation cameras (None if not available)");
}
