/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "py_cameras.hpp"
#include "py_io.hpp"
#include "py_mcp.hpp"
#include "py_packages.hpp"
#include "py_params.hpp"
#include "py_plugins.hpp"
#include "py_scene.hpp"
#include "py_splat_data.hpp"
#include "py_tensor.hpp"
#include "py_ui.hpp"

#include "control/command_api.hpp"
#include "control/control_boundary.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "python/runner.hpp"
#include "training/strategies/istrategy.hpp"
#include "training/trainer.hpp"
#include "visualizer/scene/scene_manager.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
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
    using lfs::training::TrainingSnapshot;

    CommandCenter& get_command_center() {
        auto* cc = lfs::event::command_center();
        if (!cc) {
            throw std::runtime_error("Training system not initialized");
        }
        return *cc;
    }

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

    class PyScopedHandler {
    public:
        PyScopedHandler() = default;
        ~PyScopedHandler() = default;

        PyScopedHandler(const PyScopedHandler&) = delete;
        PyScopedHandler& operator=(const PyScopedHandler&) = delete;
        PyScopedHandler(PyScopedHandler&&) = default;
        PyScopedHandler& operator=(PyScopedHandler&&) = default;

        void on_training_start(nb::callable cb) { add_hook(ControlHook::TrainingStart, cb); }
        void on_iteration_start(nb::callable cb) { add_hook(ControlHook::IterationStart, cb); }
        void on_pre_optimizer_step(nb::callable cb) { add_hook(ControlHook::PreOptimizerStep, cb); }
        void on_post_step(nb::callable cb) { add_hook(ControlHook::PostStep, cb); }
        void on_training_end(nb::callable cb) { add_hook(ControlHook::TrainingEnd, cb); }

        void clear() {
            handler_ = lfs::event::ScopedHandler();
            owned_callbacks_.clear();
        }

    private:
        void add_hook(ControlHook hook, nb::callable cb) {
            nb::object fn = nb::cast<nb::object>(cb);
            owned_callbacks_.push_back(fn);

            handler_.subscribe_hook(hook, [fn](const HookContext& ctx) {
                nb::gil_scoped_acquire gil;
                try {
                    nb::dict d;
                    d["iter"] = ctx.iteration;
                    d["loss"] = ctx.loss;
                    d["num_splats"] = ctx.num_gaussians;
                    d["is_refining"] = ctx.is_refining;
                    fn(d);
                } catch (const std::exception& e) {
                    LOG_ERROR("Python hook error: {}", e.what());
                }
            });
        }

        lfs::event::ScopedHandler handler_;
        std::vector<nb::object> owned_callbacks_;
    };

    class PyContextView {
    public:
        PyContextView() : snapshot_(get_command_center().snapshot()) {
            if (snapshot_.trainer) {
                strategy_ = snapshot_.trainer->getParams().optimization.strategy;
            }
        }

        int iteration() const { return snapshot_.iteration; }
        int max_iterations() const { return snapshot_.max_iterations; }
        float loss() const { return snapshot_.loss; }
        std::size_t num_gaussians() const { return snapshot_.num_gaussians; }
        bool is_refining() const { return snapshot_.is_refining; }
        bool is_training() const { return snapshot_.is_running; }
        bool is_paused() const { return snapshot_.is_paused; }

        std::string phase() const {
            switch (snapshot_.phase) {
            case TrainingPhase::Idle: return "idle";
            case TrainingPhase::IterationStart: return "iteration_start";
            case TrainingPhase::Forward: return "forward";
            case TrainingPhase::Backward: return "backward";
            case TrainingPhase::OptimizerStep: return "optimizer_step";
            case TrainingPhase::SafeControl: return "safe_control";
            default: return "unknown";
            }
        }

        std::string strategy() const { return strategy_; }

        void refresh() {
            snapshot_ = get_command_center().snapshot();
            if (snapshot_.trainer) {
                strategy_ = snapshot_.trainer->getParams().optimization.strategy;
            } else {
                strategy_ = "none";
            }
        }

    private:
        TrainingSnapshot snapshot_;
        std::string strategy_ = "none";
    };

    struct PyGaussiansView {
        std::size_t count() const {
            const auto snap = get_command_center().snapshot();
            if (!snap.trainer)
                return 0;
            return snap.trainer->get_strategy_mutable().get_model().size();
        }

        int sh_degree() const {
            const auto snap = get_command_center().snapshot();
            if (!snap.trainer)
                return 0;
            return snap.trainer->get_strategy_mutable().get_model().get_active_sh_degree();
        }

        int max_sh_degree() const {
            const auto snap = get_command_center().snapshot();
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
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("scale_lr failed: {}", result.error()));
            }
        }

        void set_lr(float value) {
            Command cmd;
            cmd.op = "set_lr";
            cmd.target = CommandTarget::Optimizer;
            cmd.selection = {SelectionKind::All};
            cmd.args["value"] = static_cast<double>(value);
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("set_lr failed: {}", result.error()));
            }
        }

        float get_lr() const {
            const auto snap = get_command_center().snapshot();
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
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("clamp failed: {}", result.error()));
            }
        }

        void scale(const std::string& attr, float factor) {
            Command cmd;
            cmd.op = "scale_attribute";
            cmd.target = CommandTarget::Model;
            cmd.selection = {SelectionKind::All};
            cmd.args["attribute"] = attr;
            cmd.args["factor"] = static_cast<double>(factor);
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("scale failed: {}", result.error()));
            }
        }

        void set(const std::string& attr, float value) {
            Command cmd;
            cmd.op = "set_attribute";
            cmd.target = CommandTarget::Model;
            cmd.selection = {SelectionKind::All};
            cmd.args["attribute"] = attr;
            cmd.args["value"] = static_cast<double>(value);
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("set failed: {}", result.error()));
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
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("pause failed: {}", result.error()));
            }
        }

        void resume() {
            Command cmd;
            cmd.op = "resume";
            cmd.target = CommandTarget::Session;
            cmd.selection = {SelectionKind::All};
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("resume failed: {}", result.error()));
            }
        }

        void request_stop() {
            Command cmd;
            cmd.op = "request_stop";
            cmd.target = CommandTarget::Session;
            cmd.selection = {SelectionKind::All};
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("request_stop failed: {}", result.error()));
            }
        }
    };

    // Thread-local Trainer pointer for get_scene() to access Scene during hooks
    thread_local lfs::training::Trainer* g_current_trainer = nullptr;

    // Thread-local Scene pointer for GUI console context (set before executing user code)
    thread_local lfs::vis::Scene* g_console_scene = nullptr;

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

    // Get Scene from current trainer (for use in hooks) or console context (for GUI)
    lfs::vis::Scene* get_scene_internal() {
        // First try the current trainer (headless mode during hooks)
        if (g_current_trainer) {
            return g_current_trainer->getScene();
        }
        // Fall back to console context (injected by GUI before executing user code)
        return g_console_scene;
    }

    // Set scene context for GUI console execution
    void set_scene_context(lfs::vis::Scene* scene) { g_console_scene = scene; }

    // Clear scene context after GUI console execution
    void clear_scene_context() { g_console_scene = nullptr; }

} // namespace

NB_MODULE(lichtfeld, m) {
    m.doc() = "LichtFeld Python control module for Gaussian splatting";

    // Add user site-packages to sys.path on module import
    {
        auto user_packages = lfs::python::get_user_packages_dir();
        nb::module_ sys = nb::module_::import_("sys");
        nb::list path = nb::cast<nb::list>(sys.attr("path"));
        std::string pkg_path = user_packages.string();
        bool found = false;
        for (size_t i = 0; i < path.size(); ++i) {
            if (nb::cast<std::string>(path[i]) == pkg_path) {
                found = true;
                break;
            }
        }
        if (!found) {
            path.insert(0, pkg_path.c_str());
        }
    }

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

    nb::class_<PyScopedHandler>(m, "ScopedHandler")
        .def(nb::init<>())
        .def("on_training_start", &PyScopedHandler::on_training_start, nb::arg("callback"),
             "Register training start callback")
        .def("on_iteration_start", &PyScopedHandler::on_iteration_start, nb::arg("callback"),
             "Register iteration start callback")
        .def("on_pre_optimizer_step", &PyScopedHandler::on_pre_optimizer_step, nb::arg("callback"),
             "Register pre-optimizer callback")
        .def("on_post_step", &PyScopedHandler::on_post_step, nb::arg("callback"),
             "Register post-step callback")
        .def("on_training_end", &PyScopedHandler::on_training_end, nb::arg("callback"),
             "Register training end callback")
        .def("clear", &PyScopedHandler::clear, "Unregister all callbacks");

    // Context view class (caches snapshot on creation, call refresh() to update)
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
        .def_prop_ro("strategy", &PyContextView::strategy)
        .def("refresh", &PyContextView::refresh, "Update cached snapshot from current state");

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

    // Register Tensor class
    lfs::python::register_tensor(m);

    // Scene submodule
    auto scene_module = m.def_submodule("scene", "Scene graph API");
    lfs::python::register_splat_data(scene_module);
    lfs::python::register_scene(scene_module);
    lfs::python::register_cameras(scene_module);

    // I/O submodule
    auto io_module = m.def_submodule("io", "File I/O operations");
    lfs::python::register_io(io_module);

    // Packages submodule (uses uv for package management)
    lfs::python::register_packages(m);

    // UI submodule (ImGui widgets and panel registration)
    auto ui_module = m.def_submodule("ui", "User interface API");
    lfs::python::register_ui(ui_module);

    // Parameters (OptimizationParameters with RNA-style property access)
    lfs::python::register_params(m);

    // Plugin system (pure Python implementation, C++ bindings for convenience)
    lfs::python::register_plugins(m);

    // MCP (Model Context Protocol) tool registration
    lfs::python::register_mcp(m);

    // Logging submodule
    auto log_module = m.def_submodule("log", "Logging utilities");
    log_module.def(
        "info",
        [](const std::string& msg) { LOG_INFO("[Python] {}", msg); },
        nb::arg("message"),
        "Log an info message");
    log_module.def(
        "debug",
        [](const std::string& msg) { LOG_DEBUG("[Python] {}", msg); },
        nb::arg("message"),
        "Log a debug message");
    log_module.def(
        "warn",
        [](const std::string& msg) { LOG_WARN("[Python] {}", msg); },
        nb::arg("message"),
        "Log a warning message");
    log_module.def(
        "error",
        [](const std::string& msg) { LOG_ERROR("[Python] {}", msg); },
        nb::arg("message"),
        "Log an error message");

    auto app_module = m.def_submodule("app", "Application-level operations");
    app_module.def(
        "open",
        [](const std::string& path_str) {
            namespace cmd = lfs::core::events::cmd;
            cmd::LoadFile{.path = path_str, .is_dataset = true}.emit();
        },
        nb::arg("path"),
        "Open a dataset or file in the application");

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

    // Run a Python script file
    m.def(
        "run", [](const std::string& path) {
            const std::filesystem::path script_path(path);
            if (!std::filesystem::exists(script_path)) {
                throw std::runtime_error("Script not found: " + path);
            }

            std::ifstream file(script_path);
            if (!file) {
                throw std::runtime_error("Cannot open script: " + path);
            }

            std::stringstream buffer;
            buffer << file.rdbuf();
            const std::string code = buffer.str();

            // Add script directory to sys.path and set __file__
            const auto parent = script_path.parent_path().string();
            const auto abs_path = std::filesystem::absolute(script_path).string();

            // Get Python's exec function
            nb::object builtins = nb::module_::import_("builtins");
            nb::object py_exec = builtins.attr("exec");

            std::string setup_code = std::format(
                "import sys\n"
                "if '{}' not in sys.path: sys.path.insert(0, '{}')\n"
                "__file__ = '{}'\n",
                parent, parent, abs_path);

            py_exec(setup_code);
            py_exec(code);

            LOG_INFO("Executed script: {}", path);
        },
        nb::arg("path"), "Execute a Python script file");

    // List scene contents
    m.def(
        "list_scene", []() {
            auto* scene = get_scene_internal();
            if (!scene) {
                nb::print("No scene available");
                return;
            }

            const auto nodes = scene->getNodes();
            nb::print(nb::str("Scene: {} nodes, {} gaussians\n").format(nodes.size(), scene->getTotalGaussianCount()));

            // Build id->node map
            std::unordered_map<int32_t, const lfs::vis::SceneNode*> node_map;
            for (const auto* n : nodes) {
                node_map[n->id] = n;
            }

            // Recursive print function
            std::function<void(const lfs::vis::SceneNode*, int)> print_node =
                [&](const lfs::vis::SceneNode* node, int depth) {
                    std::string indent(depth * 2, ' ');
                    char vis = node->visible ? '+' : '-';
                    char lock = node->locked ? 'L' : ' ';

                    std::string type_name;
                    switch (node->type) {
                    case lfs::vis::NodeType::SPLAT: type_name = "SPLAT"; break;
                    case lfs::vis::NodeType::POINTCLOUD: type_name = "POINTCLOUD"; break;
                    case lfs::vis::NodeType::GROUP: type_name = "GROUP"; break;
                    case lfs::vis::NodeType::CROPBOX: type_name = "CROPBOX"; break;
                    case lfs::vis::NodeType::DATASET: type_name = "DATASET"; break;
                    case lfs::vis::NodeType::CAMERA_GROUP: type_name = "CAMERA_GROUP"; break;
                    case lfs::vis::NodeType::CAMERA: type_name = "CAMERA"; break;
                    case lfs::vis::NodeType::IMAGE_GROUP: type_name = "IMAGE_GROUP"; break;
                    case lfs::vis::NodeType::IMAGE: type_name = "IMAGE"; break;
                    default: type_name = "UNKNOWN"; break;
                    }

                    std::string info = std::format("[{}{}] {} ({}, id={})",
                                                   vis, lock, node->name, type_name, node->id);

                    if (node->gaussian_count > 0) {
                        info += std::format(" [{} splats]", node->gaussian_count);
                    }

                    nb::print(nb::str("{}{}").format(indent, info));

                    for (int32_t child_id : node->children) {
                        auto it = node_map.find(child_id);
                        if (it != node_map.end()) {
                            print_node(it->second, depth + 1);
                        }
                    }
                };

            // Print root nodes
            for (const auto* node : nodes) {
                if (node->parent_id == lfs::vis::NULL_NODE) {
                    print_node(node, 0);
                }
            }
        },
        "Print the scene graph tree");

    // Frame callback for animations
    m.def(
        "on_frame", [](nb::callable cb) {
            nb::object ocb = nb::cast<nb::object>(cb);
            lfs::python::set_frame_callback([ocb](float dt) {
                try {
                    ocb(dt);
                } catch (const std::exception& e) {
                    LOG_ERROR("on_frame callback error: {}", e.what());
                    lfs::python::clear_frame_callback();
                }
            });
            LOG_INFO("Frame callback registered");
        },
        nb::arg("callback"), "Register a callback to be called each frame with delta time (seconds)");

    m.def(
        "stop_animation", []() {
            lfs::python::clear_frame_callback();
            LOG_INFO("Frame callback cleared");
        },
        "Stop any running animation (clears frame callback)");

    // Create 4x4 matrix tensor from nested list (for transforms)
    m.def(
        "mat4", [](const std::vector<std::vector<float>>& rows) -> lfs::python::PyTensor {
            if (rows.size() != 4) {
                throw std::runtime_error("mat4 requires 4 rows");
            }
            for (const auto& row : rows) {
                if (row.size() != 4) {
                    throw std::runtime_error("mat4 requires 4 columns per row");
                }
            }
            auto tensor = lfs::core::Tensor::empty({4, 4}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
            float* data = tensor.ptr<float>();
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    data[i * 4 + j] = rows[i][j];
                }
            }
            return lfs::python::PyTensor(std::move(tensor));
        },
        nb::arg("rows"), "Create a 4x4 matrix tensor from nested list [[r0], [r1], [r2], [r3]]");

    // Quick help
    m.def(
        "help", []() {
            nb::print(R"(LichtFeld Python API

Scene:
  lf.get_scene()      - Get the scene object (None if unavailable)
  lf.list_scene()     - Print scene graph tree
  lf.train_cameras()  - Get training camera dataset
  lf.val_cameras()    - Get validation camera dataset

Training:
  lf.context()        - Get training context (iteration, loss, etc.)
  lf.gaussians()      - Get gaussians info (count, sh_degree)
  lf.session()        - Get session for pause/resume/optimizer control

Hooks (decorators):
  @lf.on_training_start
  @lf.on_iteration_start
  @lf.on_post_step
  @lf.on_pre_optimizer_step
  @lf.on_training_end

Plugin Hooks (RAII):
  handler = lf.ScopedHandler()
  handler.on_iteration_start(callback)
  # Auto-unregisters when handler is destroyed

Utilities:
  lf.run("script.py") - Execute a Python script file
  lf.help()           - Show this help

Example:
  scene = lf.get_scene()
  for node in scene.get_nodes():
      print(node.name, node.type)
)");
        },
        "Show help for lichtfeld module");

    // Internal context API for GUI console (C++ calls these before/after executing user code)
    m.def(
        "_set_scene_context", [](nb::capsule scene_capsule) {
            auto* scene = static_cast<lfs::vis::Scene*>(scene_capsule.data());
            set_scene_context(scene);
        },
        nb::arg("scene_capsule"), "Internal: Set scene context for console execution");

    m.def(
        "_clear_scene_context", []() { clear_scene_context(); },
        "Internal: Clear scene context after console execution");

    // Module metadata
    m.attr("__version__") = "0.1.0";
    m.attr("__all__") = nb::make_tuple(
        // Core access
        "context", "gaussians", "session", "get_scene",
        "train_cameras", "val_cameras",
        // Types
        "Tensor", "Hook", "ScopedHandler",
        // Hook decorators
        "on_training_start", "on_iteration_start",
        "on_post_step", "on_pre_optimizer_step", "on_training_end",
        // Animation
        "on_frame", "stop_animation",
        // Utilities
        "run", "list_scene", "mat4", "help",
        // Submodules
        "scene", "io", "packages", "mcp");
}
