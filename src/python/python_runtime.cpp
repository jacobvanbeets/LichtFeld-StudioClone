/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_runtime.hpp"

#include <atomic>
#include <mutex>

#ifdef LFS_BUILD_PYTHON_BINDINGS
#include <Python.h>
#endif

namespace lfs::python {

    namespace {
        EnsureInitializedCallback g_ensure_initialized_callback = nullptr;
        DrawPanelsCallback g_draw_callback = nullptr;
        DrawSinglePanelCallback g_draw_single_callback = nullptr;
        HasPanelsCallback g_has_callback = nullptr;
        GetPanelNamesCallback g_panel_names_callback = nullptr;
        CleanupCallback g_cleanup_callback = nullptr;
        PrepareUIContextCallback g_prepare_ui_context = nullptr;

        // Operation context (short-lived, per-call)
        void* g_scene_for_python = nullptr;

        ApplicationSceneContext g_app_scene_context;
        std::atomic<bool> g_gil_state_ready{false};

#ifdef LFS_BUILD_PYTHON_BINDINGS
        // Main thread GIL state - stored here in the shared library
        // to ensure single copy across all modules
        std::atomic<PyThreadState*> g_main_thread_state{nullptr};
#endif

        // Initialization guards - must be in shared library
        std::once_flag g_py_init_once;
        std::once_flag g_redirect_once;
        std::atomic<bool> g_plugins_loaded{false};

        // ImGui context - shared from exe to pyd for Windows DLL boundary crossing
        void* g_imgui_context{nullptr};
    } // namespace

    // Operation context (short-lived)
    void set_scene_for_python(void* scene) { g_scene_for_python = scene; }
    void* get_scene_for_python() { return g_scene_for_python; }

    // Application context (long-lived)
    void ApplicationSceneContext::set(vis::Scene* scene) {
        scene_.store(scene);
        generation_.fetch_add(1);
    }

    vis::Scene* ApplicationSceneContext::get() const { return scene_.load(); }

    uint64_t ApplicationSceneContext::generation() const { return generation_.load(); }

    void set_application_scene(vis::Scene* scene) { g_app_scene_context.set(scene); }

    vis::Scene* get_application_scene() { return g_app_scene_context.get(); }

    uint64_t get_scene_generation() { return g_app_scene_context.generation(); }

    void set_gil_state_ready(const bool ready) { g_gil_state_ready.store(ready, std::memory_order_release); }
    bool is_gil_state_ready() { return g_gil_state_ready.load(std::memory_order_acquire); }

#ifdef LFS_BUILD_PYTHON_BINDINGS
    void set_main_thread_state(void* state) {
        g_main_thread_state.store(static_cast<PyThreadState*>(state), std::memory_order_release);
    }

    void* get_main_thread_state() {
        return g_main_thread_state.load(std::memory_order_acquire);
    }

    void acquire_gil_main_thread() {
        PyThreadState* state = g_main_thread_state.exchange(nullptr, std::memory_order_acq_rel);
        if (state) {
            PyEval_RestoreThread(state);
        }
    }

    void release_gil_main_thread() {
        PyThreadState* state = PyEval_SaveThread();
        g_main_thread_state.store(state, std::memory_order_release);
    }
#else
    // Stubs when Python bindings disabled
    void set_main_thread_state(void*) {}
    void* get_main_thread_state() { return nullptr; }
    void acquire_gil_main_thread() {}
    void release_gil_main_thread() {}
#endif

    // Initialization guards
    void call_once_py_init(std::function<void()> fn) {
        std::call_once(g_py_init_once, std::move(fn));
    }

    void call_once_redirect(std::function<void()> fn) {
        std::call_once(g_redirect_once, std::move(fn));
    }

    void mark_plugins_loaded() { g_plugins_loaded.store(true, std::memory_order_release); }
    bool are_plugins_loaded() { return g_plugins_loaded.load(std::memory_order_acquire); }

    // ImGui context sharing
    void set_imgui_context(void* ctx) {
        g_imgui_context = ctx;
    }

    void* get_imgui_context() {
        return g_imgui_context;
    }

    void set_ensure_initialized_callback(EnsureInitializedCallback cb) {
        g_ensure_initialized_callback = cb;
    }

    void set_panel_draw_callback(DrawPanelsCallback cb) {
        g_draw_callback = cb;
    }

    void set_panel_draw_single_callback(DrawSinglePanelCallback cb) {
        g_draw_single_callback = cb;
    }

    void set_panel_has_callback(HasPanelsCallback cb) {
        g_has_callback = cb;
    }

    void set_panel_names_callback(GetPanelNamesCallback cb) {
        g_panel_names_callback = cb;
    }

    void set_python_cleanup_callback(CleanupCallback cb) {
        g_cleanup_callback = cb;
    }

    void clear_panel_callbacks() {
        g_ensure_initialized_callback = nullptr;
        g_draw_callback = nullptr;
        g_draw_single_callback = nullptr;
        g_has_callback = nullptr;
        g_panel_names_callback = nullptr;
        g_cleanup_callback = nullptr;
        g_prepare_ui_context = nullptr;
    }

    void set_prepare_ui_context_callback(PrepareUIContextCallback cb) {
        g_prepare_ui_context = cb;
    }

#ifdef LFS_BUILD_PYTHON_BINDINGS
    std::string extract_python_error() {
        PyObject *type, *value, *tb;
        PyErr_Fetch(&type, &value, &tb);
        std::string msg = "(unknown error)";

        if (value) {
            PyObject* str = PyObject_Str(value);
            if (str) {
                const char* c_msg = PyUnicode_AsUTF8(str);
                if (c_msg) {
                    msg = c_msg;
                }
                Py_DECREF(str);
            }
        }

        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(tb);
        return msg;
    }
#else
    std::string extract_python_error() {
        return "(Python bindings not enabled)";
    }
#endif

    void invoke_python_cleanup() {
        if (g_cleanup_callback) {
            g_cleanup_callback();
        }
    }

    void draw_python_panels(PanelSpace space, lfs::vis::Scene* /* scene */) {
        if (!g_draw_callback)
            return;
#ifdef LFS_BUILD_PYTHON_BINDINGS
        if (!Py_IsInitialized() || !is_gil_state_ready())
            return;

        if (g_prepare_ui_context) {
            g_prepare_ui_context();
        }

        const PyGILState_STATE gil = PyGILState_Ensure();
        g_draw_callback(space);
        PyGILState_Release(gil);
#else
        g_draw_callback(space);
#endif
    }

    bool has_python_panels(PanelSpace space) {
        if (g_ensure_initialized_callback) {
            g_ensure_initialized_callback();
        }

        if (g_has_callback) {
#ifdef LFS_BUILD_PYTHON_BINDINGS
            if (!Py_IsInitialized() || !is_gil_state_ready())
                return false;
            const PyGILState_STATE gil = PyGILState_Ensure();
            const bool result = g_has_callback(space);
            PyGILState_Release(gil);
            return result;
#else
            return g_has_callback(space);
#endif
        }
        return false;
    }

    std::vector<std::string> get_python_panel_names(PanelSpace space) {
        if (g_ensure_initialized_callback) {
            g_ensure_initialized_callback();
        }

        if (!g_panel_names_callback)
            return {};

#ifdef LFS_BUILD_PYTHON_BINDINGS
        if (!Py_IsInitialized() || !is_gil_state_ready())
            return {};
        const PyGILState_STATE gil = PyGILState_Ensure();
#endif

        std::vector<std::string> result;
        g_panel_names_callback(
            space,
            [](const char* name, void* ctx) {
                static_cast<std::vector<std::string>*>(ctx)->emplace_back(name);
            },
            &result);

#ifdef LFS_BUILD_PYTHON_BINDINGS
        PyGILState_Release(gil);
#endif
        return result;
    }

    void draw_python_panel(const std::string& name, lfs::vis::Scene* /* scene */) {
        if (!g_draw_single_callback) {
            return;
        }

#ifdef LFS_BUILD_PYTHON_BINDINGS
        if (!Py_IsInitialized() || !is_gil_state_ready()) {
            return;
        }

        if (g_prepare_ui_context) {
            g_prepare_ui_context();
        }

        const PyGILState_STATE gil = PyGILState_Ensure();
        g_draw_single_callback(name.c_str());
        PyGILState_Release(gil);
#else
        g_draw_single_callback(name.c_str());
#endif
    }

} // namespace lfs::python
