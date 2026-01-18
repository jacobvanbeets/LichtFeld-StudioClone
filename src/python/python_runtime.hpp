/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#ifdef _WIN32
#ifdef LFS_PYTHON_RUNTIME_EXPORTS
#define LFS_PYTHON_RUNTIME_API __declspec(dllexport)
#else
#define LFS_PYTHON_RUNTIME_API __declspec(dllimport)
#endif
#else
#define LFS_PYTHON_RUNTIME_API
#endif

namespace lfs::vis {
    class Scene;
}

namespace lfs::python {

    // Panel space types
    enum class PanelSpace {
        SidePanel,
        Floating,
        ViewportOverlay
    };

    // Panel callbacks (C-style function pointers for DLL boundary safety)
    using DrawPanelsCallback = void (*)(PanelSpace);
    using DrawSinglePanelCallback = void (*)(const char*);
    using HasPanelsCallback = bool (*)(PanelSpace);
    using CleanupCallback = void (*)();
    using EnsureInitializedCallback = void (*)();
    using PanelNameVisitor = void (*)(const char* name, void* user_data);
    using GetPanelNamesCallback = void (*)(PanelSpace, PanelNameVisitor, void* user_data);

    // Register callbacks from the Python module
    LFS_PYTHON_RUNTIME_API void set_ensure_initialized_callback(EnsureInitializedCallback cb);
    LFS_PYTHON_RUNTIME_API void set_panel_draw_callback(DrawPanelsCallback cb);
    LFS_PYTHON_RUNTIME_API void set_panel_draw_single_callback(DrawSinglePanelCallback cb);
    LFS_PYTHON_RUNTIME_API void set_panel_has_callback(HasPanelsCallback cb);
    LFS_PYTHON_RUNTIME_API void set_panel_names_callback(GetPanelNamesCallback cb);
    LFS_PYTHON_RUNTIME_API void set_python_cleanup_callback(CleanupCallback cb);
    LFS_PYTHON_RUNTIME_API void clear_panel_callbacks();

    // Debug: dumps callback addresses to stderr (detects duplicate DLL loading)
    LFS_PYTHON_RUNTIME_API void debug_dump_callbacks(const char* caller);

    // C++ interface for the visualizer
    LFS_PYTHON_RUNTIME_API void draw_python_panels(PanelSpace space, lfs::vis::Scene* scene = nullptr);
    LFS_PYTHON_RUNTIME_API void draw_python_panel(const std::string& name, lfs::vis::Scene* scene = nullptr);
    LFS_PYTHON_RUNTIME_API bool has_python_panels(PanelSpace space);
    LFS_PYTHON_RUNTIME_API std::vector<std::string> get_python_panel_names(PanelSpace space);
    LFS_PYTHON_RUNTIME_API void invoke_python_cleanup();

    // Operation context for Python code (short-lived, per-call)
    LFS_PYTHON_RUNTIME_API void set_scene_for_python(void* scene);
    LFS_PYTHON_RUNTIME_API void* get_scene_for_python();

    // RAII guard for operation context (used for capability invocations)
    class SceneContextGuard {
    public:
        explicit SceneContextGuard(void* scene) { set_scene_for_python(scene); }
        ~SceneContextGuard() { set_scene_for_python(nullptr); }

        SceneContextGuard(const SceneContextGuard&) = delete;
        SceneContextGuard& operator=(const SceneContextGuard&) = delete;
        SceneContextGuard(SceneContextGuard&&) = delete;
        SceneContextGuard& operator=(SceneContextGuard&&) = delete;
    };

    // Application scene context (long-lived, persists while model is loaded)
    class LFS_PYTHON_RUNTIME_API ApplicationSceneContext {
    public:
        void set(vis::Scene* scene);
        vis::Scene* get() const;
        uint64_t generation() const;

    private:
        std::atomic<vis::Scene*> scene_{nullptr};
        std::atomic<uint64_t> generation_{0};
    };

    LFS_PYTHON_RUNTIME_API void set_application_scene(vis::Scene* scene);
    LFS_PYTHON_RUNTIME_API vis::Scene* get_application_scene();
    LFS_PYTHON_RUNTIME_API uint64_t get_scene_generation();

    LFS_PYTHON_RUNTIME_API void set_gil_state_ready(bool ready);
    LFS_PYTHON_RUNTIME_API bool is_gil_state_ready();

    // Main thread GIL management - these live in the shared library
    // to ensure a single copy across all modules (exe and pyd)
    // Note: void* used to avoid Python.h dependency in header
    LFS_PYTHON_RUNTIME_API void set_main_thread_state(void* state);
    LFS_PYTHON_RUNTIME_API void* get_main_thread_state();
    LFS_PYTHON_RUNTIME_API void acquire_gil_main_thread();
    LFS_PYTHON_RUNTIME_API void release_gil_main_thread();

    // Initialization guards - must be in shared library to avoid duplication
    // These prevent double-init when static lib is linked into both exe and pyd
    LFS_PYTHON_RUNTIME_API void call_once_py_init(std::function<void()> fn);
    LFS_PYTHON_RUNTIME_API void call_once_redirect(std::function<void()> fn);
    LFS_PYTHON_RUNTIME_API void mark_plugins_loaded();
    LFS_PYTHON_RUNTIME_API bool are_plugins_loaded();

} // namespace lfs::python
