/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_runtime.hpp"

#include <atomic>
#include <cstdio>
#include <mutex>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

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

        // Unique ID for this DLL instance (detects duplicate loading)
        const void* const DLL_INSTANCE_ID = &g_draw_single_callback;

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

    void set_ensure_initialized_callback(EnsureInitializedCallback cb) {
        std::fprintf(stderr, "[pyrt] set_ensure_initialized_callback: %p -> %p (globals at %p)\n",
                     reinterpret_cast<void*>(g_ensure_initialized_callback),
                     reinterpret_cast<void*>(cb),
                     reinterpret_cast<void*>(&g_ensure_initialized_callback));
        std::fflush(stderr);
        g_ensure_initialized_callback = cb;
    }

    void set_panel_draw_callback(DrawPanelsCallback cb) {
        std::fprintf(stderr, "[pyrt] set_panel_draw_callback: %p -> %p (global at %p)\n",
                     reinterpret_cast<void*>(g_draw_callback),
                     reinterpret_cast<void*>(cb),
                     reinterpret_cast<void*>(&g_draw_callback));
        std::fflush(stderr);
        g_draw_callback = cb;
    }

    void set_panel_draw_single_callback(DrawSinglePanelCallback cb) {
        std::fprintf(stderr, "[pyrt] set_panel_draw_single_callback: %p -> %p (global at %p)\n",
                     reinterpret_cast<void*>(g_draw_single_callback),
                     reinterpret_cast<void*>(cb),
                     reinterpret_cast<void*>(&g_draw_single_callback));
        std::fflush(stderr);
        g_draw_single_callback = cb;
    }

    void set_panel_has_callback(HasPanelsCallback cb) {
        std::fprintf(stderr, "[pyrt] set_panel_has_callback: %p -> %p (global at %p)\n",
                     reinterpret_cast<void*>(g_has_callback),
                     reinterpret_cast<void*>(cb),
                     reinterpret_cast<void*>(&g_has_callback));
        std::fflush(stderr);
        g_has_callback = cb;
    }

    void set_panel_names_callback(GetPanelNamesCallback cb) {
        std::fprintf(stderr, "[pyrt] set_panel_names_callback: %p -> %p (global at %p)\n",
                     reinterpret_cast<void*>(g_panel_names_callback),
                     reinterpret_cast<void*>(cb),
                     reinterpret_cast<void*>(&g_panel_names_callback));
        std::fflush(stderr);
        g_panel_names_callback = cb;
    }

    void set_python_cleanup_callback(CleanupCallback cb) {
        std::fprintf(stderr, "[pyrt] set_python_cleanup_callback: %p -> %p (global at %p)\n",
                     reinterpret_cast<void*>(g_cleanup_callback),
                     reinterpret_cast<void*>(cb),
                     reinterpret_cast<void*>(&g_cleanup_callback));
        std::fflush(stderr);
        g_cleanup_callback = cb;
    }

    void clear_panel_callbacks() {
        g_ensure_initialized_callback = nullptr;
        g_draw_callback = nullptr;
        g_draw_single_callback = nullptr;
        g_has_callback = nullptr;
        g_panel_names_callback = nullptr;
        g_cleanup_callback = nullptr;
    }

    void debug_dump_callbacks(const char* caller) {
        std::fprintf(stderr, "\n[pyrt] ======== CALLBACK STATE DUMP ========\n");
        std::fprintf(stderr, "[pyrt] Caller: %s\n", caller);
        std::fprintf(stderr, "[pyrt] DLL_INSTANCE_ID: %p  <-- MUST MATCH between pyd and exe!\n", DLL_INSTANCE_ID);
        std::fprintf(stderr, "[pyrt] g_draw_single_callback:\n");
        std::fprintf(stderr, "[pyrt]   value = %p %s\n",
                     reinterpret_cast<void*>(g_draw_single_callback),
                     g_draw_single_callback ? "(SET)" : "(NULL - PROBLEM!)");
        std::fprintf(stderr, "[pyrt]   &addr = %p\n", reinterpret_cast<void*>(&g_draw_single_callback));
        std::fprintf(stderr, "[pyrt] g_draw_callback:        value=%p\n", reinterpret_cast<void*>(g_draw_callback));
        std::fprintf(stderr, "[pyrt] g_has_callback:         value=%p\n", reinterpret_cast<void*>(g_has_callback));
        std::fprintf(stderr, "[pyrt] g_panel_names_callback: value=%p\n", reinterpret_cast<void*>(g_panel_names_callback));
        std::fprintf(stderr, "[pyrt] ======== END DUMP ========\n\n");
        std::fflush(stderr);
    }

    void invoke_python_cleanup() {
        if (g_cleanup_callback) {
            g_cleanup_callback();
        }
    }

    void draw_python_panels(PanelSpace space, lfs::vis::Scene* /* scene */) {
        std::fprintf(stderr, "[pyrt] draw_python_panels: space=%d, callback=%p\n",
                     static_cast<int>(space), reinterpret_cast<void*>(g_draw_callback));
        std::fflush(stderr);
        if (!g_draw_callback)
            return;
#ifdef LFS_BUILD_PYTHON_BINDINGS
        if (!Py_IsInitialized() || !is_gil_state_ready())
            return;
        const PyGILState_STATE gil = PyGILState_Ensure();
        g_draw_callback(space);
        PyGILState_Release(gil);
#else
        g_draw_callback(space);
#endif
    }

    bool has_python_panels(PanelSpace space) {
        std::fprintf(stderr, "[pyrt] has_python_panels: space=%d\n", static_cast<int>(space));
        std::fflush(stderr);

        if (g_ensure_initialized_callback) {
            std::fprintf(stderr, "[pyrt] has_python_panels: calling ensure_initialized\n");
            std::fflush(stderr);
            g_ensure_initialized_callback();
        }

        if (g_has_callback) {
#ifdef LFS_BUILD_PYTHON_BINDINGS
            if (!Py_IsInitialized() || !is_gil_state_ready())
                return false;
            const PyGILState_STATE gil = PyGILState_Ensure();
            const bool result = g_has_callback(space);
            PyGILState_Release(gil);
            std::fprintf(stderr, "[pyrt] has_python_panels: result=%d\n", result);
            std::fflush(stderr);
            return result;
#else
            return g_has_callback(space);
#endif
        }
        return false;
    }

    std::vector<std::string> get_python_panel_names(PanelSpace space) {
        std::fprintf(stderr, "[pyrt] get_python_panel_names: space=%d\n", static_cast<int>(space));
        std::fflush(stderr);

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
        std::fprintf(stderr, "[pyrt] get_python_panel_names: returning %zu names\n", result.size());
        std::fflush(stderr);
        return result;
    }

    // Test callback defined in this DLL to verify calling convention works
    static void test_callback(const char* label) {
        std::fprintf(stderr, "[pyrt] TEST_CALLBACK ENTERED OK! label='%s'\n", label ? label : "(null)");
        std::fflush(stderr);
    }

    void draw_python_panel(const std::string& name, lfs::vis::Scene* /* scene */) {
        std::fprintf(stderr, "\n[pyrt] ============ draw_python_panel START ============\n");
        std::fprintf(stderr, "[pyrt] name='%s' (length=%zu)\n", name.c_str(), name.length());
        std::fprintf(stderr, "[pyrt] name.c_str() address: %p\n", static_cast<const void*>(name.c_str()));
        std::fflush(stderr);

        // First call: dump full state to detect DLL mismatch
        static bool first_call = true;
        if (first_call) {
            debug_dump_callbacks("draw_python_panel (FIRST CALL from exe)");
            first_call = false;
        }

        std::fprintf(stderr, "[pyrt] DLL_INSTANCE_ID=%p\n", DLL_INSTANCE_ID);
        std::fprintf(stderr, "[pyrt] g_draw_single_callback=%p\n", reinterpret_cast<void*>(g_draw_single_callback));
        std::fprintf(stderr, "[pyrt] &g_draw_single_callback=%p\n", reinterpret_cast<void*>(&g_draw_single_callback));
        std::fflush(stderr);

        if (!g_draw_single_callback) {
            std::fprintf(stderr, "[pyrt] ERROR: g_draw_single_callback is NULL!\n");
            std::fprintf(stderr, "[pyrt] ============ draw_python_panel END (NULL callback) ============\n\n");
            std::fflush(stderr);
            return;
        }

#ifdef LFS_BUILD_PYTHON_BINDINGS
        std::fprintf(stderr, "[pyrt] Python state check:\n");
        std::fprintf(stderr, "[pyrt]   Py_IsInitialized() = %d\n", Py_IsInitialized());
        std::fprintf(stderr, "[pyrt]   is_gil_state_ready() = %d\n", is_gil_state_ready());
        std::fflush(stderr);

        if (!Py_IsInitialized() || !is_gil_state_ready()) {
            std::fprintf(stderr, "[pyrt] Python not ready, returning\n");
            std::fprintf(stderr, "[pyrt] ============ draw_python_panel END (Py not ready) ============\n\n");
            std::fflush(stderr);
            return;
        }

        std::fprintf(stderr, "[pyrt] Acquiring GIL via PyGILState_Ensure()...\n");
        std::fflush(stderr);
        const PyGILState_STATE gil = PyGILState_Ensure();
        std::fprintf(stderr, "[pyrt] GIL acquired, state=%d\n", static_cast<int>(gil));
        std::fflush(stderr);

        // TEST: First call our local test_callback to verify calling convention works
        std::fprintf(stderr, "[pyrt] ---- TEST: Calling LOCAL test_callback ----\n");
        std::fprintf(stderr, "[pyrt] test_callback address: %p\n", reinterpret_cast<void*>(&test_callback));
        std::fflush(stderr);
        test_callback(name.c_str());
        std::fprintf(stderr, "[pyrt] ---- TEST: LOCAL test_callback SUCCEEDED ----\n");
        std::fflush(stderr);

        // Now call the actual callback from pyd
        std::fprintf(stderr, "[pyrt] ---- NOW CALLING PYD CALLBACK ----\n");
        std::fprintf(stderr, "[pyrt] g_draw_single_callback = %p\n", reinterpret_cast<void*>(g_draw_single_callback));

#ifdef _WIN32
        // Check if callback address is valid on Windows
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery(reinterpret_cast<void*>(g_draw_single_callback), &mbi, sizeof(mbi))) {
            std::fprintf(stderr, "[pyrt] Memory info for callback:\n");
            std::fprintf(stderr, "[pyrt]   BaseAddress: %p\n", mbi.BaseAddress);
            std::fprintf(stderr, "[pyrt]   AllocationBase: %p\n", mbi.AllocationBase);
            std::fprintf(stderr, "[pyrt]   RegionSize: %zu\n", mbi.RegionSize);
            std::fprintf(stderr, "[pyrt]   State: %lx (MEM_COMMIT=1000, MEM_FREE=10000, MEM_RESERVE=2000)\n", mbi.State);
            std::fprintf(stderr, "[pyrt]   Protect: %lx (PAGE_EXECUTE_READ=20, PAGE_EXECUTE_READWRITE=40)\n", mbi.Protect);
            std::fprintf(stderr, "[pyrt]   Type: %lx (MEM_IMAGE=1000000, MEM_MAPPED=40000, MEM_PRIVATE=20000)\n", mbi.Type);
            std::fflush(stderr);

            if (mbi.State != MEM_COMMIT) {
                std::fprintf(stderr, "[pyrt] ERROR: Memory not committed!\n");
                std::fflush(stderr);
            }
            if (!(mbi.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY))) {
                std::fprintf(stderr, "[pyrt] ERROR: Memory not executable! Protect=%lx\n", mbi.Protect);
                std::fflush(stderr);
            }
        } else {
            std::fprintf(stderr, "[pyrt] ERROR: VirtualQuery failed! GetLastError=%lu\n", GetLastError());
            std::fflush(stderr);
        }
#endif

        std::fprintf(stderr, "[pyrt] Argument: name.c_str() = %p = '%s'\n",
                     static_cast<const void*>(name.c_str()), name.c_str());
        std::fprintf(stderr, "[pyrt] About to call: g_draw_single_callback(\"%s\")\n", name.c_str());
        std::fprintf(stderr, "[pyrt] >>>>> CALLING NOW <<<<<\n");
        std::fflush(stderr);

        g_draw_single_callback(name.c_str());

        std::fprintf(stderr, "[pyrt] >>>>> CALLBACK RETURNED <<<<<\n");
        std::fprintf(stderr, "[pyrt] Releasing GIL...\n");
        std::fflush(stderr);
        PyGILState_Release(gil);
        std::fprintf(stderr, "[pyrt] GIL released\n");
#else
        std::fprintf(stderr, "[pyrt] (Non-Python build) Calling callback directly\n");
        std::fflush(stderr);
        g_draw_single_callback(name.c_str());
#endif
        std::fprintf(stderr, "[pyrt] ============ draw_python_panel END (SUCCESS) ============\n\n");
        std::fflush(stderr);
    }

} // namespace lfs::python
