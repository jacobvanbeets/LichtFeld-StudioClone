/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "runner.hpp"
#include "package_manager.hpp"

#include <filesystem>
#include <format>
#include <string>

#include <core/logger.hpp>

#ifdef LFS_BUILD_PYTHON_BINDINGS
#include "py_panel_registry.hpp"
#include "training/control/control_boundary.hpp"
#include <Python.h>
#include <mutex>
#endif

namespace lfs::python {

#ifdef LFS_BUILD_PYTHON_BINDINGS
    static std::once_flag g_py_init_once;
    static std::function<void(const std::string&, bool)> g_output_callback;
    static std::mutex g_output_mutex;

    // Python C extension for capturing output
    static PyObject* capture_write(PyObject* self, PyObject* args) {
        (void)self;
        const char* text = nullptr;
        int is_stderr = 0;
        if (!PyArg_ParseTuple(args, "si", &text, &is_stderr)) {
            return nullptr;
        }
        {
            std::lock_guard lock(g_output_mutex);
            if (g_output_callback && text) {
                g_output_callback(text, is_stderr != 0);
            }
        }
        Py_RETURN_NONE;
    }

    static PyMethodDef g_capture_methods[] = {
        {"write", capture_write, METH_VARARGS, "Write to output callback"},
        {nullptr, nullptr, 0, nullptr}};

    static PyModuleDef g_capture_module = {
        PyModuleDef_HEAD_INIT, "_lfs_output", nullptr, -1, g_capture_methods};

    static PyObject* init_capture_module() {
        return PyModule_Create(&g_capture_module);
    }

    static void install_output_capture() {
        // Register the capture module
        PyImport_AppendInittab("_lfs_output", init_capture_module);
    }

    static void redirect_output() {
        const char* redirect_code = R"(
import sys
import _lfs_output

class OutputCapture:
    def __init__(self, is_stderr=False):
        self._is_stderr = 1 if is_stderr else 0
    def write(self, text):
        if text:
            _lfs_output.write(text, self._is_stderr)
    def flush(self):
        pass

sys.stdout = OutputCapture(False)
sys.stderr = OutputCapture(True)
)";
        PyRun_SimpleString(redirect_code);
        LOG_DEBUG("Python output capture installed");
    }
#endif

    void set_output_callback(std::function<void(const std::string&, bool)> callback) {
#ifdef LFS_BUILD_PYTHON_BINDINGS
        std::lock_guard lock(g_output_mutex);
        g_output_callback = std::move(callback);
#else
        (void)callback;
#endif
    }

    void write_output(const std::string& text, bool is_error) {
#ifdef LFS_BUILD_PYTHON_BINDINGS
        std::lock_guard lock(g_output_mutex);
        if (g_output_callback) {
            g_output_callback(text, is_error);
        }
#else
        (void)text;
        (void)is_error;
#endif
    }

#ifdef LFS_BUILD_PYTHON_BINDINGS
    static PyThreadState* g_main_thread_state = nullptr;
#endif

    std::filesystem::path get_user_packages_dir() {
        return PackageManager::instance().site_packages_dir();
    }

    void ensure_initialized() {
#ifdef LFS_BUILD_PYTHON_BINDINGS
        std::call_once(g_py_init_once, [] {
            install_output_capture();
            Py_Initialize();
            PyEval_InitThreads();

            // Add user site-packages to sys.path (before releasing GIL)
            std::filesystem::path user_packages = get_user_packages_dir();
            if (!std::filesystem::exists(user_packages)) {
                std::error_code ec;
                std::filesystem::create_directories(user_packages, ec);
                if (ec) {
                    LOG_WARN("Failed to create user packages dir: {}", ec.message());
                }
            }

            PyObject* sys_path = PySys_GetObject("path");
            if (sys_path) {
                PyObject* py_path = PyUnicode_FromString(user_packages.string().c_str());
                PyList_Insert(sys_path, 0, py_path);
                Py_DECREF(py_path);
                LOG_INFO("Added user packages dir to Python path: {}", user_packages.string());
            }

            g_main_thread_state = PyEval_SaveThread();
            LOG_INFO("Python interpreter initialized and GIL released");
        });
#endif
    }

    void finalize() {
#ifdef LFS_BUILD_PYTHON_BINDINGS
        if (g_main_thread_state && Py_IsInitialized()) {
            // Restore the main thread state (reacquire GIL)
            PyEval_RestoreThread(g_main_thread_state);
            g_main_thread_state = nullptr;

            // Clear all callbacks that hold Python objects (nanobind::object)
            // This must be done while GIL is held since nanobind::object
            // destructor decrements Python reference counts
            lfs::training::ControlBoundary::instance().clear_all();

            // Clear frame callback if set
            clear_frame_callback();

            // Clear Python UI registries that hold nb::object references
            // These singletons would otherwise destroy nb::objects during
            // static destruction, after Python is gone
            invoke_python_cleanup();

            // Run garbage collection to clean up cycles
            PyGC_Collect();

            // NOTE: We intentionally do NOT call Py_FinalizeEx() here.
            // nanobind stores type information in static storage that gets
            // destroyed during C++ static destruction (after main returns).
            // If we call Py_FinalizeEx(), Python type objects are destroyed,
            // and then nanobind's static destructors crash trying to access them.
            //
            // By not calling Py_FinalizeEx():
            // 1. All our callbacks and references are properly cleaned up above
            // 2. Python interpreter stays alive until program exit
            // 3. OS reclaims all memory when process exits (no actual leak)
            // 4. No crashes during static destruction
            //
            // This is a known limitation of embedding Python with nanobind.
            // The memory "leak" is only until process exit.
            LOG_INFO("Python cleanup completed (interpreter left running for safe exit)");
        }
#endif
    }

    bool was_python_used() {
#ifdef LFS_BUILD_PYTHON_BINDINGS
        return g_main_thread_state != nullptr || Py_IsInitialized();
#else
        return false;
#endif
    }

#ifdef LFS_BUILD_PYTHON_BINDINGS
    static std::once_flag g_redirect_once;
#endif

    void install_output_redirect() {
#ifdef LFS_BUILD_PYTHON_BINDINGS
        std::call_once(g_redirect_once, [] {
            const PyGILState_STATE gil = PyGILState_Ensure();
            redirect_output();
            PyGILState_Release(gil);
        });
#endif
    }

    void update_python_path() {
#ifdef LFS_BUILD_PYTHON_BINDINGS
        const auto packages = get_user_packages_dir();
        if (!std::filesystem::exists(packages))
            return;

        const PyGILState_STATE gil = PyGILState_Ensure();

        PyObject* const sys_path = PySys_GetObject("path");
        if (sys_path) {
            const auto path_str = packages.string();
            PyObject* const py_path = PyUnicode_FromString(path_str.c_str());
            if (PySequence_Contains(sys_path, py_path) == 0) {
                PyList_Insert(sys_path, 0, py_path);
                LOG_INFO("Added to sys.path: {}", path_str);
            }
            Py_DECREF(py_path);
        }

        PyGILState_Release(gil);
#endif
    }

    std::expected<void, std::string> run_scripts(const std::vector<std::filesystem::path>& scripts) {
        if (scripts.empty()) {
            return {};
        }

#ifndef LFS_BUILD_PYTHON_BINDINGS
        return std::unexpected("Python bindings disabled; rebuild with -DBUILD_PYTHON_BINDINGS=ON");
#else
        ensure_initialized();

        const PyGILState_STATE gil_state = PyGILState_Ensure();

        // Install output redirect (calls redirect_output() once)
        std::call_once(g_redirect_once, [] { redirect_output(); });

        // Add build directory (where lichtfeld.so lives) to sys.path
        {
            const std::filesystem::path build_python_dir =
                std::filesystem::path(PROJECT_ROOT_PATH) / "build" / "src" / "python";

            PyObject* sys_path = PySys_GetObject("path"); // borrowed
            PyObject* py_path = PyUnicode_FromString(build_python_dir.string().c_str());
            PyList_Append(sys_path, py_path);
            Py_DECREF(py_path);
            LOG_DEBUG("Added {} to Python path", build_python_dir.string());
        }

        // Pre-import lichtfeld module to catch any initialization errors early
        {
            PyObject* lf_module = PyImport_ImportModule("lichtfeld");
            if (!lf_module) {
                PyErr_Print();
                PyGILState_Release(gil_state);
                return std::unexpected("Failed to import lichtfeld module - check build output");
            }
            Py_DECREF(lf_module);
            LOG_INFO("Successfully pre-imported lichtfeld module");
        }

        for (const auto& script : scripts) {
            if (!std::filesystem::exists(script)) {
                PyGILState_Release(gil_state);
                return std::unexpected(std::format("Python script not found: {}", script.string()));
            }

            // Ensure script directory is on sys.path
            const auto parent = script.parent_path().string();
            if (!parent.empty()) {
                PyObject* sys_path = PySys_GetObject("path"); // borrowed ref
                PyObject* py_parent = PyUnicode_FromString(parent.c_str());
                if (sys_path && py_parent) {
                    PyList_Append(sys_path, py_parent);
                }
                Py_XDECREF(py_parent);
            }

            FILE* const fp = fopen(script.string().c_str(), "r");
            if (!fp) {
                PyGILState_Release(gil_state);
                return std::unexpected(std::format("Failed to open Python script: {}", script.string()));
            }

            LOG_INFO("Executing Python script: {}", script.string());
            const int rc = PyRun_SimpleFileEx(fp, script.string().c_str(), /*closeit=*/1);
            if (rc != 0) {
                PyGILState_Release(gil_state);
                return std::unexpected(std::format("Python script failed: {} (rc={})", script.string(), rc));
            }

            LOG_INFO("Python script completed: {}", script.string());
        }

        PyGILState_Release(gil_state);
        return {};
#endif
    }

    std::string format_python_code(const std::string& code) {
#ifndef LFS_BUILD_PYTHON_BINDINGS
        return code;
#else
        if (code.empty())
            return code;

        auto& pm = PackageManager::instance();
        if (!pm.is_installed("black")) {
            if (!pm.ensure_venv()) {
                LOG_ERROR("Failed to create venv for black");
                return code;
            }
            LOG_INFO("Installing black...");
            const auto install_result = pm.install("black");
            if (!install_result.success) {
                LOG_ERROR("Failed to install black: {}", install_result.error);
                return code;
            }
            update_python_path();
        }

        ensure_initialized();
        const PyGILState_STATE gil = PyGILState_Ensure();

        static constexpr const char* FORMAT_CODE = R"(
def _lfs_format_code(code):
    import importlib, sys
    importlib.invalidate_caches()
    try:
        import black
    except ImportError as e:
        print(f"[format] ImportError: {e}", file=sys.stderr)
        return None

    # Normalize unicode characters that break parsing (from copy-paste)
    replacements = {
        '\u201c': '"', '\u201d': '"',  # Smart double quotes
        '\u2018': "'", '\u2019': "'",  # Smart single quotes
        '\u2212': '-',                  # Unicode minus
        '\u2013': '-', '\u2014': '-',  # En-dash, em-dash
        '\u00a0': ' ',                  # Non-breaking space
        '\u2003': ' ', '\u2002': ' ',  # Em space, en space
        '\u2009': ' ',                  # Thin space
    }
    for old, new in replacements.items():
        code = code.replace(old, new)

    # Normalize line endings and remove trailing whitespace
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.rstrip() for line in code.split('\n')]

    # Remove leading empty lines
    while lines and not lines[0].strip():
        lines.pop(0)

    # Remove trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return code

    # Convert tabs to spaces consistently
    lines = [line.replace('\t', '    ') for line in lines]

    # Find indentation levels for all non-empty lines
    indents = []
    for line in lines:
        if line.strip():
            indents.append(len(line) - len(line.lstrip()))

    if not indents:
        return code

    # If first line has 0 indent but others have consistent indent,
    # this is likely a copy-paste issue - use the mode of other indents
    min_indent = min(indents)
    if min_indent == 0 and len(indents) > 1:
        other_indents = [i for i in indents[1:] if i > 0]
        if other_indents:
            # Find the smallest non-zero indent from other lines
            min_other = min(other_indents)
            # Check if most lines use this or a multiple of it
            consistent = sum(1 for i in other_indents if i >= min_other) / len(other_indents)
            if consistent > 0.5:
                min_indent = min_other

    # Remove common indentation
    dedented = []
    for line in lines:
        if line.strip():
            current_indent = len(line) - len(line.lstrip())
            if current_indent >= min_indent:
                dedented.append(line[min_indent:])
            else:
                dedented.append(line.lstrip())  # Line has less indent, just strip it
        else:
            dedented.append('')

    cleaned = '\n'.join(dedented)

    try:
        return black.format_str(cleaned, mode=black.Mode())
    except Exception as e:
        print(f"[format] Error: {e}", file=sys.stderr)
        print(f"[format] Code was:\n{repr(cleaned)}", file=sys.stderr)
        return None
)";

        PyRun_SimpleString(FORMAT_CODE);

        PyObject* const main_module = PyImport_AddModule("__main__");
        if (!main_module) {
            PyGILState_Release(gil);
            return code;
        }

        PyObject* const main_dict = PyModule_GetDict(main_module);
        PyObject* const format_func = PyDict_GetItemString(main_dict, "_lfs_format_code");
        if (!format_func || !PyCallable_Check(format_func)) {
            PyGILState_Release(gil);
            return code;
        }

        std::string result = code;
        PyObject* const py_code = PyUnicode_FromString(code.c_str());
        PyObject* const py_result = PyObject_CallFunctionObjArgs(format_func, py_code, nullptr);
        Py_DECREF(py_code);

        if (py_result) {
            if (PyUnicode_Check(py_result)) {
                const char* const formatted = PyUnicode_AsUTF8(py_result);
                if (formatted)
                    result = formatted;
            }
            Py_DECREF(py_result);
        } else {
            PyErr_Clear();
        }

        PyGILState_Release(gil);
        return result;
#endif
    }

    // Frame callback for animations
    static std::function<void(float)> g_frame_callback;
    static std::mutex g_frame_mutex;

    void set_frame_callback(std::function<void(float)> callback) {
        std::lock_guard lock(g_frame_mutex);
        g_frame_callback = std::move(callback);
    }

    void clear_frame_callback() {
        std::lock_guard lock(g_frame_mutex);
        g_frame_callback = nullptr;
    }

    bool has_frame_callback() {
        std::lock_guard lock(g_frame_mutex);
        return g_frame_callback != nullptr;
    }

    void tick_frame_callback(float dt) {
        std::function<void(float)> cb;
        {
            std::lock_guard lock(g_frame_mutex);
            cb = g_frame_callback;
        }
        if (cb) {
#ifdef LFS_BUILD_PYTHON_BINDINGS
            const PyGILState_STATE gil = PyGILState_Ensure();
            try {
                cb(dt);
            } catch (const std::exception& e) {
                LOG_ERROR("Frame callback error: {}", e.what());
            }
            PyGILState_Release(gil);
#else
            cb(dt);
#endif
        }
    }

} // namespace lfs::python
