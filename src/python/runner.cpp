/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "runner.hpp"
#include "package_manager.hpp"

#include <cstdio>
#include <filesystem>
#include <format>
#include <string>

#include <core/executable_path.hpp>
#include <core/logger.hpp>
#include <core/path_utils.hpp>

#ifdef LFS_BUILD_PYTHON_BINDINGS
#include "py_panel_registry.hpp"
#include "training/control/control_boundary.hpp"
#include <Python.h>
#include <mutex>
#endif

namespace lfs::python {

#ifdef LFS_BUILD_PYTHON_BINDINGS
    static std::once_flag g_py_init_once;
    static bool g_we_initialized_python = false;

    namespace {
        struct EnsureInitializedRegistrar {
            EnsureInitializedRegistrar() {
                set_ensure_initialized_callback(ensure_initialized);
            }
        };
        static EnsureInitializedRegistrar g_registrar;
    } // namespace

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

    static void register_output_module_post_init() {
        PyObject* modules = PyImport_GetModuleDict();
        if (PyDict_GetItemString(modules, "_lfs_output")) {
            return;
        }
        PyObject* module = PyModule_Create(&g_capture_module);
        if (module) {
            PyDict_SetItemString(modules, "_lfs_output", module);
            Py_DECREF(module);
        }
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
    static bool g_plugins_loaded = false;

    static void add_dll_directories() {
#ifdef _WIN32
        // Python 3.8+ on Windows requires os.add_dll_directory() for DLL loading
        // First add the executable directory using C++ (more reliable)
        const auto exe_dir = lfs::core::getExecutableDir();
        const auto exe_dir_str = lfs::core::path_to_utf8(exe_dir);

        std::string add_dll_code = std::format(R"(
import os
def _add_dll_dirs():
    dirs_to_add = [
        r'{}',  # Executable directory
    ]
    # Also add CUDA path if available
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        dirs_to_add.append(os.path.join(cuda_path, 'bin'))

    # Add vcpkg bin if it exists
    vcpkg_bin = os.path.join(r'{}', 'vcpkg_installed', 'x64-windows', 'bin')
    if os.path.isdir(vcpkg_bin):
        dirs_to_add.append(vcpkg_bin)

    for d in dirs_to_add:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
                print(f'[DLL] Added: {{d}}')
            except Exception as e:
                print(f'[DLL] Failed to add {{d}}: {{e}}')
_add_dll_dirs()
)",
                                               exe_dir_str, exe_dir_str);

        PyRun_SimpleString(add_dll_code.c_str());
        LOG_INFO("Windows DLL directories configured for: {}", exe_dir_str);
#endif
    }

    static void ensure_plugins_loaded() {
        if (g_plugins_loaded)
            return;
        g_plugins_loaded = true;

        add_dll_directories();

        LOG_INFO("Attempting to import lichtfeld module...");
        PyObject* lf = PyImport_ImportModule("lichtfeld");
        if (!lf) {
            PyErr_Print();
            LOG_ERROR("Failed to import lichtfeld for plugin loading");
            LOG_ERROR("Check that lichtfeld.pyd is in sys.path and all DLL dependencies are available");
            return;
        }
        LOG_INFO("lichtfeld module imported successfully");

        PyObject* lfs_plugins = PyImport_ImportModule("lfs_plugins");
        if (lfs_plugins) {
            PyObject* register_fn = PyObject_GetAttrString(lfs_plugins, "register_builtin_panels");
            if (register_fn) {
                PyObject* result = PyObject_CallNoArgs(register_fn);
                if (!result) {
                    PyErr_Print();
                    LOG_ERROR("Failed to register builtin panels");
                } else {
                    Py_DECREF(result);
                }
                Py_DECREF(register_fn);
            }
            Py_DECREF(lfs_plugins);
        }

        PyObject* plugins = PyObject_GetAttrString(lf, "plugins");
        if (plugins) {
            PyObject* load_all = PyObject_GetAttrString(plugins, "load_all");
            if (load_all) {
                PyObject* results = PyObject_CallNoArgs(load_all);
                if (results && PyDict_Check(results)) {
                    PyObject* key;
                    PyObject* value;
                    Py_ssize_t pos = 0;
                    while (PyDict_Next(results, &pos, &key, &value)) {
                        const char* name = PyUnicode_AsUTF8(key);
                        if (PyObject_IsTrue(value)) {
                            LOG_INFO("Loaded plugin: {}", name ? name : "<unknown>");
                        } else {
                            LOG_ERROR("Failed to load plugin: {}", name ? name : "<unknown>");
                            PyObject* get_traceback = PyObject_GetAttrString(plugins, "get_traceback");
                            if (get_traceback) {
                                PyObject* tb = PyObject_CallOneArg(get_traceback, key);
                                if (tb && !Py_IsNone(tb) && PyUnicode_Check(tb)) {
                                    LOG_ERROR("Traceback:\n{}", PyUnicode_AsUTF8(tb));
                                }
                                Py_XDECREF(tb);
                                Py_DECREF(get_traceback);
                            }
                        }
                    }
                } else if (!results) {
                    PyErr_Print();
                    LOG_ERROR("Plugin load_all() failed");
                }
                Py_XDECREF(results);
                Py_DECREF(load_all);
            }
            Py_DECREF(plugins);
        }

        Py_DECREF(lf);
    }
#endif

    std::filesystem::path get_user_packages_dir() {
        return PackageManager::instance().site_packages_dir();
    }

    void ensure_initialized() {
#ifdef LFS_BUILD_PYTHON_BINDINGS
        std::call_once(g_py_init_once, [] {
            if (!Py_IsInitialized()) {
                PyImport_AppendInittab("_lfs_output", init_capture_module);

                const auto python_home = lfs::core::getPythonHome();
                static std::wstring python_home_wstr;
                if (!python_home.empty()) {
                    python_home_wstr = python_home.wstring();
                    Py_SetPythonHome(python_home_wstr.c_str());
                    LOG_INFO("Set Python home: {}", lfs::core::path_to_utf8(python_home));
                }

                Py_Initialize();
                g_we_initialized_python = true;
                LOG_INFO("Python interpreter initialized by application");
            } else {
                LOG_WARN("Python already initialized by external code (e.g., .pyd loading)");
                g_we_initialized_python = false;
            }

            PyEval_InitThreads();
            register_output_module_post_init();

            // Add user site-packages to sys.path
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
                const auto user_packages_utf8 = lfs::core::path_to_utf8(user_packages);
                PyObject* py_path = PyUnicode_FromString(user_packages_utf8.c_str());
                PyList_Insert(sys_path, 0, py_path);
                Py_DECREF(py_path);
                LOG_INFO("Added user packages dir to Python path: {}", user_packages_utf8);

                const auto python_module_dir = lfs::core::getPythonModuleDir();
                if (!python_module_dir.empty()) {
                    const auto python_module_dir_utf8 = lfs::core::path_to_utf8(python_module_dir);
                    PyObject* const py_mod_path = PyUnicode_FromString(python_module_dir_utf8.c_str());
                    PyList_Insert(sys_path, 0, py_mod_path);
                    Py_DECREF(py_mod_path);
                    LOG_INFO("Added Python module dir to path: {}", python_module_dir_utf8);
                } else {
                    const auto exe_dir_utf8 = lfs::core::path_to_utf8(lfs::core::getExecutableDir());
                    LOG_WARN("Python module (lichtfeld.pyd) not found. Searched: {}/src/python, {}",
                             exe_dir_utf8, exe_dir_utf8);
                }
            }

            // Load plugins BEFORE releasing the GIL - the .pyd initialization code
            // needs valid thread state to access Python internals (PyImport_GetModuleDict etc.)
            LOG_INFO("Loading plugins while holding GIL...");
            ensure_plugins_loaded();
            LOG_INFO("Plugin loading complete");

            // Now release the GIL for other threads
            if (g_we_initialized_python) {
                g_main_thread_state = PyEval_SaveThread();
                LOG_INFO("GIL released (main thread state saved)");
            } else {
                g_main_thread_state = nullptr;
                LOG_INFO("External Python init - using PyGILState for GIL management");
            }
        });
#endif
    }

    void finalize() {
#ifdef LFS_BUILD_PYTHON_BINDINGS
        if (!Py_IsInitialized()) {
            return;
        }

        // Acquire GIL appropriately based on how Python was initialized
        if (g_main_thread_state) {
            // We initialized Python - restore main thread state
            PyEval_RestoreThread(g_main_thread_state);
            g_main_thread_state = nullptr;
        } else if (g_we_initialized_python) {
            // We initialized but thread state is null (shouldn't happen)
            LOG_WARN("finalize: unexpected null thread state");
            return;
        } else {
            // External initialization - use PyGILState
            PyGILState_Ensure();
        }

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
            const auto path_str = lfs::core::path_to_utf8(packages);
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

        // Add Python module directory (where lichtfeld.so lives) to sys.path
        {
            const auto python_module_dir = lfs::core::getPythonModuleDir();
            if (!python_module_dir.empty()) {
                const auto python_module_dir_utf8 = lfs::core::path_to_utf8(python_module_dir);
                PyObject* sys_path = PySys_GetObject("path"); // borrowed
                PyObject* py_path = PyUnicode_FromString(python_module_dir_utf8.c_str());
                PyList_Append(sys_path, py_path);
                Py_DECREF(py_path);
                LOG_DEBUG("Added {} to Python path", python_module_dir_utf8);
            }
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

        // Load plugins after lichtfeld is fully imported
        ensure_plugins_loaded();

        for (const auto& script : scripts) {
            const auto script_utf8 = lfs::core::path_to_utf8(script);
            if (!std::filesystem::exists(script)) {
                PyGILState_Release(gil_state);
                return std::unexpected(std::format("Python script not found: {}", script_utf8));
            }

            // Ensure script directory is on sys.path
            const auto parent_utf8 = lfs::core::path_to_utf8(script.parent_path());
            if (!parent_utf8.empty()) {
                PyObject* sys_path = PySys_GetObject("path"); // borrowed ref
                PyObject* py_parent = PyUnicode_FromString(parent_utf8.c_str());
                if (sys_path && py_parent) {
                    PyList_Append(sys_path, py_parent);
                }
                Py_XDECREF(py_parent);
            }

#ifdef _WIN32
            FILE* const fp = _wfopen(script.wstring().c_str(), L"r");
#else
            FILE* const fp = fopen(script.c_str(), "r");
#endif
            if (!fp) {
                PyGILState_Release(gil_state);
                return std::unexpected(std::format("Failed to open Python script: {}", script_utf8));
            }

            LOG_INFO("Executing Python script: {}", script_utf8);
            const int rc = PyRun_SimpleFileEx(fp, script_utf8.c_str(), /*closeit=*/1);
            if (rc != 0) {
                PyGILState_Release(gil_state);
                return std::unexpected(std::format("Python script failed: {} (rc={})", script_utf8, rc));
            }

            LOG_INFO("Python script completed: {}", script_utf8);
        }

        PyGILState_Release(gil_state);
        return {};
#endif
    }

    FormatResult format_python_code(const std::string& code) {
#ifndef LFS_BUILD_PYTHON_BINDINGS
        return {code, "", true};
#else
        if (code.empty())
            return {code, "", true};

        auto& pm = PackageManager::instance();
        if (!pm.is_installed("black")) {
            if (!pm.ensure_venv()) {
                LOG_ERROR("Failed to create venv for black");
                return {code, "Failed to create venv for black", false};
            }
            LOG_INFO("Installing black...");
            const auto install_result = pm.install("black");
            if (!install_result.success) {
                LOG_ERROR("Failed to install black: {}", install_result.error);
                return {code, install_result.error, false};
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
        return (None, f"ImportError: {e}")

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
        return (code, None)

    # Convert tabs to spaces consistently
    lines = [line.replace('\t', '    ') for line in lines]

    # Find indentation levels for all non-empty lines
    indents = []
    for line in lines:
        if line.strip():
            indents.append(len(line) - len(line.lstrip()))

    if not indents:
        return (code, None)

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
        return (black.format_str(cleaned, mode=black.Mode()), None)
    except Exception as e:
        return (None, str(e))
)";

        PyRun_SimpleString(FORMAT_CODE);

        PyObject* const main_module = PyImport_AddModule("__main__");
        if (!main_module) {
            PyGILState_Release(gil);
            return {code, "Failed to get __main__ module", false};
        }

        PyObject* const main_dict = PyModule_GetDict(main_module);
        PyObject* const format_func = PyDict_GetItemString(main_dict, "_lfs_format_code");
        if (!format_func || !PyCallable_Check(format_func)) {
            PyGILState_Release(gil);
            return {code, "Format function not found", false};
        }

        FormatResult result{code, "", false};
        PyObject* const py_code = PyUnicode_FromString(code.c_str());
        PyObject* const py_result = PyObject_CallFunctionObjArgs(format_func, py_code, nullptr);
        Py_DECREF(py_code);

        if (py_result && PyTuple_Check(py_result) && PyTuple_Size(py_result) == 2) {
            PyObject* formatted = PyTuple_GetItem(py_result, 0);
            PyObject* error = PyTuple_GetItem(py_result, 1);

            if (formatted && PyUnicode_Check(formatted)) {
                const char* const str = PyUnicode_AsUTF8(formatted);
                if (str) {
                    result.code = str;
                    result.success = true;
                }
            }

            if (error && !Py_IsNone(error) && PyUnicode_Check(error)) {
                const char* const err = PyUnicode_AsUTF8(error);
                if (err) {
                    result.error = err;
                    result.success = false;
                }
            }

            Py_DECREF(py_result);
        } else {
            if (PyErr_Occurred()) {
                PyErr_Clear();
            }
            result.error = "Format function returned unexpected result";
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

    CapabilityResult invoke_capability(const std::string& name, const std::string& args_json) {
#ifndef LFS_BUILD_PYTHON_BINDINGS
        return {false, "", "Python bindings disabled"};
#else
        ensure_initialized();
        const PyGILState_STATE gil = PyGILState_Ensure();
        CapabilityResult result;

        PyObject* lichtfeld = PyImport_ImportModule("lichtfeld");
        if (!lichtfeld) {
            PyErr_Print();
            PyGILState_Release(gil);
            return {false, "", "Failed to import lichtfeld"};
        }
        Py_DECREF(lichtfeld);

        ensure_plugins_loaded();

        PyObject* lfs_plugins = PyImport_ImportModule("lfs_plugins");
        if (!lfs_plugins) {
            PyErr_Print();
            PyGILState_Release(gil);
            return {false, "", "Failed to import lfs_plugins"};
        }

        PyObject* registry_class = PyObject_GetAttrString(lfs_plugins, "CapabilityRegistry");
        if (!registry_class) {
            Py_DECREF(lfs_plugins);
            PyGILState_Release(gil);
            return {false, "", "CapabilityRegistry not found"};
        }

        PyObject* instance_method = PyObject_GetAttrString(registry_class, "instance");
        PyObject* registry = PyObject_CallNoArgs(instance_method);
        Py_DECREF(instance_method);
        Py_DECREF(registry_class);

        if (!registry) {
            Py_DECREF(lfs_plugins);
            PyGILState_Release(gil);
            return {false, "", "Failed to get capability registry instance"};
        }

        PyObject* json_module = PyImport_ImportModule("json");
        PyObject* loads = PyObject_GetAttrString(json_module, "loads");
        PyObject* dumps = PyObject_GetAttrString(json_module, "dumps");
        PyObject* py_args_str = PyUnicode_FromString(args_json.c_str());
        PyObject* args_dict = PyObject_CallOneArg(loads, py_args_str);
        Py_DECREF(py_args_str);

        if (!args_dict) {
            PyErr_Clear();
            args_dict = PyDict_New();
        }

        PyObject* invoke_method = PyObject_GetAttrString(registry, "invoke");
        PyObject* py_name = PyUnicode_FromString(name.c_str());
        PyObject* py_result = PyObject_CallFunctionObjArgs(invoke_method, py_name, args_dict, nullptr);
        Py_DECREF(py_name);
        Py_DECREF(args_dict);
        Py_DECREF(invoke_method);

        if (py_result && PyDict_Check(py_result)) {
            PyObject* success = PyDict_GetItemString(py_result, "success");
            result.success = success && PyObject_IsTrue(success);

            if (!result.success) {
                PyObject* error = PyDict_GetItemString(py_result, "error");
                if (error && PyUnicode_Check(error)) {
                    result.error = PyUnicode_AsUTF8(error);
                }
            }

            PyObject* json_str = PyObject_CallOneArg(dumps, py_result);
            if (json_str) {
                result.result_json = PyUnicode_AsUTF8(json_str);
                Py_DECREF(json_str);
            }
            Py_DECREF(py_result);
        } else {
            if (PyErr_Occurred())
                PyErr_Print();
            result = {false, "", "Capability invocation failed"};
        }

        Py_DECREF(dumps);
        Py_DECREF(loads);
        Py_DECREF(json_module);
        Py_DECREF(registry);
        Py_DECREF(lfs_plugins);
        PyGILState_Release(gil);
        return result;
#endif
    }

    bool has_capability(const std::string& name) {
#ifndef LFS_BUILD_PYTHON_BINDINGS
        return false;
#else
        ensure_initialized();
        const PyGILState_STATE gil = PyGILState_Ensure();
        bool result = false;

        PyObject* lfs_plugins = PyImport_ImportModule("lfs_plugins");
        if (lfs_plugins) {
            PyObject* registry_class = PyObject_GetAttrString(lfs_plugins, "CapabilityRegistry");
            if (registry_class) {
                PyObject* instance_method = PyObject_GetAttrString(registry_class, "instance");
                PyObject* registry = PyObject_CallNoArgs(instance_method);
                if (registry) {
                    PyObject* has_method = PyObject_GetAttrString(registry, "has");
                    PyObject* py_name = PyUnicode_FromString(name.c_str());
                    PyObject* py_result = PyObject_CallOneArg(has_method, py_name);
                    if (py_result) {
                        result = PyObject_IsTrue(py_result);
                        Py_DECREF(py_result);
                    }
                    Py_DECREF(py_name);
                    Py_DECREF(has_method);
                    Py_DECREF(registry);
                }
                Py_DECREF(instance_method);
                Py_DECREF(registry_class);
            }
            Py_DECREF(lfs_plugins);
        }

        PyGILState_Release(gil);
        return result;
#endif
    }

    std::vector<CapabilityInfo> list_capabilities() {
        std::vector<CapabilityInfo> result;
#ifndef LFS_BUILD_PYTHON_BINDINGS
        return result;
#else
        ensure_initialized();
        const PyGILState_STATE gil = PyGILState_Ensure();

        PyObject* lfs_plugins = PyImport_ImportModule("lfs_plugins");
        if (lfs_plugins) {
            PyObject* registry_class = PyObject_GetAttrString(lfs_plugins, "CapabilityRegistry");
            if (registry_class) {
                PyObject* instance_method = PyObject_GetAttrString(registry_class, "instance");
                PyObject* registry = PyObject_CallNoArgs(instance_method);
                if (registry) {
                    PyObject* list_method = PyObject_GetAttrString(registry, "list_all");
                    PyObject* caps = PyObject_CallNoArgs(list_method);
                    if (caps && PyList_Check(caps)) {
                        const Py_ssize_t n = PyList_Size(caps);
                        for (Py_ssize_t i = 0; i < n; ++i) {
                            PyObject* cap = PyList_GetItem(caps, i);
                            CapabilityInfo info;

                            PyObject* name = PyObject_GetAttrString(cap, "name");
                            if (name && PyUnicode_Check(name))
                                info.name = PyUnicode_AsUTF8(name);
                            Py_XDECREF(name);

                            PyObject* desc = PyObject_GetAttrString(cap, "description");
                            if (desc && PyUnicode_Check(desc))
                                info.description = PyUnicode_AsUTF8(desc);
                            Py_XDECREF(desc);

                            PyObject* plugin = PyObject_GetAttrString(cap, "plugin_name");
                            if (plugin && PyUnicode_Check(plugin))
                                info.plugin_name = PyUnicode_AsUTF8(plugin);
                            Py_XDECREF(plugin);

                            result.push_back(info);
                        }
                        Py_DECREF(caps);
                    }
                    Py_DECREF(list_method);
                    Py_DECREF(registry);
                }
                Py_DECREF(instance_method);
                Py_DECREF(registry_class);
            }
            Py_DECREF(lfs_plugins);
        }

        PyGILState_Release(gil);
        return result;
#endif
    }

} // namespace lfs::python
