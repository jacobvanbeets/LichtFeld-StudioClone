/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_plugins.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace lfs::python {

    namespace {
        nb::object get_plugin_manager() {
            return nb::module_::import_("lfs_plugins").attr("PluginManager").attr("instance")();
        }
    } // namespace

    void register_plugins(nb::module_& m) {
        auto plugins = m.def_submodule("plugins", "Plugin management");

        plugins.def(
            "discover", []() { return get_plugin_manager().attr("discover")(); },
            "Discover plugins in ~/.lichtfeld/plugins/");

        plugins.def(
            "load", [](const std::string& name) { return nb::cast<bool>(get_plugin_manager().attr("load")(name)); },
            nb::arg("name"), "Load plugin");

        plugins.def(
            "unload", [](const std::string& name) { return nb::cast<bool>(get_plugin_manager().attr("unload")(name)); },
            nb::arg("name"), "Unload plugin");

        plugins.def(
            "reload", [](const std::string& name) { return nb::cast<bool>(get_plugin_manager().attr("reload")(name)); },
            nb::arg("name"), "Reload plugin");

        plugins.def(
            "load_all", []() { return get_plugin_manager().attr("load_all")(); }, "Load all auto_start plugins");

        plugins.def(
            "list_loaded", []() { return get_plugin_manager().attr("list_loaded")(); }, "List loaded plugins");

        plugins.def(
            "start_watcher", []() { get_plugin_manager().attr("start_watcher")(); }, "Start file watcher");

        plugins.def(
            "stop_watcher", []() { get_plugin_manager().attr("stop_watcher")(); }, "Stop file watcher");

        plugins.def(
            "get_state", [](const std::string& name) { return get_plugin_manager().attr("get_state")(name); },
            nb::arg("name"), "Get plugin state");

        plugins.def(
            "get_error", [](const std::string& name) { return get_plugin_manager().attr("get_error")(name); },
            nb::arg("name"), "Get plugin error");

        plugins.def(
            "get_traceback", [](const std::string& name) { return get_plugin_manager().attr("get_traceback")(name); },
            nb::arg("name"), "Get plugin error traceback");

        plugins.def(
            "install",
            [](const std::string& url, const bool auto_load) {
                return nb::cast<std::string>(get_plugin_manager().attr("install")(url, nb::none(), auto_load));
            },
            nb::arg("url"), nb::arg("auto_load") = true, "Install from GitHub URL");

        plugins.def(
            "update", [](const std::string& name) { return nb::cast<bool>(get_plugin_manager().attr("update")(name)); },
            nb::arg("name"), "Update plugin");

        plugins.def(
            "uninstall",
            [](const std::string& name) { return nb::cast<bool>(get_plugin_manager().attr("uninstall")(name)); },
            nb::arg("name"), "Uninstall plugin");

        try {
            nb::module_::import_("lfs_plugins").attr("register_builtin_panels")();
            get_plugin_manager().attr("load_all")();
        } catch (...) {
        }
    }

} // namespace lfs::python
