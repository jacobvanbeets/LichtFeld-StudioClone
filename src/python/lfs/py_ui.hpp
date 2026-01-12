/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "python/py_panel_registry.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace lfs::vis::gui {
    struct UIContext;
}

namespace lfs::python {

    // PanelSpace enum is defined in py_panel_registry.hpp

    // Context passed to poll() for conditional panel visibility
    class PyPanelContext {
    public:
        bool is_training() const;
        bool has_scene() const;
        bool has_selection() const;
        size_t num_gaussians() const;
        int iteration() const;
        float loss() const;
    };

    // UI layout object passed to draw() - wraps ImGui calls
    class PyUILayout {
    public:
        // Text
        void label(const std::string& text);
        void heading(const std::string& text);
        void text_colored(const std::string& text, std::tuple<float, float, float, float> color);
        void text_selectable(const std::string& text, float height = 0);
        void bullet_text(const std::string& text);

        // Buttons
        bool button(const std::string& label, std::tuple<float, float> size = {0, 0});
        bool small_button(const std::string& label);
        std::tuple<bool, bool> checkbox(const std::string& label, bool value);
        std::tuple<bool, int> radio_button(const std::string& label, int current, int value);

        // Sliders
        std::tuple<bool, float> slider_float(const std::string& label, float value, float min, float max);
        std::tuple<bool, int> slider_int(const std::string& label, int value, int min, int max);
        std::tuple<bool, std::tuple<float, float>> slider_float2(const std::string& label,
                                                                 std::tuple<float, float> value,
                                                                 float min, float max);
        std::tuple<bool, std::tuple<float, float, float>> slider_float3(const std::string& label,
                                                                        std::tuple<float, float, float> value,
                                                                        float min, float max);

        // Drags
        std::tuple<bool, float> drag_float(const std::string& label, float value,
                                           float speed = 1.0f, float min = 0.0f, float max = 0.0f);
        std::tuple<bool, int> drag_int(const std::string& label, int value,
                                       float speed = 1.0f, int min = 0, int max = 0);

        // Input
        std::tuple<bool, std::string> input_text(const std::string& label, const std::string& value);
        std::tuple<bool, float> input_float(const std::string& label, float value);
        std::tuple<bool, int> input_int(const std::string& label, int value);
        std::tuple<bool, std::string> path_input(const std::string& label, const std::string& value,
                                                 bool folder_mode = true, const std::string& dialog_title = "");

        // Color
        std::tuple<bool, std::tuple<float, float, float>> color_edit3(const std::string& label,
                                                                      std::tuple<float, float, float> color);
        std::tuple<bool, std::tuple<float, float, float, float>> color_edit4(const std::string& label,
                                                                             std::tuple<float, float, float, float> color);
        bool color_button(const std::string& label, std::tuple<float, float, float, float> color,
                          std::tuple<float, float> size = {0, 0});

        // Selection
        std::tuple<bool, int> combo(const std::string& label, int current_idx,
                                    const std::vector<std::string>& items);
        std::tuple<bool, int> listbox(const std::string& label, int current_idx,
                                      const std::vector<std::string>& items, int height_items = -1);

        // Layout
        void separator();
        void spacing();
        void same_line(float offset = 0.0f, float spacing = -1.0f);
        void new_line();
        void indent(float width = 0.0f);
        void unindent(float width = 0.0f);
        void set_next_item_width(float width);

        // Grouping
        void begin_group();
        void end_group();
        bool collapsing_header(const std::string& label, bool default_open = false);
        bool tree_node(const std::string& label);
        void tree_pop();

        // Tables
        bool begin_table(const std::string& id, int columns);
        void end_table();
        void table_next_row();
        void table_next_column();
        bool table_set_column_index(int column);

        // Misc
        void progress_bar(float fraction, const std::string& overlay = "");
        void set_tooltip(const std::string& text);
        bool is_item_hovered();
        bool is_item_clicked(int button = 0);
        void push_id(const std::string& id);
        void push_id_int(int id);
        void pop_id();

        // Window control (for floating panels)
        bool begin_window(const std::string& title, bool* open = nullptr);
        void end_window();

        // RNA-style property widget (auto-generates from metadata)
        // Returns (changed, new_value) - draws widget based on property type
        std::tuple<bool, nb::object> prop(nb::object data,
                                          const std::string& prop_id,
                                          std::optional<std::string> text = std::nullopt);
    };

    // Panel info stored in registry
    struct PyPanelInfo {
        nb::object panel_class;
        nb::object panel_instance;
        std::string label;
        PanelSpace space = PanelSpace::Floating;
        int order = 100;
        bool enabled = true;
    };

    // Singleton registry for Python panels
    class PyPanelRegistry {
    public:
        static PyPanelRegistry& instance();

        void register_panel(nb::object panel_class);
        void unregister_panel(nb::object panel_class);
        void unregister_all();

        void draw_panels(PanelSpace space);
        void draw_single_panel(const std::string& label);
        bool has_panels(PanelSpace space) const;

        std::vector<std::string> get_panel_names(PanelSpace space) const;
        void set_panel_enabled(const std::string& label, bool enabled);
        bool is_panel_enabled(const std::string& label) const;

    private:
        PyPanelRegistry() = default;
        ~PyPanelRegistry() = default;
        PyPanelRegistry(const PyPanelRegistry&) = delete;
        PyPanelRegistry& operator=(const PyPanelRegistry&) = delete;

        mutable std::mutex mutex_;
        std::vector<PyPanelInfo> panels_;
    };

    // Theme palette wrapper (read-only)
    struct PyThemePalette {
        std::tuple<float, float, float, float> background;
        std::tuple<float, float, float, float> surface;
        std::tuple<float, float, float, float> surface_bright;
        std::tuple<float, float, float, float> primary;
        std::tuple<float, float, float, float> primary_dim;
        std::tuple<float, float, float, float> secondary;
        std::tuple<float, float, float, float> text;
        std::tuple<float, float, float, float> text_dim;
        std::tuple<float, float, float, float> border;
        std::tuple<float, float, float, float> success;
        std::tuple<float, float, float, float> warning;
        std::tuple<float, float, float, float> error;
        std::tuple<float, float, float, float> info;
    };

    // Theme sizes wrapper (read-only)
    struct PyThemeSizes {
        float window_rounding;
        float frame_rounding;
        float popup_rounding;
        float scrollbar_rounding;
        float tab_rounding;
        float border_size;
        std::tuple<float, float> window_padding;
        std::tuple<float, float> frame_padding;
        std::tuple<float, float> item_spacing;
        float toolbar_button_size;
        float toolbar_padding;
        float toolbar_spacing;
    };

    // Theme wrapper (read-only)
    struct PyTheme {
        std::string name;
        PyThemePalette palette;
        PyThemeSizes sizes;
    };

    // Get current theme
    PyTheme get_current_theme();

    // Hook position enum (mirrors ui_hooks.hpp but with Python bindings)
    enum class PyHookPosition {
        Prepend, // Run before native content
        Append   // Run after native content
    };

    // UI Hook registry for Python callbacks
    class PyUIHookRegistry {
    public:
        static PyUIHookRegistry& instance();

        // Register a Python callback for a hook point
        void add_hook(const std::string& panel,
                      const std::string& section,
                      nb::object callback,
                      PyHookPosition position = PyHookPosition::Append);

        // Remove a specific hook
        void remove_hook(const std::string& panel,
                         const std::string& section,
                         nb::object callback);

        // Clear hooks for a panel/section
        void clear_hooks(const std::string& panel, const std::string& section = "");

        // Clear all hooks
        void clear_all();

        // Invoke hooks - called from C++ panels
        void invoke(const std::string& panel,
                    const std::string& section,
                    PyHookPosition position);

        // Check if hooks exist
        bool has_hooks(const std::string& panel, const std::string& section) const;

        // Get all registered hook points
        std::vector<std::string> get_hook_points() const;

    private:
        PyUIHookRegistry() = default;
        ~PyUIHookRegistry() = default;
        PyUIHookRegistry(const PyUIHookRegistry&) = delete;
        PyUIHookRegistry& operator=(const PyUIHookRegistry&) = delete;

        struct HookEntry {
            nb::object callback;
            PyHookPosition position;
        };

        mutable std::mutex mutex_;
        std::unordered_map<std::string, std::vector<HookEntry>> hooks_;
    };

    // Register UI classes with nanobind module
    void register_ui(nb::module_& m);

} // namespace lfs::python
