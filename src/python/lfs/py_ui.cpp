/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_ui.hpp"
#include "control/command_api.hpp"
#include "core/logger.hpp"
#include "core/property_registry.hpp"
#include "gui/utils/windows_utils.hpp"
#include "py_params.hpp"
#include "python/ui_hooks.hpp"
#include "visualizer/theme/theme.hpp"

#include <imgui.h>

namespace lfs::python {

    using lfs::training::CommandCenter;

    namespace {
        constexpr size_t INPUT_TEXT_BUFFER_SIZE = 1024;

        std::tuple<float, float, float, float> imvec4_to_tuple(const ImVec4& c) {
            return {c.x, c.y, c.z, c.w};
        }

        ImVec4 tuple_to_imvec4(const std::tuple<float, float, float, float>& t) {
            return {std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t)};
        }
    } // namespace

    // PyPanelContext implementation
    bool PyPanelContext::is_training() const {
        return CommandCenter::instance().snapshot().is_running;
    }

    bool PyPanelContext::has_scene() const {
        // Scene is always available in GUI mode
        return true;
    }

    bool PyPanelContext::has_selection() const {
        // TODO: Access scene selection state
        return false;
    }

    size_t PyPanelContext::num_gaussians() const {
        return CommandCenter::instance().snapshot().num_gaussians;
    }

    int PyPanelContext::iteration() const {
        return CommandCenter::instance().snapshot().iteration;
    }

    float PyPanelContext::loss() const {
        return CommandCenter::instance().snapshot().loss;
    }

    // PyUILayout implementation - Text
    void PyUILayout::label(const std::string& text) {
        ImGui::TextUnformatted(text.c_str());
    }

    void PyUILayout::heading(const std::string& text) {
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]); // Bold font
        ImGui::TextUnformatted(text.c_str());
        ImGui::PopFont();
    }

    void PyUILayout::text_colored(const std::string& text, std::tuple<float, float, float, float> color) {
        ImGui::TextColored(tuple_to_imvec4(color), "%s", text.c_str());
    }

    void PyUILayout::text_selectable(const std::string& text, const float height) {
        constexpr ImGuiInputTextFlags FLAGS = ImGuiInputTextFlags_ReadOnly;
        const float h = height > 0 ? height : ImGui::GetTextLineHeight() * 3;
        ImGui::InputTextMultiline("##selectable", const_cast<char*>(text.c_str()), text.size() + 1, ImVec2(-1, h), FLAGS);
    }

    void PyUILayout::bullet_text(const std::string& text) {
        ImGui::BulletText("%s", text.c_str());
    }

    bool PyUILayout::button(const std::string& label, std::tuple<float, float> size) {
        return ImGui::Button(label.c_str(), {std::get<0>(size), std::get<1>(size)});
    }

    bool PyUILayout::small_button(const std::string& label) {
        return ImGui::SmallButton(label.c_str());
    }

    std::tuple<bool, bool> PyUILayout::checkbox(const std::string& label, bool value) {
        bool v = value;
        bool changed = ImGui::Checkbox(label.c_str(), &v);
        return {changed, v};
    }

    std::tuple<bool, int> PyUILayout::radio_button(const std::string& label, int current, int value) {
        bool clicked = ImGui::RadioButton(label.c_str(), current == value);
        return {clicked, clicked ? value : current};
    }

    // Sliders
    std::tuple<bool, float> PyUILayout::slider_float(const std::string& label, float value, float min, float max) {
        float v = value;
        bool changed = ImGui::SliderFloat(label.c_str(), &v, min, max);
        return {changed, v};
    }

    std::tuple<bool, int> PyUILayout::slider_int(const std::string& label, int value, int min, int max) {
        int v = value;
        bool changed = ImGui::SliderInt(label.c_str(), &v, min, max);
        return {changed, v};
    }

    std::tuple<bool, std::tuple<float, float>> PyUILayout::slider_float2(
        const std::string& label, std::tuple<float, float> value, float min, float max) {
        float v[2] = {std::get<0>(value), std::get<1>(value)};
        bool changed = ImGui::SliderFloat2(label.c_str(), v, min, max);
        return {changed, {v[0], v[1]}};
    }

    std::tuple<bool, std::tuple<float, float, float>> PyUILayout::slider_float3(
        const std::string& label, std::tuple<float, float, float> value, float min, float max) {
        float v[3] = {std::get<0>(value), std::get<1>(value), std::get<2>(value)};
        bool changed = ImGui::SliderFloat3(label.c_str(), v, min, max);
        return {changed, {v[0], v[1], v[2]}};
    }

    // Drags
    std::tuple<bool, float> PyUILayout::drag_float(const std::string& label, float value,
                                                   float speed, float min, float max) {
        float v = value;
        bool changed = ImGui::DragFloat(label.c_str(), &v, speed, min, max);
        return {changed, v};
    }

    std::tuple<bool, int> PyUILayout::drag_int(const std::string& label, int value,
                                               float speed, int min, int max) {
        int v = value;
        bool changed = ImGui::DragInt(label.c_str(), &v, speed, min, max);
        return {changed, v};
    }

    // Input
    std::tuple<bool, std::string> PyUILayout::input_text(const std::string& label, const std::string& value) {
        char buffer[INPUT_TEXT_BUFFER_SIZE];
        std::strncpy(buffer, value.c_str(), sizeof(buffer) - 1);
        buffer[sizeof(buffer) - 1] = '\0';
        bool changed = ImGui::InputText(label.c_str(), buffer, sizeof(buffer));
        return {changed, std::string(buffer)};
    }

    std::tuple<bool, float> PyUILayout::input_float(const std::string& label, float value) {
        float v = value;
        bool changed = ImGui::InputFloat(label.c_str(), &v);
        return {changed, v};
    }

    std::tuple<bool, int> PyUILayout::input_int(const std::string& label, int value) {
        int v = value;
        bool changed = ImGui::InputInt(label.c_str(), &v);
        return {changed, v};
    }

    std::tuple<bool, std::string> PyUILayout::path_input(const std::string& label, const std::string& value,
                                                         const bool folder_mode, const std::string& dialog_title) {
        char buffer[INPUT_TEXT_BUFFER_SIZE];
        std::strncpy(buffer, value.c_str(), sizeof(buffer) - 1);
        buffer[sizeof(buffer) - 1] = '\0';

        const float available = ImGui::GetContentRegionAvail().x;
        const float button_width = ImGui::CalcTextSize("...").x + ImGui::GetStyle().FramePadding.x * 2;
        const float input_width = available - button_width - ImGui::GetStyle().ItemSpacing.x;

        ImGui::SetNextItemWidth(input_width);
        bool changed = ImGui::InputText(label.c_str(), buffer, sizeof(buffer));

        ImGui::SameLine();
        const std::string btn_id = "...##" + label + "_browse";
        if (ImGui::Button(btn_id.c_str())) {
            const std::filesystem::path start_path = value.empty() ? std::filesystem::path{} : std::filesystem::path{value};
            std::filesystem::path result;
            if (folder_mode) {
                result = lfs::vis::gui::SelectFolderDialog(
                    dialog_title.empty() ? "Select Folder" : dialog_title, start_path);
            } else {
                result = lfs::vis::gui::OpenImageFileDialog(start_path);
            }
            if (!result.empty()) {
                std::strncpy(buffer, result.string().c_str(), sizeof(buffer) - 1);
                buffer[sizeof(buffer) - 1] = '\0';
                changed = true;
            }
        }

        return {changed, std::string(buffer)};
    }

    // Color
    std::tuple<bool, std::tuple<float, float, float>> PyUILayout::color_edit3(
        const std::string& label, std::tuple<float, float, float> color) {
        float c[3] = {std::get<0>(color), std::get<1>(color), std::get<2>(color)};
        bool changed = ImGui::ColorEdit3(label.c_str(), c);
        return {changed, {c[0], c[1], c[2]}};
    }

    std::tuple<bool, std::tuple<float, float, float, float>> PyUILayout::color_edit4(
        const std::string& label, std::tuple<float, float, float, float> color) {
        float c[4] = {std::get<0>(color), std::get<1>(color), std::get<2>(color), std::get<3>(color)};
        bool changed = ImGui::ColorEdit4(label.c_str(), c);
        return {changed, {c[0], c[1], c[2], c[3]}};
    }

    bool PyUILayout::color_button(const std::string& label, std::tuple<float, float, float, float> color,
                                  std::tuple<float, float> size) {
        return ImGui::ColorButton(label.c_str(), tuple_to_imvec4(color), 0,
                                  {std::get<0>(size), std::get<1>(size)});
    }

    // Selection
    std::tuple<bool, int> PyUILayout::combo(const std::string& label, int current_idx,
                                            const std::vector<std::string>& items) {
        int idx = current_idx;
        bool changed = false;
        if (ImGui::BeginCombo(label.c_str(), (idx >= 0 && idx < static_cast<int>(items.size()))
                                                 ? items[idx].c_str()
                                                 : "")) {
            for (int i = 0; i < static_cast<int>(items.size()); ++i) {
                bool selected = (i == idx);
                if (ImGui::Selectable(items[i].c_str(), selected)) {
                    idx = i;
                    changed = true;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        return {changed, idx};
    }

    std::tuple<bool, int> PyUILayout::listbox(const std::string& label, int current_idx,
                                              const std::vector<std::string>& items, int height_items) {
        int idx = current_idx;
        bool changed = false;
        if (ImGui::BeginListBox(label.c_str(), {0, height_items > 0 ? height_items * ImGui::GetTextLineHeightWithSpacing() : 0})) {
            for (int i = 0; i < static_cast<int>(items.size()); ++i) {
                bool selected = (i == idx);
                if (ImGui::Selectable(items[i].c_str(), selected)) {
                    idx = i;
                    changed = true;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndListBox();
        }
        return {changed, idx};
    }

    // Layout
    void PyUILayout::separator() {
        ImGui::Separator();
    }

    void PyUILayout::spacing() {
        ImGui::Spacing();
    }

    void PyUILayout::same_line(float offset, float spacing) {
        ImGui::SameLine(offset, spacing);
    }

    void PyUILayout::new_line() {
        ImGui::NewLine();
    }

    void PyUILayout::indent(float width) {
        ImGui::Indent(width);
    }

    void PyUILayout::unindent(float width) {
        ImGui::Unindent(width);
    }

    void PyUILayout::set_next_item_width(float width) {
        ImGui::SetNextItemWidth(width);
    }

    // Grouping
    void PyUILayout::begin_group() {
        ImGui::BeginGroup();
    }

    void PyUILayout::end_group() {
        ImGui::EndGroup();
    }

    bool PyUILayout::collapsing_header(const std::string& label, bool default_open) {
        return ImGui::CollapsingHeader(label.c_str(), default_open ? ImGuiTreeNodeFlags_DefaultOpen : 0);
    }

    bool PyUILayout::tree_node(const std::string& label) {
        return ImGui::TreeNode(label.c_str());
    }

    void PyUILayout::tree_pop() {
        ImGui::TreePop();
    }

    // Tables
    bool PyUILayout::begin_table(const std::string& id, int columns) {
        return ImGui::BeginTable(id.c_str(), columns);
    }

    void PyUILayout::end_table() {
        ImGui::EndTable();
    }

    void PyUILayout::table_next_row() {
        ImGui::TableNextRow();
    }

    void PyUILayout::table_next_column() {
        ImGui::TableNextColumn();
    }

    bool PyUILayout::table_set_column_index(int column) {
        return ImGui::TableSetColumnIndex(column);
    }

    // Misc
    void PyUILayout::progress_bar(float fraction, const std::string& overlay) {
        ImGui::ProgressBar(fraction, {-FLT_MIN, 0}, overlay.empty() ? nullptr : overlay.c_str());
    }

    void PyUILayout::set_tooltip(const std::string& text) {
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", text.c_str());
        }
    }

    bool PyUILayout::is_item_hovered() {
        return ImGui::IsItemHovered();
    }

    bool PyUILayout::is_item_clicked(int button) {
        return ImGui::IsItemClicked(button);
    }

    void PyUILayout::push_id(const std::string& id) {
        ImGui::PushID(id.c_str());
    }

    void PyUILayout::push_id_int(int id) {
        ImGui::PushID(id);
    }

    void PyUILayout::pop_id() {
        ImGui::PopID();
    }

    // Window control
    bool PyUILayout::begin_window(const std::string& title, bool* open) {
        return ImGui::Begin(title.c_str(), open);
    }

    void PyUILayout::end_window() {
        ImGui::End();
    }

    // RNA-style property widget
    std::tuple<bool, nb::object> PyUILayout::prop(nb::object data,
                                                  const std::string& prop_id,
                                                  std::optional<std::string> text) {
        using namespace lfs::core::prop;

        // Check if data is PyOptimizationParams
        if (!nb::isinstance<PyOptimizationParams>(data)) {
            throw std::runtime_error("prop() currently only supports OptimizationParams");
        }

        auto& params = nb::cast<PyOptimizationParams&>(data);
        auto* meta = PropertyRegistry::instance().get_property("optimization", prop_id);
        if (!meta) {
            throw std::runtime_error("Unknown property: " + prop_id);
        }

        std::string display_name = text.value_or(meta->name);
        bool changed = false;
        nb::object new_value;

        switch (meta->type) {
        case PropType::Float: {
            float current = nb::cast<float>(params.get(prop_id));
            auto [c, nv] = (meta->ui_hint == PropUIHint::Slider)
                               ? slider_float(display_name, current,
                                              static_cast<float>(meta->min_value),
                                              static_cast<float>(meta->max_value))
                               : drag_float(display_name, current,
                                            static_cast<float>(meta->step),
                                            static_cast<float>(meta->min_value),
                                            static_cast<float>(meta->max_value));
            changed = c;
            new_value = nb::cast(nv);
            break;
        }
        case PropType::Int: {
            int current = nb::cast<int>(params.get(prop_id));
            auto [c, nv] = (meta->ui_hint == PropUIHint::Slider)
                               ? slider_int(display_name, current,
                                            static_cast<int>(meta->min_value),
                                            static_cast<int>(meta->max_value))
                               : drag_int(display_name, current, 1.0f,
                                          static_cast<int>(meta->min_value),
                                          static_cast<int>(meta->max_value));
            changed = c;
            new_value = nb::cast(nv);
            break;
        }
        case PropType::SizeT: {
            int current = static_cast<int>(nb::cast<size_t>(params.get(prop_id)));
            auto [c, nv] = drag_int(display_name, current, 1.0f,
                                    static_cast<int>(meta->min_value),
                                    static_cast<int>(meta->max_value));
            changed = c;
            new_value = nb::cast(static_cast<size_t>(nv));
            break;
        }
        case PropType::Bool: {
            bool current = nb::cast<bool>(params.get(prop_id));
            auto [c, nv] = checkbox(display_name, current);
            changed = c;
            new_value = nb::cast(nv);
            break;
        }
        case PropType::String: {
            std::string current = nb::cast<std::string>(params.get(prop_id));
            auto [c, nv] = input_text(display_name, current);
            changed = c;
            new_value = nb::cast(nv);
            break;
        }
        case PropType::Enum: {
            int current = nb::cast<int>(params.get(prop_id));
            std::vector<std::string> items;
            int current_idx = 0;
            for (size_t i = 0; i < meta->enum_items.size(); ++i) {
                items.push_back(meta->enum_items[i].name);
                if (meta->enum_items[i].value == current) {
                    current_idx = static_cast<int>(i);
                }
            }
            auto [c, new_idx] = combo(display_name, current_idx, items);
            changed = c;
            if (changed && new_idx >= 0 && new_idx < static_cast<int>(meta->enum_items.size())) {
                new_value = nb::cast(meta->enum_items[new_idx].value);
            } else {
                new_value = nb::cast(current);
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported property type for prop()");
        }

        // Show tooltip on hover
        if (!meta->description.empty() && is_item_hovered()) {
            set_tooltip(meta->description);
        }

        // Update property if changed
        if (changed && !meta->is_readonly()) {
            params.set(prop_id, new_value);
        }

        return {changed, new_value};
    }

    // PyPanelRegistry implementation
    PyPanelRegistry& PyPanelRegistry::instance() {
        static PyPanelRegistry registry;
        return registry;
    }

    void PyPanelRegistry::register_panel(nb::object panel_class) {
        std::lock_guard lock(mutex_);

        // Helper to get attribute with panel_ prefix, falling back to bl_ for backward compat
        auto get_panel_attr = [&panel_class](const char* panel_name,
                                             const char* bl_name) -> std::optional<nb::object> {
            if (nb::hasattr(panel_class, panel_name)) {
                return panel_class.attr(panel_name);
            }
            if (nb::hasattr(panel_class, bl_name)) {
                return panel_class.attr(bl_name);
            }
            return std::nullopt;
        };

        // Extract panel attributes (panel_ prefix, bl_ for backward compat)
        std::string label = "Python Panel";
        PanelSpace space = PanelSpace::Floating;
        int order = 100;
        std::string category;

        if (auto v = get_panel_attr("panel_label", "bl_label")) {
            label = nb::cast<std::string>(*v);
        }

        if (auto v = get_panel_attr("panel_space", "bl_space")) {
            std::string space_str = nb::cast<std::string>(*v);
            if (space_str == "SIDE_PANEL") {
                space = PanelSpace::SidePanel;
            } else if (space_str == "FLOATING") {
                space = PanelSpace::Floating;
            } else if (space_str == "VIEWPORT_OVERLAY") {
                space = PanelSpace::ViewportOverlay;
            }
        }

        if (auto v = get_panel_attr("panel_order", "bl_order")) {
            order = nb::cast<int>(*v);
        }

        if (auto v = get_panel_attr("panel_category", "bl_category")) {
            category = nb::cast<std::string>(*v);
        }

        for (auto& p : panels_) {
            if (p.label == label) {
                p.panel_class = panel_class;
                p.panel_instance = panel_class();
                p.space = space;
                p.order = order;
                return;
            }
        }

        // Create panel instance
        nb::object instance = panel_class();

        PyPanelInfo info;
        info.panel_class = panel_class;
        info.panel_instance = instance;
        info.label = label;
        info.space = space;
        info.order = order;
        info.enabled = true;

        panels_.push_back(std::move(info));
        std::sort(panels_.begin(), panels_.end(),
                  [](const PyPanelInfo& a, const PyPanelInfo& b) { return a.order < b.order; });
    }

    void PyPanelRegistry::unregister_panel(nb::object panel_class) {
        std::lock_guard lock(mutex_);

        std::string label;
        if (nb::hasattr(panel_class, "panel_label")) {
            label = nb::cast<std::string>(panel_class.attr("panel_label"));
        } else if (nb::hasattr(panel_class, "bl_label")) {
            label = nb::cast<std::string>(panel_class.attr("bl_label"));
        }

        panels_.erase(std::remove_if(panels_.begin(), panels_.end(),
                                     [&label](const PyPanelInfo& p) { return p.label == label; }),
                      panels_.end());
    }

    void PyPanelRegistry::unregister_all() {
        std::lock_guard lock(mutex_);
        panels_.clear();
    }

    void PyPanelRegistry::draw_panels(PanelSpace space) {
        std::vector<PyPanelInfo> panels_copy;
        {
            std::lock_guard lock(mutex_);
            panels_copy = panels_;
        }

        if (panels_copy.empty()) {
            return;
        }

        nb::gil_scoped_acquire gil;

        for (auto& panel : panels_copy) {
            if (panel.space != space || !panel.enabled) {
                continue;
            }

            try {
                // Check poll() if defined
                bool should_draw = true;
                if (nb::hasattr(panel.panel_class, "poll")) {
                    PyPanelContext ctx;
                    should_draw = nb::cast<bool>(panel.panel_class.attr("poll")(ctx));
                }

                if (!should_draw) {
                    continue;
                }

                // Draw based on panel space
                if (space == PanelSpace::Floating) {
                    bool open = true;
                    if (ImGui::Begin(panel.label.c_str(), &open)) {
                        PyUILayout layout;
                        panel.panel_instance.attr("draw")(layout);
                    }
                    ImGui::End();

                    if (!open) {
                        // Panel was closed
                        std::lock_guard lock(mutex_);
                        for (auto& p : panels_) {
                            if (p.label == panel.label) {
                                p.enabled = false;
                                break;
                            }
                        }
                    }
                } else if (space == PanelSpace::SidePanel) {
                    if (ImGui::CollapsingHeader(panel.label.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                        PyUILayout layout;
                        panel.panel_instance.attr("draw")(layout);
                    }
                } else if (space == PanelSpace::ViewportOverlay) {
                    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                                             ImGuiWindowFlags_NoResize |
                                             ImGuiWindowFlags_NoMove |
                                             ImGuiWindowFlags_NoScrollbar |
                                             ImGuiWindowFlags_NoSavedSettings |
                                             ImGuiWindowFlags_AlwaysAutoResize |
                                             ImGuiWindowFlags_NoFocusOnAppearing |
                                             ImGuiWindowFlags_NoBringToFrontOnFocus;

                    ImGui::SetNextWindowBgAlpha(0.5f);
                    if (ImGui::Begin(panel.label.c_str(), nullptr, flags)) {
                        PyUILayout layout;
                        panel.panel_instance.attr("draw")(layout);
                    }
                    ImGui::End();
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Python panel '{}' error: {}", panel.label, e.what());
            }
        }
    }

    bool PyPanelRegistry::has_panels(PanelSpace space) const {
        std::lock_guard lock(mutex_);
        for (const auto& p : panels_) {
            if (p.space == space && p.enabled) {
                return true;
            }
        }
        return false;
    }

    void PyPanelRegistry::draw_single_panel(const std::string& label) {
        PyPanelInfo panel_copy;
        bool found = false;
        {
            std::lock_guard lock(mutex_);
            for (const auto& panel : panels_) {
                if (panel.label == label && panel.enabled) {
                    panel_copy = panel;
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            LOG_DEBUG("Panel '{}' not found or disabled", label);
            return;
        }

        nb::gil_scoped_acquire gil;

        try {
            PyUILayout layout;
            panel_copy.panel_instance.attr("draw")(layout);
        } catch (const nb::python_error& e) {
            LOG_ERROR("Python panel '{}' Python error: {}", panel_copy.label, e.what());
        } catch (const std::exception& e) {
            LOG_ERROR("Python panel '{}' error: {}", panel_copy.label, e.what());
        }
    }

    std::vector<std::string> PyPanelRegistry::get_panel_names(PanelSpace space) const {
        std::lock_guard lock(mutex_);
        std::vector<std::string> names;
        for (const auto& p : panels_) {
            if (p.space == space) {
                names.push_back(p.label);
            }
        }
        return names;
    }

    void PyPanelRegistry::set_panel_enabled(const std::string& label, bool enabled) {
        std::lock_guard lock(mutex_);
        for (auto& p : panels_) {
            if (p.label == label) {
                p.enabled = enabled;
                return;
            }
        }
    }

    bool PyPanelRegistry::is_panel_enabled(const std::string& label) const {
        std::lock_guard lock(mutex_);
        for (const auto& p : panels_) {
            if (p.label == label) {
                return p.enabled;
            }
        }
        return false;
    }

    // Theme access
    PyTheme get_current_theme() {
        const auto& t = lfs::vis::theme();

        PyTheme py_theme;
        py_theme.name = t.name;

        // Palette
        py_theme.palette.background = imvec4_to_tuple(t.palette.background);
        py_theme.palette.surface = imvec4_to_tuple(t.palette.surface);
        py_theme.palette.surface_bright = imvec4_to_tuple(t.palette.surface_bright);
        py_theme.palette.primary = imvec4_to_tuple(t.palette.primary);
        py_theme.palette.primary_dim = imvec4_to_tuple(t.palette.primary_dim);
        py_theme.palette.secondary = imvec4_to_tuple(t.palette.secondary);
        py_theme.palette.text = imvec4_to_tuple(t.palette.text);
        py_theme.palette.text_dim = imvec4_to_tuple(t.palette.text_dim);
        py_theme.palette.border = imvec4_to_tuple(t.palette.border);
        py_theme.palette.success = imvec4_to_tuple(t.palette.success);
        py_theme.palette.warning = imvec4_to_tuple(t.palette.warning);
        py_theme.palette.error = imvec4_to_tuple(t.palette.error);
        py_theme.palette.info = imvec4_to_tuple(t.palette.info);

        // Sizes
        py_theme.sizes.window_rounding = t.sizes.window_rounding;
        py_theme.sizes.frame_rounding = t.sizes.frame_rounding;
        py_theme.sizes.popup_rounding = t.sizes.popup_rounding;
        py_theme.sizes.scrollbar_rounding = t.sizes.scrollbar_rounding;
        py_theme.sizes.tab_rounding = t.sizes.tab_rounding;
        py_theme.sizes.border_size = t.sizes.border_size;
        py_theme.sizes.window_padding = {t.sizes.window_padding.x, t.sizes.window_padding.y};
        py_theme.sizes.frame_padding = {t.sizes.frame_padding.x, t.sizes.frame_padding.y};
        py_theme.sizes.item_spacing = {t.sizes.item_spacing.x, t.sizes.item_spacing.y};
        py_theme.sizes.toolbar_button_size = t.sizes.toolbar_button_size;
        py_theme.sizes.toolbar_padding = t.sizes.toolbar_padding;
        py_theme.sizes.toolbar_spacing = t.sizes.toolbar_spacing;

        return py_theme;
    }

    // PyUIHookRegistry implementation
    PyUIHookRegistry& PyUIHookRegistry::instance() {
        static PyUIHookRegistry registry;
        return registry;
    }

    void PyUIHookRegistry::add_hook(const std::string& panel,
                                    const std::string& section,
                                    nb::object callback,
                                    PyHookPosition position) {
        std::lock_guard lock(mutex_);

        const std::string key = panel + ":" + section;

        HookEntry entry;
        entry.callback = std::move(callback);
        entry.position = position;

        hooks_[key].push_back(std::move(entry));
    }

    void PyUIHookRegistry::remove_hook(const std::string& panel,
                                       const std::string& section,
                                       nb::object callback) {
        std::lock_guard lock(mutex_);

        const std::string key = panel + ":" + section;
        auto it = hooks_.find(key);
        if (it == hooks_.end()) {
            return;
        }

        auto& hooks = it->second;
        hooks.erase(std::remove_if(hooks.begin(), hooks.end(),
                                   [&callback](const HookEntry& entry) {
                                       return entry.callback.is(callback);
                                   }),
                    hooks.end());
    }

    void PyUIHookRegistry::clear_hooks(const std::string& panel, const std::string& section) {
        std::lock_guard lock(mutex_);

        if (section.empty()) {
            const std::string prefix = panel + ":";
            std::erase_if(hooks_, [&prefix](const auto& kv) { return kv.first.starts_with(prefix); });
        } else {
            hooks_.erase(panel + ":" + section);
        }
    }

    void PyUIHookRegistry::clear_all() {
        std::lock_guard lock(mutex_);
        hooks_.clear();
    }

    void PyUIHookRegistry::invoke(const std::string& panel,
                                  const std::string& section,
                                  PyHookPosition position) {
        std::vector<nb::object> callbacks_to_invoke;

        {
            std::lock_guard lock(mutex_);

            const std::string key = panel + ":" + section;
            auto it = hooks_.find(key);
            if (it == hooks_.end()) {
                return;
            }

            for (const auto& entry : it->second) {
                if (entry.position == position) {
                    callbacks_to_invoke.push_back(entry.callback);
                }
            }
        }

        if (callbacks_to_invoke.empty()) {
            return;
        }

        nb::gil_scoped_acquire gil;

        for (const auto& cb : callbacks_to_invoke) {
            try {
                PyUILayout layout;
                cb(layout);
            } catch (const std::exception& e) {
                LOG_ERROR("UI hook {}:{} error: {}", panel, section, e.what());
            }
        }
    }

    bool PyUIHookRegistry::has_hooks(const std::string& panel, const std::string& section) const {
        std::lock_guard lock(mutex_);

        const std::string key = panel + ":" + section;
        auto it = hooks_.find(key);
        return it != hooks_.end() && !it->second.empty();
    }

    std::vector<std::string> PyUIHookRegistry::get_hook_points() const {
        std::lock_guard lock(mutex_);

        std::vector<std::string> points;
        points.reserve(hooks_.size());
        for (const auto& [key, hooks] : hooks_) {
            if (!hooks.empty()) {
                points.push_back(key);
            }
        }
        return points;
    }

    // Register UI classes with nanobind module
    void register_ui(nb::module_& m) {
        // PanelSpace enum
        nb::enum_<PanelSpace>(m, "PanelSpace")
            .value("SIDE_PANEL", PanelSpace::SidePanel)
            .value("FLOATING", PanelSpace::Floating)
            .value("VIEWPORT_OVERLAY", PanelSpace::ViewportOverlay);

        // PyPanelContext
        nb::class_<PyPanelContext>(m, "PanelContext")
            .def(nb::init<>())
            .def_prop_ro("is_training", &PyPanelContext::is_training)
            .def_prop_ro("has_scene", &PyPanelContext::has_scene)
            .def_prop_ro("has_selection", &PyPanelContext::has_selection)
            .def_prop_ro("num_gaussians", &PyPanelContext::num_gaussians)
            .def_prop_ro("iteration", &PyPanelContext::iteration)
            .def_prop_ro("loss", &PyPanelContext::loss);

        // PyUILayout
        nb::class_<PyUILayout>(m, "UILayout")
            .def(nb::init<>())
            // Text
            .def("label", &PyUILayout::label, nb::arg("text"))
            .def("heading", &PyUILayout::heading, nb::arg("text"))
            .def("text_colored", &PyUILayout::text_colored, nb::arg("text"), nb::arg("color"))
            .def("text_selectable", &PyUILayout::text_selectable, nb::arg("text"), nb::arg("height") = 0.0f)
            .def("bullet_text", &PyUILayout::bullet_text, nb::arg("text"))
            // Buttons
            .def("button", &PyUILayout::button, nb::arg("label"), nb::arg("size") = std::make_tuple(0.0f, 0.0f))
            .def("small_button", &PyUILayout::small_button, nb::arg("label"))
            .def("checkbox", &PyUILayout::checkbox, nb::arg("label"), nb::arg("value"))
            .def("radio_button", &PyUILayout::radio_button, nb::arg("label"), nb::arg("current"), nb::arg("value"))
            // Sliders
            .def("slider_float", &PyUILayout::slider_float, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"))
            .def("slider_int", &PyUILayout::slider_int, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"))
            .def("slider_float2", &PyUILayout::slider_float2, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"))
            .def("slider_float3", &PyUILayout::slider_float3, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"))
            // Drags
            .def("drag_float", &PyUILayout::drag_float, nb::arg("label"), nb::arg("value"),
                 nb::arg("speed") = 1.0f, nb::arg("min") = 0.0f, nb::arg("max") = 0.0f)
            .def("drag_int", &PyUILayout::drag_int, nb::arg("label"), nb::arg("value"),
                 nb::arg("speed") = 1.0f, nb::arg("min") = 0, nb::arg("max") = 0)
            // Input
            .def("input_text", &PyUILayout::input_text, nb::arg("label"), nb::arg("value"))
            .def("input_float", &PyUILayout::input_float, nb::arg("label"), nb::arg("value"))
            .def("input_int", &PyUILayout::input_int, nb::arg("label"), nb::arg("value"))
            .def("path_input", &PyUILayout::path_input, nb::arg("label"), nb::arg("value"),
                 nb::arg("folder_mode") = true, nb::arg("dialog_title") = "")
            // Color
            .def("color_edit3", &PyUILayout::color_edit3, nb::arg("label"), nb::arg("color"))
            .def("color_edit4", &PyUILayout::color_edit4, nb::arg("label"), nb::arg("color"))
            .def("color_button", &PyUILayout::color_button, nb::arg("label"), nb::arg("color"),
                 nb::arg("size") = std::make_tuple(0.0f, 0.0f))
            // Selection
            .def("combo", &PyUILayout::combo, nb::arg("label"), nb::arg("current_idx"), nb::arg("items"))
            .def("listbox", &PyUILayout::listbox, nb::arg("label"), nb::arg("current_idx"),
                 nb::arg("items"), nb::arg("height_items") = -1)
            // Layout
            .def("separator", &PyUILayout::separator)
            .def("spacing", &PyUILayout::spacing)
            .def("same_line", &PyUILayout::same_line, nb::arg("offset") = 0.0f, nb::arg("spacing") = -1.0f)
            .def("new_line", &PyUILayout::new_line)
            .def("indent", &PyUILayout::indent, nb::arg("width") = 0.0f)
            .def("unindent", &PyUILayout::unindent, nb::arg("width") = 0.0f)
            .def("set_next_item_width", &PyUILayout::set_next_item_width, nb::arg("width"))
            // Grouping
            .def("begin_group", &PyUILayout::begin_group)
            .def("end_group", &PyUILayout::end_group)
            .def("collapsing_header", &PyUILayout::collapsing_header, nb::arg("label"), nb::arg("default_open") = false)
            .def("tree_node", &PyUILayout::tree_node, nb::arg("label"))
            .def("tree_pop", &PyUILayout::tree_pop)
            // Tables
            .def("begin_table", &PyUILayout::begin_table, nb::arg("id"), nb::arg("columns"))
            .def("end_table", &PyUILayout::end_table)
            .def("table_next_row", &PyUILayout::table_next_row)
            .def("table_next_column", &PyUILayout::table_next_column)
            .def("table_set_column_index", &PyUILayout::table_set_column_index, nb::arg("column"))
            // Misc
            .def("progress_bar", &PyUILayout::progress_bar, nb::arg("fraction"), nb::arg("overlay") = "")
            .def("set_tooltip", &PyUILayout::set_tooltip, nb::arg("text"))
            .def("is_item_hovered", &PyUILayout::is_item_hovered)
            .def("is_item_clicked", &PyUILayout::is_item_clicked, nb::arg("button") = 0)
            .def("push_id", &PyUILayout::push_id, nb::arg("id"))
            .def("push_id_int", &PyUILayout::push_id_int, nb::arg("id"))
            .def("pop_id", &PyUILayout::pop_id)
            // Window
            .def("begin_window", &PyUILayout::begin_window, nb::arg("title"), nb::arg("open") = nullptr)
            .def("end_window", &PyUILayout::end_window)
            // RNA-style property widget
            .def("prop", &PyUILayout::prop, nb::arg("data"), nb::arg("prop_id"),
                 nb::arg("text") = nb::none(),
                 "Draw a property widget based on metadata (auto-selects widget type)");

        // Theme types
        nb::class_<PyThemePalette>(m, "ThemePalette")
            .def_ro("background", &PyThemePalette::background)
            .def_ro("surface", &PyThemePalette::surface)
            .def_ro("surface_bright", &PyThemePalette::surface_bright)
            .def_ro("primary", &PyThemePalette::primary)
            .def_ro("primary_dim", &PyThemePalette::primary_dim)
            .def_ro("secondary", &PyThemePalette::secondary)
            .def_ro("text", &PyThemePalette::text)
            .def_ro("text_dim", &PyThemePalette::text_dim)
            .def_ro("border", &PyThemePalette::border)
            .def_ro("success", &PyThemePalette::success)
            .def_ro("warning", &PyThemePalette::warning)
            .def_ro("error", &PyThemePalette::error)
            .def_ro("info", &PyThemePalette::info);

        nb::class_<PyThemeSizes>(m, "ThemeSizes")
            .def_ro("window_rounding", &PyThemeSizes::window_rounding)
            .def_ro("frame_rounding", &PyThemeSizes::frame_rounding)
            .def_ro("popup_rounding", &PyThemeSizes::popup_rounding)
            .def_ro("scrollbar_rounding", &PyThemeSizes::scrollbar_rounding)
            .def_ro("tab_rounding", &PyThemeSizes::tab_rounding)
            .def_ro("border_size", &PyThemeSizes::border_size)
            .def_ro("window_padding", &PyThemeSizes::window_padding)
            .def_ro("frame_padding", &PyThemeSizes::frame_padding)
            .def_ro("item_spacing", &PyThemeSizes::item_spacing)
            .def_ro("toolbar_button_size", &PyThemeSizes::toolbar_button_size)
            .def_ro("toolbar_padding", &PyThemeSizes::toolbar_padding)
            .def_ro("toolbar_spacing", &PyThemeSizes::toolbar_spacing);

        nb::class_<PyTheme>(m, "Theme")
            .def_ro("name", &PyTheme::name)
            .def_ro("palette", &PyTheme::palette)
            .def_ro("sizes", &PyTheme::sizes);

        // Theme accessor
        m.def("theme", &get_current_theme, "Get the current theme");

        // Panel class (base for Python panels)
        m.def(
            "Panel", []() {
                return nb::make_tuple(); // Placeholder - panels inherit from object directly
            },
            "Base class marker for Python panels");

        // Panel registration functions
        m.def(
            "register_panel", [](nb::object panel_class) {
                PyPanelRegistry::instance().register_panel(panel_class);
            },
            nb::arg("panel_class"), "Register a Python panel class");

        m.def(
            "unregister_panel", [](nb::object panel_class) {
                PyPanelRegistry::instance().unregister_panel(panel_class);
            },
            nb::arg("panel_class"), "Unregister a Python panel class");

        m.def(
            "unregister_all_panels", []() {
                PyPanelRegistry::instance().unregister_all();
            },
            "Unregister all Python panels");

        m.def(
            "get_panel_names", [](const std::string& space) {
                PanelSpace ps = PanelSpace::Floating;
                if (space == "SIDE_PANEL") {
                    ps = PanelSpace::SidePanel;
                } else if (space == "VIEWPORT_OVERLAY") {
                    ps = PanelSpace::ViewportOverlay;
                }
                return PyPanelRegistry::instance().get_panel_names(ps);
            },
            nb::arg("space") = "FLOATING", "Get names of registered panels in a space");

        m.def(
            "set_panel_enabled", [](const std::string& label, bool enabled) {
                PyPanelRegistry::instance().set_panel_enabled(label, enabled);
            },
            nb::arg("label"), nb::arg("enabled"), "Enable or disable a panel");

        m.def(
            "is_panel_enabled", [](const std::string& label) {
                return PyPanelRegistry::instance().is_panel_enabled(label);
            },
            nb::arg("label"), "Check if a panel is enabled");

        // Hook position enum
        nb::enum_<PyHookPosition>(m, "HookPosition")
            .value("PREPEND", PyHookPosition::Prepend)
            .value("APPEND", PyHookPosition::Append);

        // Hook registration functions
        m.def(
            "add_hook",
            [](const std::string& panel, const std::string& section,
               nb::object callback, const std::string& position) {
                PyHookPosition pos = PyHookPosition::Append;
                if (position == "prepend" || position == "PREPEND") {
                    pos = PyHookPosition::Prepend;
                }
                PyUIHookRegistry::instance().add_hook(panel, section, callback, pos);
            },
            nb::arg("panel"), nb::arg("section"), nb::arg("callback"),
            nb::arg("position") = "append",
            "Add a UI hook callback for a panel section");

        m.def(
            "remove_hook",
            [](const std::string& panel, const std::string& section, nb::object callback) {
                PyUIHookRegistry::instance().remove_hook(panel, section, callback);
            },
            nb::arg("panel"), nb::arg("section"), nb::arg("callback"),
            "Remove a UI hook callback");

        m.def(
            "clear_hooks",
            [](const std::string& panel, const std::string& section) {
                PyUIHookRegistry::instance().clear_hooks(panel, section);
            },
            nb::arg("panel"), nb::arg("section") = "",
            "Clear UI hooks for a panel/section");

        m.def(
            "clear_all_hooks", []() {
                PyUIHookRegistry::instance().clear_all();
            },
            "Clear all UI hooks");

        m.def(
            "get_hook_points", []() {
                return PyUIHookRegistry::instance().get_hook_points();
            },
            "Get list of registered hook points");

        // Decorator-style hook registration
        m.def(
            "hook",
            [](const std::string& panel, const std::string& section,
               const std::string& position) {
                // Returns a decorator function
                return nb::cpp_function([panel, section, position](nb::object func) {
                    PyHookPosition pos = PyHookPosition::Append;
                    if (position == "prepend" || position == "PREPEND") {
                        pos = PyHookPosition::Prepend;
                    }
                    PyUIHookRegistry::instance().add_hook(panel, section, func, pos);
                    return func; // Return the function unchanged
                });
            },
            nb::arg("panel"), nb::arg("section"), nb::arg("position") = "append",
            "Decorator to register a function as a UI hook");

        // File dialogs
        m.def(
            "open_image_dialog",
            [](const std::string& start_dir) -> std::string {
                std::filesystem::path start_path;
                if (!start_dir.empty()) {
                    start_path = start_dir;
                }
                auto result = lfs::vis::gui::OpenImageFileDialog(start_path);
                return result.empty() ? "" : result.string();
            },
            nb::arg("start_dir") = "",
            "Open a file dialog to select an image file. Returns empty string if cancelled.");

        m.def(
            "open_folder_dialog",
            [](const std::string& title, const std::string& start_dir) -> std::string {
                std::filesystem::path start_path;
                if (!start_dir.empty()) {
                    start_path = start_dir;
                }
                auto result = lfs::vis::gui::SelectFolderDialog(title, start_path);
                return result.empty() ? "" : result.string();
            },
            nb::arg("title") = "Select Folder", nb::arg("start_dir") = "",
            "Open a folder selection dialog. Returns empty string if cancelled.");

        // Register callbacks for the visualizer to call into the Python panel system
        set_panel_draw_callback([](PanelSpace space) {
            PyPanelRegistry::instance().draw_panels(space);
        });

        set_panel_draw_single_callback([](const std::string& label) {
            PyPanelRegistry::instance().draw_single_panel(label);
        });

        set_panel_has_callback([](PanelSpace space) {
            return PyPanelRegistry::instance().has_panels(space);
        });

        set_panel_names_callback([](PanelSpace space) {
            return PyPanelRegistry::instance().get_panel_names(space);
        });

        // Register cleanup callback for proper shutdown
        set_python_cleanup_callback([]() {
            PyPanelRegistry::instance().unregister_all();
            PyUIHookRegistry::instance().clear_all();
        });

        set_python_hook_invoker([](const std::string& panel, const std::string& section, bool prepend) {
            auto& registry = PyUIHookRegistry::instance();
            if (registry.has_hooks(panel, section)) {
                registry.invoke(panel, section, prepend ? PyHookPosition::Prepend : PyHookPosition::Append);
            }
        });
    }

} // namespace lfs::python
