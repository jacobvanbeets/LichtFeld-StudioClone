/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "property_system.hpp"

#include <mutex>
#include <unordered_map>

namespace lfs::core::prop {

    class PropertyRegistry {
    public:
        static PropertyRegistry& instance();

        void register_group(PropertyGroup group);
        [[nodiscard]] const PropertyGroup* get_group(const std::string& group_id) const;
        [[nodiscard]] const PropertyMeta* get_property(const std::string& group_id,
                                                       const std::string& prop_id) const;
        [[nodiscard]] std::vector<std::string> get_group_ids() const;

        size_t subscribe(PropertyCallback callback);
        size_t subscribe(const std::string& group_id, const std::string& prop_id, PropertyCallback callback);
        void unsubscribe(size_t id);
        void notify(const std::string& group_id, const std::string& prop_id,
                    const std::any& old_value, const std::any& new_value);

    private:
        PropertyRegistry() = default;

        mutable std::mutex mutex_;
        std::unordered_map<std::string, PropertyGroup> groups_;
        std::unordered_map<size_t, PropertyCallback> global_subscribers_;
        std::unordered_map<std::string, std::unordered_map<size_t, PropertyCallback>> prop_subscribers_;
        size_t next_id_ = 1;
    };

    template <typename StructT>
    class PropertyGroupBuilder {
    public:
        PropertyGroupBuilder(const std::string& group_id, const std::string& group_name)
            : group_{.id = group_id, .name = group_name} {}

        PropertyGroupBuilder& float_prop(float StructT::*member,
                                         const std::string& id,
                                         const std::string& name,
                                         float default_val,
                                         float min_val,
                                         float max_val,
                                         const std::string& desc = "",
                                         PropUIHint hint = PropUIHint::Slider) {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::Float;
            meta.ui_hint = hint;
            meta.default_value = default_val;
            meta.min_value = min_val;
            meta.max_value = max_val;
            meta.soft_min = min_val;
            meta.soft_max = max_val;
            meta.step = (max_val - min_val) / 100.0;

            meta.getter = [member](const void* obj) -> std::any {
                return static_cast<const StructT*>(obj)->*member;
            };
            meta.setter = [member](void* obj, const std::any& val) {
                static_cast<StructT*>(obj)->*member = std::any_cast<float>(val);
            };

            group_.properties.push_back(std::move(meta));
            return *this;
        }

        PropertyGroupBuilder& int_prop(int StructT::*member,
                                       const std::string& id,
                                       const std::string& name,
                                       int default_val,
                                       int min_val,
                                       int max_val,
                                       const std::string& desc = "",
                                       PropUIHint hint = PropUIHint::Slider) {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::Int;
            meta.ui_hint = hint;
            meta.default_value = default_val;
            meta.min_value = min_val;
            meta.max_value = max_val;
            meta.soft_min = min_val;
            meta.soft_max = max_val;
            meta.step = 1.0;

            meta.getter = [member](const void* obj) -> std::any {
                return static_cast<const StructT*>(obj)->*member;
            };
            meta.setter = [member](void* obj, const std::any& val) {
                static_cast<StructT*>(obj)->*member = std::any_cast<int>(val);
            };

            group_.properties.push_back(std::move(meta));
            return *this;
        }

        PropertyGroupBuilder& size_prop(size_t StructT::*member,
                                        const std::string& id,
                                        const std::string& name,
                                        size_t default_val,
                                        size_t min_val,
                                        size_t max_val,
                                        const std::string& desc = "",
                                        PropUIHint hint = PropUIHint::Input) {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::SizeT;
            meta.ui_hint = hint;
            meta.default_value = static_cast<double>(default_val);
            meta.min_value = static_cast<double>(min_val);
            meta.max_value = static_cast<double>(max_val);
            meta.soft_min = static_cast<double>(min_val);
            meta.soft_max = static_cast<double>(max_val);
            meta.step = 1.0;

            meta.getter = [member](const void* obj) -> std::any {
                return static_cast<const StructT*>(obj)->*member;
            };
            meta.setter = [member](void* obj, const std::any& val) {
                static_cast<StructT*>(obj)->*member = std::any_cast<size_t>(val);
            };

            group_.properties.push_back(std::move(meta));
            return *this;
        }

        PropertyGroupBuilder& bool_prop(bool StructT::*member,
                                        const std::string& id,
                                        const std::string& name,
                                        bool default_val,
                                        const std::string& desc = "") {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::Bool;
            meta.ui_hint = PropUIHint::Checkbox;
            meta.default_value = default_val ? 1.0 : 0.0;

            meta.getter = [member](const void* obj) -> std::any {
                return static_cast<const StructT*>(obj)->*member;
            };
            meta.setter = [member](void* obj, const std::any& val) {
                static_cast<StructT*>(obj)->*member = std::any_cast<bool>(val);
            };

            group_.properties.push_back(std::move(meta));
            return *this;
        }

        PropertyGroupBuilder& string_prop(std::string StructT::*member,
                                          const std::string& id,
                                          const std::string& name,
                                          const std::string& default_val = "",
                                          const std::string& desc = "") {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::String;
            meta.ui_hint = PropUIHint::Input;
            meta.default_string = default_val;

            meta.getter = [member](const void* obj) -> std::any {
                return static_cast<const StructT*>(obj)->*member;
            };
            meta.setter = [member](void* obj, const std::any& val) {
                static_cast<StructT*>(obj)->*member = std::any_cast<std::string>(val);
            };

            group_.properties.push_back(std::move(meta));
            return *this;
        }

        template <typename EnumT>
        PropertyGroupBuilder& enum_prop(EnumT StructT::*member,
                                        const std::string& id,
                                        const std::string& name,
                                        EnumT default_val,
                                        std::initializer_list<std::pair<std::string, EnumT>> items,
                                        const std::string& desc = "") {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::Enum;
            meta.ui_hint = PropUIHint::Combo;
            meta.default_enum = static_cast<int>(default_val);

            for (const auto& [item_name, item_val] : items) {
                EnumItem ei;
                ei.name = item_name;
                ei.identifier = item_name;
                ei.value = static_cast<int>(item_val);
                meta.enum_items.push_back(std::move(ei));
            }

            meta.getter = [member](const void* obj) -> std::any {
                return static_cast<int>(static_cast<const StructT*>(obj)->*member);
            };
            meta.setter = [member](void* obj, const std::any& val) {
                static_cast<StructT*>(obj)->*member = static_cast<EnumT>(std::any_cast<int>(val));
            };

            group_.properties.push_back(std::move(meta));
            return *this;
        }

        PropertyGroupBuilder& flags(uint32_t f) {
            if (!group_.properties.empty()) {
                group_.properties.back().flags = f;
            }
            return *this;
        }

        PropertyGroupBuilder& category(const std::string& cat) {
            if (!group_.properties.empty()) {
                group_.properties.back().group = cat;
            }
            return *this;
        }

        void build() { PropertyRegistry::instance().register_group(std::move(group_)); }
        [[nodiscard]] PropertyGroup get() const { return group_; }

    private:
        PropertyGroup group_;
    };

} // namespace lfs::core::prop
