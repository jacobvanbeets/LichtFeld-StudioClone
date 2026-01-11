/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <any>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lfs::core::prop {

    enum class PropType { Bool,
                          Int,
                          Float,
                          String,
                          Enum,
                          SizeT };

    enum class PropUIHint { Default,
                            Slider,
                            Drag,
                            Input,
                            Checkbox,
                            Combo,
                            Hidden };

    enum PropFlags : uint32_t {
        PROP_NONE = 0,
        PROP_READONLY = 1 << 0,
        PROP_LIVE_UPDATE = 1 << 1,
        PROP_NEEDS_RESTART = 1 << 2,
        PROP_ANIMATABLE = 1 << 3,
    };

    inline PropFlags operator|(PropFlags a, PropFlags b) {
        return static_cast<PropFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }

    inline PropFlags operator&(PropFlags a, PropFlags b) {
        return static_cast<PropFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
    }

    struct EnumItem {
        std::string name;
        std::string identifier;
        int value;
    };

    struct PropertyMeta {
        std::string id;
        std::string name;
        std::string description;
        std::string group;
        PropType type = PropType::Float;
        PropUIHint ui_hint = PropUIHint::Default;
        uint32_t flags = PROP_NONE;

        double min_value = 0.0;
        double max_value = 1.0;
        double soft_min = 0.0;
        double soft_max = 1.0;
        double step = 1.0;
        double default_value = 0.0;
        std::string default_string;
        std::vector<EnumItem> enum_items;
        int default_enum = 0;

        std::function<std::any(const void*)> getter;
        std::function<void(void*, const std::any&)> setter;

        [[nodiscard]] bool has_flag(PropFlags f) const { return (flags & f) != PROP_NONE; }
        [[nodiscard]] bool is_readonly() const { return has_flag(PROP_READONLY); }
        [[nodiscard]] bool is_live_update() const { return has_flag(PROP_LIVE_UPDATE); }
        [[nodiscard]] bool needs_restart() const { return has_flag(PROP_NEEDS_RESTART); }
    };

    struct PropertyGroup {
        std::string id;
        std::string name;
        std::vector<PropertyMeta> properties;

        [[nodiscard]] const PropertyMeta* find(const std::string& prop_id) const {
            for (const auto& p : properties) {
                if (p.id == prop_id)
                    return &p;
            }
            return nullptr;
        }
    };

    using PropertyCallback = std::function<void(const std::string& group_id,
                                                const std::string& prop_id,
                                                const std::any& old_value,
                                                const std::any& new_value)>;

} // namespace lfs::core::prop
