/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>
#include <imgui.h>

namespace lfs::core {
    class Tensor;
}

namespace lfs::vis {
    class VulkanContext;
}

namespace lfs::vis::gui {

    LFS_VIS_API void setImGuiVulkanTextureContext(VulkanContext* context);
    [[nodiscard]] LFS_VIS_API VulkanContext* getImGuiVulkanTextureContext();

    class LFS_VIS_API ImGuiVulkanTexture {
    public:
        ImGuiVulkanTexture() = default;
        ~ImGuiVulkanTexture();

        ImGuiVulkanTexture(const ImGuiVulkanTexture&) = delete;
        ImGuiVulkanTexture& operator=(const ImGuiVulkanTexture&) = delete;

        ImGuiVulkanTexture(ImGuiVulkanTexture&& other) noexcept;
        ImGuiVulkanTexture& operator=(ImGuiVulkanTexture&& other) noexcept;

        [[nodiscard]] bool upload(const std::uint8_t* pixels, int width, int height, int channels);
        [[nodiscard]] bool upload(const lfs::core::Tensor& image, int expected_width, int expected_height);
        [[nodiscard]] ImTextureID textureId() const;
        [[nodiscard]] bool valid() const;
        void reset();

    private:
        struct Impl;
        Impl* impl_ = nullptr;
    };

} // namespace lfs::vis::gui
