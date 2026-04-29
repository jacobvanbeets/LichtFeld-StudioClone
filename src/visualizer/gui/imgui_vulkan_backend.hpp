/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.h"

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include <vulkan/vulkan.h>
#endif

struct SDL_Window;
struct ImDrawData;

namespace lfs::vis {
    class VulkanContext;
}

namespace lfs::vis::gui {

    class ImGuiVulkanBackend {
    public:
        ImGuiVulkanBackend() = default;
        ~ImGuiVulkanBackend();

        ImGuiVulkanBackend(const ImGuiVulkanBackend&) = delete;
        ImGuiVulkanBackend& operator=(const ImGuiVulkanBackend&) = delete;

        [[nodiscard]] bool init(SDL_Window* window, VulkanContext& context);
        void shutdown();
        void newFrame();

#ifdef LFS_VULKAN_VIEWER_ENABLED
        void renderDrawData(ImDrawData* draw_data, VkCommandBuffer command_buffer);
#endif

        [[nodiscard]] bool initialized() const { return initialized_; }

    private:
        bool initialized_ = false;
    };

} // namespace lfs::vis::gui
