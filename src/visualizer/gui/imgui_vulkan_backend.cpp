/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/imgui_vulkan_backend.hpp"

#include "core/logger.hpp"
#include "gui/imgui_vulkan_texture.hpp"
#include "window/vulkan_context.hpp"

#include <imgui_impl_sdl3.h>

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include <imgui_impl_vulkan.h>
#endif

#include <algorithm>

namespace lfs::vis::gui {

    namespace {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        void checkVulkanResult(const VkResult result) {
            if (result < 0) {
                LOG_ERROR("ImGui Vulkan backend error: {}", static_cast<int>(result));
            }
        }
#endif
    } // namespace

    ImGuiVulkanBackend::~ImGuiVulkanBackend() {
        shutdown();
    }

    bool ImGuiVulkanBackend::init(SDL_Window* window, VulkanContext& context) {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        if (initialized_) {
            return true;
        }
        if (!window) {
            LOG_ERROR("ImGui Vulkan backend requires a valid SDL window");
            return false;
        }
        if (context.device() == VK_NULL_HANDLE || context.renderPass() == VK_NULL_HANDLE) {
            LOG_ERROR("ImGui Vulkan backend requires an initialized Vulkan context");
            return false;
        }

        if (!ImGui_ImplSDL3_InitForVulkan(window)) {
            LOG_ERROR("ImGui SDL3 Vulkan platform backend initialization failed");
            return false;
        }

        ImGui_ImplVulkan_InitInfo init_info{};
        init_info.ApiVersion = VK_API_VERSION_1_0;
        init_info.Instance = context.instance();
        init_info.PhysicalDevice = context.physicalDevice();
        init_info.Device = context.device();
        init_info.QueueFamily = context.graphicsQueueFamily();
        init_info.Queue = context.graphicsQueue();
        init_info.DescriptorPoolSize = 1024;
        init_info.MinImageCount = std::max(2u, context.minImageCount());
        init_info.ImageCount = std::max(init_info.MinImageCount, context.imageCount());
        init_info.PipelineInfoMain.RenderPass = context.renderPass();
        init_info.PipelineInfoMain.Subpass = 0;
        init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        init_info.CheckVkResultFn = checkVulkanResult;

        if (!ImGui_ImplVulkan_Init(&init_info)) {
            ImGui_ImplSDL3_Shutdown();
            LOG_ERROR("ImGui Vulkan renderer backend initialization failed");
            return false;
        }

        initialized_ = true;
        setImGuiVulkanTextureContext(&context);
        LOG_INFO("ImGui Vulkan backend initialized");
        return true;
#else
        (void)window;
        (void)context;
        LOG_ERROR("ImGui Vulkan backend requested, but Vulkan viewer dependencies are disabled");
        return false;
#endif
    }

    void ImGuiVulkanBackend::shutdown() {
        if (!initialized_) {
            return;
        }
#ifdef LFS_VULKAN_VIEWER_ENABLED
        setImGuiVulkanTextureContext(nullptr);
        ImGui_ImplVulkan_Shutdown();
#endif
        ImGui_ImplSDL3_Shutdown();
        initialized_ = false;
    }

    void ImGuiVulkanBackend::newFrame() {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        if (initialized_) {
            ImGui_ImplSDL3_NewFrame();
            ImGui_ImplVulkan_NewFrame();
        }
#endif
    }

#ifdef LFS_VULKAN_VIEWER_ENABLED
    void ImGuiVulkanBackend::renderDrawData(ImDrawData* draw_data, const VkCommandBuffer command_buffer) {
        if (!initialized_ || command_buffer == VK_NULL_HANDLE) {
            return;
        }
        ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffer);
    }
#endif

} // namespace lfs::vis::gui
