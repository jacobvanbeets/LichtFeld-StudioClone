/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include <vulkan/vulkan.h>
#endif

struct SDL_Window;

namespace lfs::vis {

    class VulkanContext {
    public:
        VulkanContext() = default;
        ~VulkanContext();

        VulkanContext(const VulkanContext&) = delete;
        VulkanContext& operator=(const VulkanContext&) = delete;

        bool init(SDL_Window* window, int framebuffer_width, int framebuffer_height);
        void shutdown();
        void notifyFramebufferResized(int width, int height);

        [[nodiscard]] bool presentBootstrapFrame(float r, float g, float b, float a);
        [[nodiscard]] const std::string& lastError() const { return last_error_; }

#ifdef LFS_VULKAN_VIEWER_ENABLED
        [[nodiscard]] VkInstance instance() const { return instance_; }
        [[nodiscard]] VkPhysicalDevice physicalDevice() const { return physical_device_; }
        [[nodiscard]] VkDevice device() const { return device_; }
        [[nodiscard]] VkSurfaceKHR surface() const { return surface_; }
        [[nodiscard]] VkQueue graphicsQueue() const { return graphics_queue_; }
        [[nodiscard]] VkQueue presentQueue() const { return present_queue_; }
        [[nodiscard]] uint32_t graphicsQueueFamily() const { return graphics_queue_family_; }
        [[nodiscard]] uint32_t presentQueueFamily() const { return present_queue_family_; }
        [[nodiscard]] VkFormat swapchainFormat() const { return swapchain_format_; }
        [[nodiscard]] VkExtent2D swapchainExtent() const { return swapchain_extent_; }
#endif

    private:
        bool fail(std::string message);

#ifdef LFS_VULKAN_VIEWER_ENABLED
        struct QueueFamilies {
            std::optional<uint32_t> graphics;
            std::optional<uint32_t> present;
            [[nodiscard]] bool complete() const { return graphics.has_value() && present.has_value(); }
        };

        struct SwapchainSupport {
            VkSurfaceCapabilitiesKHR capabilities{};
            std::vector<VkSurfaceFormatKHR> formats;
            std::vector<VkPresentModeKHR> present_modes;
        };

        bool createInstance();
        bool createSurface(SDL_Window* window);
        bool pickPhysicalDevice();
        bool createDevice();
        bool createSwapchain(int framebuffer_width, int framebuffer_height);
        bool createImageViews();
        bool createCommandPool();
        bool createCommandBuffers();
        bool createSyncObjects();
        bool recreateSwapchain();

        void destroySwapchain();

        [[nodiscard]] QueueFamilies findQueueFamilies(VkPhysicalDevice device) const;
        [[nodiscard]] bool deviceSupportsSwapchain(VkPhysicalDevice device) const;
        [[nodiscard]] SwapchainSupport querySwapchainSupport(VkPhysicalDevice device) const;
        [[nodiscard]] VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const;
        [[nodiscard]] VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes) const;
        [[nodiscard]] VkExtent2D chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                                       int framebuffer_width,
                                                       int framebuffer_height) const;

        VkInstance instance_ = VK_NULL_HANDLE;
        VkSurfaceKHR surface_ = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
        VkDevice device_ = VK_NULL_HANDLE;
        VkQueue graphics_queue_ = VK_NULL_HANDLE;
        VkQueue present_queue_ = VK_NULL_HANDLE;
        uint32_t graphics_queue_family_ = 0;
        uint32_t present_queue_family_ = 0;

        VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
        VkFormat swapchain_format_ = VK_FORMAT_UNDEFINED;
        VkExtent2D swapchain_extent_{};
        VkImageUsageFlags swapchain_image_usage_ = 0;
        std::vector<VkImage> swapchain_images_;
        std::vector<VkImageView> swapchain_image_views_;
        std::vector<VkImageLayout> swapchain_image_layouts_;

        VkCommandPool command_pool_ = VK_NULL_HANDLE;
        std::vector<VkCommandBuffer> command_buffers_;
        VkSemaphore image_available_ = VK_NULL_HANDLE;
        VkSemaphore render_finished_ = VK_NULL_HANDLE;
        VkFence in_flight_ = VK_NULL_HANDLE;

        bool framebuffer_resized_ = false;
        int framebuffer_width_ = 0;
        int framebuffer_height_ = 0;
#endif

        std::string last_error_;
    };

} // namespace lfs::vis
