/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_context.hpp"

#include "core/logger.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <limits>
#include <set>
#include <utility>

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include <SDL3/SDL_vulkan.h>
#endif

namespace lfs::vis {

    VulkanContext::~VulkanContext() {
        shutdown();
    }

    bool VulkanContext::fail(std::string message) {
        last_error_ = std::move(message);
        LOG_ERROR("Vulkan: {}", last_error_);
        return false;
    }

    bool VulkanContext::init(SDL_Window* window, const int framebuffer_width, const int framebuffer_height) {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        framebuffer_width_ = framebuffer_width;
        framebuffer_height_ = framebuffer_height;

        return createInstance() &&
               createSurface(window) &&
               pickPhysicalDevice() &&
               createDevice() &&
               createSwapchain(framebuffer_width, framebuffer_height) &&
               createImageViews() &&
               createCommandPool() &&
               createCommandBuffers() &&
               createSyncObjects();
#else
        (void)window;
        (void)framebuffer_width;
        (void)framebuffer_height;
        return fail("Vulkan viewer dependencies are disabled at compile time");
#endif
    }

    void VulkanContext::shutdown() {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
        }

        if (render_finished_ != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, render_finished_, nullptr);
            render_finished_ = VK_NULL_HANDLE;
        }
        if (image_available_ != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, image_available_, nullptr);
            image_available_ = VK_NULL_HANDLE;
        }
        if (in_flight_ != VK_NULL_HANDLE) {
            vkDestroyFence(device_, in_flight_, nullptr);
            in_flight_ = VK_NULL_HANDLE;
        }

        destroySwapchain();

        if (command_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, command_pool_, nullptr);
            command_pool_ = VK_NULL_HANDLE;
        }
        if (device_ != VK_NULL_HANDLE) {
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
        if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
            SDL_Vulkan_DestroySurface(instance_, surface_, nullptr);
            surface_ = VK_NULL_HANDLE;
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
#endif
    }

    void VulkanContext::notifyFramebufferResized(const int width, const int height) {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        framebuffer_width_ = width;
        framebuffer_height_ = height;
        framebuffer_resized_ = true;
#else
        (void)width;
        (void)height;
#endif
    }

    bool VulkanContext::presentBootstrapFrame(const float r, const float g, const float b, const float a) {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        if (device_ == VK_NULL_HANDLE || swapchain_ == VK_NULL_HANDLE || framebuffer_width_ <= 0 || framebuffer_height_ <= 0) {
            return true;
        }

        if (framebuffer_resized_) {
            framebuffer_resized_ = false;
            if (!recreateSwapchain()) {
                return false;
            }
        }

        vkWaitForFences(device_, 1, &in_flight_, VK_TRUE, std::numeric_limits<uint64_t>::max());

        uint32_t image_index = 0;
        VkResult result = vkAcquireNextImageKHR(device_, swapchain_, std::numeric_limits<uint64_t>::max(),
                                                image_available_, VK_NULL_HANDLE, &image_index);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            return recreateSwapchain();
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            return fail(std::format("vkAcquireNextImageKHR failed: {}", static_cast<int>(result)));
        }

        vkResetFences(device_, 1, &in_flight_);
        vkResetCommandBuffer(command_buffers_[image_index], 0);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vkBeginCommandBuffer(command_buffers_[image_index], &begin_info);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkBeginCommandBuffer failed: {}", static_cast<int>(result)));
        }

        VkImageSubresourceRange color_range{};
        color_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        color_range.baseMipLevel = 0;
        color_range.levelCount = 1;
        color_range.baseArrayLayer = 0;
        color_range.layerCount = 1;

        const bool use_transfer_clear = (swapchain_image_usage_ & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0;
        if (use_transfer_clear) {
            VkImageMemoryBarrier to_transfer{};
            to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_transfer.srcAccessMask = 0;
            to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_transfer.oldLayout = swapchain_image_layouts_[image_index];
            to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_transfer.image = swapchain_images_[image_index];
            to_transfer.subresourceRange = color_range;
            vkCmdPipelineBarrier(command_buffers_[image_index],
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_transfer);

            const VkClearColorValue clear_color{{r, g, b, a}};
            vkCmdClearColorImage(command_buffers_[image_index],
                                 swapchain_images_[image_index],
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &clear_color,
                                 1,
                                 &color_range);

            VkImageMemoryBarrier to_present{};
            to_present.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_present.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_present.dstAccessMask = 0;
            to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_present.image = swapchain_images_[image_index];
            to_present.subresourceRange = color_range;
            vkCmdPipelineBarrier(command_buffers_[image_index],
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_present);
        } else {
            VkImageMemoryBarrier to_present{};
            to_present.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_present.srcAccessMask = 0;
            to_present.dstAccessMask = 0;
            to_present.oldLayout = swapchain_image_layouts_[image_index];
            to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_present.image = swapchain_images_[image_index];
            to_present.subresourceRange = color_range;
            vkCmdPipelineBarrier(command_buffers_[image_index],
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_present);
        }

        result = vkEndCommandBuffer(command_buffers_[image_index]);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkEndCommandBuffer failed: {}", static_cast<int>(result)));
        }

        const VkPipelineStageFlags wait_stage = use_transfer_clear ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &image_available_;
        submit_info.pWaitDstStageMask = &wait_stage;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffers_[image_index];
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &render_finished_;
        result = vkQueueSubmit(graphics_queue_, 1, &submit_info, in_flight_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkQueueSubmit failed: {}", static_cast<int>(result)));
        }

        VkPresentInfoKHR present_info{};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &render_finished_;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain_;
        present_info.pImageIndices = &image_index;
        result = vkQueuePresentKHR(present_queue_, &present_info);
        swapchain_image_layouts_[image_index] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            return recreateSwapchain();
        }
        if (result != VK_SUCCESS) {
            return fail(std::format("vkQueuePresentKHR failed: {}", static_cast<int>(result)));
        }

        return true;
#else
        (void)r;
        (void)g;
        (void)b;
        (void)a;
        return false;
#endif
    }

#ifdef LFS_VULKAN_VIEWER_ENABLED
    bool VulkanContext::createInstance() {
        uint32_t extension_count = 0;
        const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&extension_count);
        if (!extensions || extension_count == 0) {
            return fail(std::format("SDL_Vulkan_GetInstanceExtensions failed: {}", SDL_GetError()));
        }

        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "LichtFeld Studio";
        app_info.applicationVersion = VK_MAKE_VERSION(0, 0, 0);
        app_info.pEngineName = "LichtFeld Studio";
        app_info.engineVersion = VK_MAKE_VERSION(0, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        create_info.enabledExtensionCount = extension_count;
        create_info.ppEnabledExtensionNames = extensions;

        const VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreateInstance failed: {}", static_cast<int>(result)));
        }
        return true;
    }

    bool VulkanContext::createSurface(SDL_Window* window) {
        if (!SDL_Vulkan_CreateSurface(window, instance_, nullptr, &surface_)) {
            return fail(std::format("SDL_Vulkan_CreateSurface failed: {}", SDL_GetError()));
        }
        return true;
    }

    VulkanContext::QueueFamilies VulkanContext::findQueueFamilies(VkPhysicalDevice device) const {
        QueueFamilies indices;

        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

        for (uint32_t i = 0; i < count; ++i) {
            if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
                indices.graphics = i;
            }

            VkBool32 present_supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &present_supported);
            if (present_supported == VK_TRUE) {
                indices.present = i;
            }

            if (indices.complete()) {
                break;
            }
        }
        return indices;
    }

    bool VulkanContext::deviceSupportsSwapchain(VkPhysicalDevice device) const {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> extensions(count);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());

        std::set<std::string> required{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        for (const auto& extension : extensions) {
            required.erase(extension.extensionName);
        }
        return required.empty();
    }

    VulkanContext::SwapchainSupport VulkanContext::querySwapchainSupport(VkPhysicalDevice device) const {
        SwapchainSupport details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);

        uint32_t count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &count, nullptr);
        details.formats.resize(count);
        if (count > 0) {
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &count, details.formats.data());
        }

        count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &count, nullptr);
        details.present_modes.resize(count);
        if (count > 0) {
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &count, details.present_modes.data());
        }

        return details;
    }

    bool VulkanContext::pickPhysicalDevice() {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        if (count == 0) {
            return fail("No Vulkan physical devices found");
        }

        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());

        VkPhysicalDevice fallback = VK_NULL_HANDLE;
        for (const auto device : devices) {
            const QueueFamilies families = findQueueFamilies(device);
            if (!families.complete() || !deviceSupportsSwapchain(device)) {
                continue;
            }

            const SwapchainSupport swapchain = querySwapchainSupport(device);
            if (swapchain.formats.empty() || swapchain.present_modes.empty()) {
                continue;
            }

            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(device, &props);
            if (fallback == VK_NULL_HANDLE) {
                fallback = device;
            }
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physical_device_ = device;
                break;
            }
        }

        if (physical_device_ == VK_NULL_HANDLE) {
            physical_device_ = fallback;
        }
        if (physical_device_ == VK_NULL_HANDLE) {
            return fail("No Vulkan device supports graphics presentation and swapchain creation");
        }

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_device_, &props);
        LOG_INFO("Vulkan device: {}", props.deviceName);
        return true;
    }

    bool VulkanContext::createDevice() {
        const QueueFamilies families = findQueueFamilies(physical_device_);
        if (!families.complete()) {
            return fail("Selected Vulkan device is missing graphics or present queues");
        }

        graphics_queue_family_ = *families.graphics;
        present_queue_family_ = *families.present;

        const std::set<uint32_t> unique_families{graphics_queue_family_, present_queue_family_};
        std::vector<VkDeviceQueueCreateInfo> queue_infos;
        constexpr float queue_priority = 1.0f;
        for (const uint32_t family : unique_families) {
            VkDeviceQueueCreateInfo queue_info{};
            queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_info.queueFamilyIndex = family;
            queue_info.queueCount = 1;
            queue_info.pQueuePriorities = &queue_priority;
            queue_infos.push_back(queue_info);
        }

        const std::array<const char*, 1> extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        const VkPhysicalDeviceFeatures features{};
        VkDeviceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_infos.size());
        create_info.pQueueCreateInfos = queue_infos.data();
        create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();
        create_info.pEnabledFeatures = &features;

        const VkResult result = vkCreateDevice(physical_device_, &create_info, nullptr, &device_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreateDevice failed: {}", static_cast<int>(result)));
        }

        vkGetDeviceQueue(device_, graphics_queue_family_, 0, &graphics_queue_);
        vkGetDeviceQueue(device_, present_queue_family_, 0, &present_queue_);
        return true;
    }

    VkSurfaceFormatKHR VulkanContext::chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const {
        for (const auto& format : formats) {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return format;
            }
        }
        return formats.front();
    }

    VkPresentModeKHR VulkanContext::choosePresentMode(const std::vector<VkPresentModeKHR>& modes) const {
        for (const auto mode : modes) {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return mode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D VulkanContext::chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                                    const int framebuffer_width,
                                                    const int framebuffer_height) const {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }

        VkExtent2D extent{};
        extent.width = static_cast<uint32_t>(std::max(1, framebuffer_width));
        extent.height = static_cast<uint32_t>(std::max(1, framebuffer_height));
        extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        return extent;
    }

    bool VulkanContext::createSwapchain(const int framebuffer_width, const int framebuffer_height) {
        const SwapchainSupport support = querySwapchainSupport(physical_device_);
        if (support.formats.empty() || support.present_modes.empty()) {
            return fail("Vulkan swapchain support is incomplete");
        }

        const VkSurfaceFormatKHR surface_format = chooseSurfaceFormat(support.formats);
        const VkPresentModeKHR present_mode = choosePresentMode(support.present_modes);
        const VkExtent2D extent = chooseSwapchainExtent(support.capabilities, framebuffer_width, framebuffer_height);

        VkImageUsageFlags image_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if ((support.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0) {
            return fail("Vulkan swapchain does not support color attachment usage");
        }
        if ((support.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0) {
            image_usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        } else {
            LOG_WARN("Vulkan swapchain does not support transfer clears; bootstrap frame will skip the clear pass");
        }

        uint32_t image_count = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0 && image_count > support.capabilities.maxImageCount) {
            image_count = support.capabilities.maxImageCount;
        }

        const std::array<uint32_t, 2> queue_indices{graphics_queue_family_, present_queue_family_};
        const bool shared_queues = graphics_queue_family_ != present_queue_family_;
        VkSwapchainCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface = surface_;
        create_info.minImageCount = image_count;
        create_info.imageFormat = surface_format.format;
        create_info.imageColorSpace = surface_format.colorSpace;
        create_info.imageExtent = extent;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage = image_usage;
        create_info.imageSharingMode = shared_queues ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
        create_info.queueFamilyIndexCount = shared_queues ? static_cast<uint32_t>(queue_indices.size()) : 0u;
        create_info.pQueueFamilyIndices = shared_queues ? queue_indices.data() : nullptr;
        create_info.preTransform = support.capabilities.currentTransform;
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        create_info.presentMode = present_mode;
        create_info.clipped = VK_TRUE;
        create_info.oldSwapchain = VK_NULL_HANDLE;

        const VkResult result = vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapchain_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreateSwapchainKHR failed: {}", static_cast<int>(result)));
        }

        vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
        swapchain_images_.resize(image_count);
        vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, swapchain_images_.data());
        swapchain_image_layouts_.assign(image_count, VK_IMAGE_LAYOUT_UNDEFINED);
        swapchain_format_ = surface_format.format;
        swapchain_extent_ = extent;
        swapchain_image_usage_ = image_usage;
        return true;
    }

    bool VulkanContext::createImageViews() {
        swapchain_image_views_.resize(swapchain_images_.size());
        for (size_t i = 0; i < swapchain_images_.size(); ++i) {
            VkImageViewCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            create_info.image = swapchain_images_[i];
            create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            create_info.format = swapchain_format_;
            create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            create_info.subresourceRange.baseMipLevel = 0;
            create_info.subresourceRange.levelCount = 1;
            create_info.subresourceRange.baseArrayLayer = 0;
            create_info.subresourceRange.layerCount = 1;

            const VkResult result = vkCreateImageView(device_, &create_info, nullptr, &swapchain_image_views_[i]);
            if (result != VK_SUCCESS) {
                return fail(std::format("vkCreateImageView failed: {}", static_cast<int>(result)));
            }
        }
        return true;
    }

    bool VulkanContext::createCommandPool() {
        VkCommandPoolCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        create_info.queueFamilyIndex = graphics_queue_family_;
        const VkResult result = vkCreateCommandPool(device_, &create_info, nullptr, &command_pool_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreateCommandPool failed: {}", static_cast<int>(result)));
        }
        return true;
    }

    bool VulkanContext::createCommandBuffers() {
        command_buffers_.resize(swapchain_images_.size());
        VkCommandBufferAllocateInfo allocate_info{};
        allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate_info.commandPool = command_pool_;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = static_cast<uint32_t>(command_buffers_.size());
        const VkResult result = vkAllocateCommandBuffers(device_, &allocate_info, command_buffers_.data());
        if (result != VK_SUCCESS) {
            return fail(std::format("vkAllocateCommandBuffers failed: {}", static_cast<int>(result)));
        }
        return true;
    }

    bool VulkanContext::createSyncObjects() {
        VkSemaphoreCreateInfo semaphore_info{};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        VkResult result = vkCreateSemaphore(device_, &semaphore_info, nullptr, &image_available_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreateSemaphore(image_available) failed: {}", static_cast<int>(result)));
        }
        result = vkCreateSemaphore(device_, &semaphore_info, nullptr, &render_finished_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreateSemaphore(render_finished) failed: {}", static_cast<int>(result)));
        }
        result = vkCreateFence(device_, &fence_info, nullptr, &in_flight_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreateFence failed: {}", static_cast<int>(result)));
        }
        return true;
    }

    void VulkanContext::destroySwapchain() {
        if (device_ == VK_NULL_HANDLE) {
            return;
        }

        if (!command_buffers_.empty() && command_pool_ != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device_, command_pool_, static_cast<uint32_t>(command_buffers_.size()), command_buffers_.data());
            command_buffers_.clear();
        }
        for (const VkImageView view : swapchain_image_views_) {
            vkDestroyImageView(device_, view, nullptr);
        }
        swapchain_image_views_.clear();
        swapchain_images_.clear();
        swapchain_image_layouts_.clear();

        if (swapchain_ != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
            swapchain_ = VK_NULL_HANDLE;
        }
        swapchain_image_usage_ = 0;
    }

    bool VulkanContext::recreateSwapchain() {
        if (framebuffer_width_ <= 0 || framebuffer_height_ <= 0) {
            return true;
        }

        vkDeviceWaitIdle(device_);
        destroySwapchain();
        return createSwapchain(framebuffer_width_, framebuffer_height_) &&
               createImageViews() &&
               createCommandBuffers();
    }
#endif

} // namespace lfs::vis
