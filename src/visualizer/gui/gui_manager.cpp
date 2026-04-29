/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// glad must be included before OpenGL headers
// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/gui_manager.hpp"
#include "control/command_api.hpp"
#include "core/cuda_version.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "gui/bounds_gizmo.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/layout_state.hpp"
#include "gui/native_panels.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/panel_registry.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "gui/rmlui/rmlui_system_interface.hpp"
#include "gui/rotation_gizmo.hpp"
#include "gui/scale_gizmo.hpp"
#include "gui/scene_panel_native.hpp"
#include "gui/string_keys.hpp"
#include "gui/translation_gizmo.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/file_association.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "gui/vulkan_scene_cuda_upload.hpp"
#include <implot.h>

#include "gui/gui_focus_state.hpp"
#include "input/frame_input_buffer.hpp"
#include "input/input_controller.hpp"
#include "internal/resource_paths.hpp"
#include "tools/align_tool.hpp"

#include "core/events.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "python/package_manager.hpp"
#include "python/python_runtime.hpp"
#include "python/ui_hooks.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/image_layout.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "tools/brush_tool.hpp"
#include "tools/selection_tool.hpp"
#include "visualizer_impl.hpp"
#include "window/vulkan_context.hpp"
#include <OpenImageIO/imageio.h>
#include <SDL3/SDL.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl3.h>
#ifdef LFS_VULKAN_VIEWER_ENABLED
#include <imgui_impl_vulkan.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#endif
#include <imgui_internal.h>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace lfs::vis::gui {

    namespace {
        const FrameInputBuffer* s_frame_input = nullptr;

        [[nodiscard]] std::optional<std::vector<unsigned char>> tensorToRgba8(
            const lfs::core::Tensor& image,
            const glm::ivec2 expected_size) {
            if (!image.is_valid() || image.ndim() != 3 || expected_size.x <= 0 || expected_size.y <= 0) {
                return std::nullopt;
            }

            const auto layout = lfs::rendering::detectImageLayout(image);
            if (layout == lfs::rendering::ImageLayout::Unknown) {
                LOG_ERROR("Vulkan scene upload received unsupported tensor shape [{}, {}, {}]",
                          image.size(0), image.size(1), image.size(2));
                return std::nullopt;
            }

            lfs::core::Tensor formatted = (layout == lfs::rendering::ImageLayout::HWC)
                                              ? image
                                              : image.permute({1, 2, 0}).contiguous();
            if (formatted.device() == lfs::core::Device::CUDA) {
                formatted = formatted.cpu();
            }
            if (formatted.dtype() != lfs::core::DataType::UInt8) {
                formatted = (formatted.clamp(0.0f, 1.0f) * 255.0f).to(lfs::core::DataType::UInt8);
            }
            formatted = formatted.contiguous();

            const int height = static_cast<int>(formatted.size(0));
            const int width = static_cast<int>(formatted.size(1));
            const int channels = static_cast<int>(formatted.size(2));
            if (width != expected_size.x || height != expected_size.y || !formatted.ptr<unsigned char>()) {
                LOG_ERROR("Vulkan scene upload dimension mismatch: {}x{} vs {}x{}",
                          width, height, expected_size.x, expected_size.y);
                return std::nullopt;
            }
            if (channels != 1 && channels != 3 && channels != 4) {
                LOG_ERROR("Vulkan scene upload received unsupported channel count {}", channels);
                return std::nullopt;
            }

            const unsigned char* const src = formatted.ptr<unsigned char>();
            std::vector<unsigned char> rgba(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const size_t src_offset = (static_cast<size_t>(y) * width + x) * channels;
                    const size_t dst_offset = (static_cast<size_t>(y) * width + x) * 4u;
                    if (channels == 1) {
                        rgba[dst_offset + 0] = src[src_offset];
                        rgba[dst_offset + 1] = src[src_offset];
                        rgba[dst_offset + 2] = src[src_offset];
                        rgba[dst_offset + 3] = 255;
                    } else {
                        rgba[dst_offset + 0] = src[src_offset + 0];
                        rgba[dst_offset + 1] = src[src_offset + 1];
                        rgba[dst_offset + 2] = src[src_offset + 2];
                        // Match the OpenGL viewport presentation: scene alpha is not a UI fade mask.
                        rgba[dst_offset + 3] = 255;
                    }
                }
            }
            return rgba;
        }

    } // namespace

    class VulkanSceneTexture {
        public:
            VulkanSceneTexture() = default;
            ~VulkanSceneTexture() { shutdown(); }

            VulkanSceneTexture(const VulkanSceneTexture&) = delete;
            VulkanSceneTexture& operator=(const VulkanSceneTexture&) = delete;

            bool init(lfs::vis::VulkanContext& context) {
#ifdef LFS_VULKAN_VIEWER_ENABLED
                if (initialized_) {
                    return true;
                }
                device_ = context.device();
                physical_device_ = context.physicalDevice();
                graphics_queue_ = context.graphicsQueue();
                graphics_queue_family_ = context.graphicsQueueFamily();
                external_memory_interop_available_ = context.externalMemoryInteropEnabled();
                use_dedicated_external_memory_ = context.externalMemoryDedicatedAllocationEnabled();
                if (device_ == VK_NULL_HANDLE || physical_device_ == VK_NULL_HANDLE ||
                    graphics_queue_ == VK_NULL_HANDLE) {
                    LOG_ERROR("Vulkan scene texture requires an initialized Vulkan context");
                    return false;
                }

                VkCommandPoolCreateInfo pool_info{};
                pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                pool_info.queueFamilyIndex = graphics_queue_family_;
                if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
                    LOG_ERROR("Failed to create Vulkan scene texture command pool");
                    return false;
                }

                VkSamplerCreateInfo sampler_info{};
                sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
                sampler_info.magFilter = VK_FILTER_LINEAR;
                sampler_info.minFilter = VK_FILTER_LINEAR;
                sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                sampler_info.maxLod = 1.0f;
                if (vkCreateSampler(device_, &sampler_info, nullptr, &sampler_) != VK_SUCCESS) {
                    LOG_ERROR("Failed to create Vulkan scene texture sampler");
                    shutdown();
                    return false;
                }

                initialized_ = true;
                return true;
#else
                (void)context;
                return false;
#endif
            }

            void shutdown() {
#ifdef LFS_VULKAN_VIEWER_ENABLED
                if (device_ != VK_NULL_HANDLE) {
                    vkDeviceWaitIdle(device_);
                }
                destroyImageResources();
                if (sampler_ != VK_NULL_HANDLE) {
                    vkDestroySampler(device_, sampler_, nullptr);
                    sampler_ = VK_NULL_HANDLE;
                }
                if (command_pool_ != VK_NULL_HANDLE) {
                    vkDestroyCommandPool(device_, command_pool_, nullptr);
                    command_pool_ = VK_NULL_HANDLE;
                }
                destroyCudaUploadBuffer();
#endif
                initialized_ = false;
            }

            bool upload(const lfs::core::Tensor& image,
                        const glm::ivec2 size,
                        const bool enable_cuda_interop) {
#ifdef LFS_VULKAN_VIEWER_ENABLED
                if (!initialized_) {
                    return false;
                }
                if (enable_cuda_interop && uploadWithCudaInterop(image, size)) {
                    return true;
                }
                return uploadWithStaging(image, size);
#else
                (void)image;
                (void)size;
                (void)enable_cuda_interop;
                return false;
#endif
            }

            [[nodiscard]] ImTextureID textureId() const {
#ifdef LFS_VULKAN_VIEWER_ENABLED
                return reinterpret_cast<ImTextureID>(descriptor_set_);
#else
                return 0;
#endif
            }

            [[nodiscard]] bool valid() const {
#ifdef LFS_VULKAN_VIEWER_ENABLED
                return descriptor_set_ != VK_NULL_HANDLE && image_view_ != VK_NULL_HANDLE;
#else
                return false;
#endif
            }

        private:
#ifdef LFS_VULKAN_VIEWER_ENABLED
            bool uploadWithStaging(const lfs::core::Tensor& image, const glm::ivec2 size) {
                auto rgba = tensorToRgba8(image, size);
                if (!rgba || !ensureImageResources(size, false)) {
                    return false;
                }

                const VkDeviceSize upload_size = static_cast<VkDeviceSize>(rgba->size());
                VkBuffer staging_buffer = VK_NULL_HANDLE;
                VkDeviceMemory staging_memory = VK_NULL_HANDLE;
                if (!createBuffer(upload_size,
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  staging_buffer,
                                  staging_memory)) {
                    return false;
                }

                void* mapped = nullptr;
                const VkResult map_status = vkMapMemory(device_, staging_memory, 0, upload_size, 0, &mapped);
                if (map_status != VK_SUCCESS || !mapped) {
                    LOG_ERROR("Failed to map Vulkan scene texture staging buffer");
                    vkDestroyBuffer(device_, staging_buffer, nullptr);
                    vkFreeMemory(device_, staging_memory, nullptr);
                    return false;
                }
                std::memcpy(mapped, rgba->data(), rgba->size());
                vkUnmapMemory(device_, staging_memory);

                VkCommandBuffer command_buffer = beginSingleTimeCommands();
                if (command_buffer == VK_NULL_HANDLE) {
                    vkDestroyBuffer(device_, staging_buffer, nullptr);
                    vkFreeMemory(device_, staging_memory, nullptr);
                    return false;
                }

                transitionImageLayout(command_buffer, image_layout_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

                VkBufferImageCopy copy_region{};
                copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copy_region.imageSubresource.mipLevel = 0;
                copy_region.imageSubresource.baseArrayLayer = 0;
                copy_region.imageSubresource.layerCount = 1;
                copy_region.imageExtent = {static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y), 1};
                vkCmdCopyBufferToImage(command_buffer,
                                       staging_buffer,
                                       image_,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       1,
                                       &copy_region);

                transitionImageLayout(command_buffer,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                const bool submitted = endSingleTimeCommands(command_buffer);
                image_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                vkDestroyBuffer(device_, staging_buffer, nullptr);
                vkFreeMemory(device_, staging_memory, nullptr);
                return submitted;
            }

            bool uploadWithCudaInterop(const lfs::core::Tensor& image, const glm::ivec2 size) {
                if (!external_memory_interop_available_ || cuda_interop_disabled_) {
                    return false;
                }
                if (!image.is_valid() || image.device() != lfs::core::Device::CUDA ||
                    image.dtype() != lfs::core::DataType::Float32 || image.ndim() != 3 ||
                    size.x <= 0 || size.y <= 0) {
                    return false;
                }

                const auto layout = lfs::rendering::detectImageLayout(image);
                if (layout == lfs::rendering::ImageLayout::Unknown) {
                    return false;
                }
                const int width = lfs::rendering::imageWidth(image, layout);
                const int height = lfs::rendering::imageHeight(image, layout);
                const int channels = lfs::rendering::imageChannels(image, layout);
                if (width != size.x || height != size.y || (channels != 3 && channels != 4)) {
                    return false;
                }

                if (!ensureImageResources(size, true)) {
                    cuda_interop_disabled_ = true;
                    return false;
                }
                const size_t upload_size = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uchar4);
                if (!ensureCudaUploadBuffer(upload_size)) {
                    cuda_interop_disabled_ = true;
                    return false;
                }

                lfs::core::Tensor formatted = image.is_contiguous() ? image : image.contiguous();
                const float* const src = formatted.ptr<float>();
                if (!src || !cuda_array_) {
                    return false;
                }

                vkDeviceWaitIdle(device_);
                VkCommandBuffer command_buffer = beginSingleTimeCommands();
                if (command_buffer == VK_NULL_HANDLE) {
                    return false;
                }
                transitionImageLayout(command_buffer, image_layout_, VK_IMAGE_LAYOUT_GENERAL);
                if (!endSingleTimeCommands(command_buffer)) {
                    return false;
                }
                image_layout_ = VK_IMAGE_LAYOUT_GENERAL;

                cudaStream_t stream = formatted.stream();
                cudaError_t cuda_status = uploadFloatImageToCudaArray(src,
                                                                      cuda_upload_buffer_,
                                                                      cuda_array_,
                                                                      width,
                                                                      height,
                                                                      channels,
                                                                      layout == lfs::rendering::ImageLayout::CHW,
                                                                      stream);
                if (cuda_status != cudaSuccess) {
                    LOG_WARN("CUDA to Vulkan viewport upload failed: {}", cudaGetErrorString(cuda_status));
                    cuda_interop_disabled_ = true;
                    return false;
                }
                cuda_status = cudaStreamSynchronize(stream);
                if (cuda_status != cudaSuccess) {
                    LOG_WARN("CUDA to Vulkan viewport upload synchronization failed: {}",
                             cudaGetErrorString(cuda_status));
                    cuda_interop_disabled_ = true;
                    return false;
                }

                command_buffer = beginSingleTimeCommands();
                if (command_buffer == VK_NULL_HANDLE) {
                    return false;
                }
                transitionImageLayout(command_buffer,
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                const bool submitted = endSingleTimeCommands(command_buffer);
                if (submitted) {
                    image_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                }
                return submitted;
            }

            [[nodiscard]] uint32_t findMemoryType(const uint32_t type_filter,
                                                  const VkMemoryPropertyFlags properties) const {
                VkPhysicalDeviceMemoryProperties memory_properties{};
                vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties);

                for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
                    const bool supported = (type_filter & (1u << i)) != 0;
                    const bool matches = (memory_properties.memoryTypes[i].propertyFlags & properties) == properties;
                    if (supported && matches) {
                        return i;
                    }
                }
                return std::numeric_limits<uint32_t>::max();
            }

            bool createBuffer(const VkDeviceSize size,
                              const VkBufferUsageFlags usage,
                              const VkMemoryPropertyFlags properties,
                              VkBuffer& buffer,
                              VkDeviceMemory& memory) {
                VkBufferCreateInfo buffer_info{};
                buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                buffer_info.size = size;
                buffer_info.usage = usage;
                buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
                    LOG_ERROR("Failed to create Vulkan scene staging buffer");
                    return false;
                }

                VkMemoryRequirements requirements{};
                vkGetBufferMemoryRequirements(device_, buffer, &requirements);

                VkMemoryAllocateInfo alloc_info{};
                alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                alloc_info.allocationSize = requirements.size;
                alloc_info.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, properties);
                if (alloc_info.memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
                    LOG_ERROR("No suitable Vulkan memory type for scene staging buffer");
                    vkDestroyBuffer(device_, buffer, nullptr);
                    buffer = VK_NULL_HANDLE;
                    return false;
                }

                if (vkAllocateMemory(device_, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
                    LOG_ERROR("Failed to allocate Vulkan scene staging memory");
                    vkDestroyBuffer(device_, buffer, nullptr);
                    buffer = VK_NULL_HANDLE;
                    return false;
                }

                if (vkBindBufferMemory(device_, buffer, memory, 0) != VK_SUCCESS) {
                    LOG_ERROR("Failed to bind Vulkan scene staging memory");
                    vkDestroyBuffer(device_, buffer, nullptr);
                    vkFreeMemory(device_, memory, nullptr);
                    buffer = VK_NULL_HANDLE;
                    memory = VK_NULL_HANDLE;
                    return false;
                }
                return true;
            }

            [[nodiscard]] VkExternalMemoryHandleTypeFlagBits externalMemoryHandleType() const {
#ifdef _WIN32
                return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
                return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
            }

            bool ensureImageResources(const glm::ivec2 size, const bool prefer_external_memory) {
                if (image_ != VK_NULL_HANDLE && size_ == size &&
                    (!prefer_external_memory || image_external_memory_)) {
                    return true;
                }

                vkDeviceWaitIdle(device_);
                destroyImageResources();
                size_ = size;

                VkImageCreateInfo image_info{};
                image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                image_info.imageType = VK_IMAGE_TYPE_2D;
                image_info.extent.width = static_cast<uint32_t>(size.x);
                image_info.extent.height = static_cast<uint32_t>(size.y);
                image_info.extent.depth = 1;
                image_info.mipLevels = 1;
                image_info.arrayLayers = 1;
                image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
                image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
                image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                image_info.samples = VK_SAMPLE_COUNT_1_BIT;
                image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                VkExternalMemoryImageCreateInfo external_image_info{};
                if (prefer_external_memory) {
                    external_image_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
                    external_image_info.handleTypes = externalMemoryHandleType();
                    image_info.pNext = &external_image_info;
                }
                if (vkCreateImage(device_, &image_info, nullptr, &image_) != VK_SUCCESS) {
                    LOG_ERROR("Failed to create Vulkan scene texture image");
                    return false;
                }

                VkMemoryRequirements requirements{};
                vkGetImageMemoryRequirements(device_, image_, &requirements);

                VkMemoryAllocateInfo alloc_info{};
                alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                alloc_info.allocationSize = requirements.size;
                alloc_info.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits,
                                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                VkExportMemoryAllocateInfo export_info{};
                VkMemoryDedicatedAllocateInfo dedicated_info{};
                if (prefer_external_memory) {
                    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
                    export_info.handleTypes = externalMemoryHandleType();
                    alloc_info.pNext = &export_info;
                    if (use_dedicated_external_memory_) {
                        dedicated_info.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
                        dedicated_info.image = image_;
                        export_info.pNext = &dedicated_info;
                    }
                }
                if (alloc_info.memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
                    LOG_ERROR("No suitable Vulkan device-local memory type for scene texture");
                    destroyImageResources();
                    return false;
                }

                if (vkAllocateMemory(device_, &alloc_info, nullptr, &image_memory_) != VK_SUCCESS) {
                    LOG_ERROR("Failed to allocate Vulkan scene texture memory");
                    destroyImageResources();
                    return false;
                }
                if (vkBindImageMemory(device_, image_, image_memory_, 0) != VK_SUCCESS) {
                    LOG_ERROR("Failed to bind Vulkan scene texture memory");
                    destroyImageResources();
                    return false;
                }
                image_memory_size_ = alloc_info.allocationSize;

                if (prefer_external_memory && !importCudaExternalMemory()) {
                    destroyImageResources();
                    return false;
                }

                VkImageViewCreateInfo view_info{};
                view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                view_info.image = image_;
                view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
                view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                view_info.subresourceRange.baseMipLevel = 0;
                view_info.subresourceRange.levelCount = 1;
                view_info.subresourceRange.baseArrayLayer = 0;
                view_info.subresourceRange.layerCount = 1;
                if (vkCreateImageView(device_, &view_info, nullptr, &image_view_) != VK_SUCCESS) {
                    LOG_ERROR("Failed to create Vulkan scene texture image view");
                    destroyImageResources();
                    return false;
                }

                descriptor_set_ = ImGui_ImplVulkan_AddTexture(
                    sampler_,
                    image_view_,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
                image_external_memory_ = prefer_external_memory;
                return descriptor_set_ != VK_NULL_HANDLE;
            }

            void destroyImageResources() {
                destroyCudaImageInterop();
                if (descriptor_set_ != VK_NULL_HANDLE) {
                    ImGui_ImplVulkan_RemoveTexture(descriptor_set_);
                    descriptor_set_ = VK_NULL_HANDLE;
                }
                if (image_view_ != VK_NULL_HANDLE) {
                    vkDestroyImageView(device_, image_view_, nullptr);
                    image_view_ = VK_NULL_HANDLE;
                }
                if (image_ != VK_NULL_HANDLE) {
                    vkDestroyImage(device_, image_, nullptr);
                    image_ = VK_NULL_HANDLE;
                }
                if (image_memory_ != VK_NULL_HANDLE) {
                    vkFreeMemory(device_, image_memory_, nullptr);
                    image_memory_ = VK_NULL_HANDLE;
                }
                image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
                image_external_memory_ = false;
                image_memory_size_ = 0;
                size_ = {0, 0};
            }

            bool importCudaExternalMemory() {
                if (image_memory_ == VK_NULL_HANDLE || image_memory_size_ == 0 || size_.x <= 0 || size_.y <= 0) {
                    return false;
                }

                cudaExternalMemoryHandleDesc handle_desc{};
#ifdef _WIN32
                auto get_memory_handle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
                    vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR"));
                if (!get_memory_handle) {
                    LOG_WARN("Vulkan win32 external memory export entry point is unavailable");
                    return false;
                }

                VkMemoryGetWin32HandleInfoKHR handle_info{};
                handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
                handle_info.memory = image_memory_;
                handle_info.handleType = externalMemoryHandleType();
                HANDLE handle = nullptr;
                const VkResult handle_result = get_memory_handle(device_, &handle_info, &handle);
                if (handle_result != VK_SUCCESS || !handle) {
                    LOG_WARN("Failed to export Vulkan scene texture memory handle: {}",
                             static_cast<int>(handle_result));
                    return false;
                }

                handle_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
                handle_desc.handle.win32.handle = handle;
                handle_desc.size = image_memory_size_;
                if (use_dedicated_external_memory_) {
                    handle_desc.flags = cudaExternalMemoryDedicated;
                }
                const cudaError_t import_status = cudaImportExternalMemory(&cuda_external_memory_, &handle_desc);
                CloseHandle(handle);
                if (import_status != cudaSuccess) {
                    LOG_WARN("Failed to import Vulkan scene texture memory into CUDA: {}",
                             cudaGetErrorString(import_status));
                    return false;
                }
#else
                auto get_memory_fd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
                    vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR"));
                if (!get_memory_fd) {
                    LOG_WARN("Vulkan fd external memory export entry point is unavailable");
                    return false;
                }

                VkMemoryGetFdInfoKHR fd_info{};
                fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
                fd_info.memory = image_memory_;
                fd_info.handleType = externalMemoryHandleType();
                int fd = -1;
                const VkResult fd_result = get_memory_fd(device_, &fd_info, &fd);
                if (fd_result != VK_SUCCESS || fd < 0) {
                    LOG_WARN("Failed to export Vulkan scene texture memory fd: {}", static_cast<int>(fd_result));
                    return false;
                }

                handle_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
                handle_desc.handle.fd = fd;
                handle_desc.size = image_memory_size_;
                if (use_dedicated_external_memory_) {
                    handle_desc.flags = cudaExternalMemoryDedicated;
                }
                const cudaError_t import_status = cudaImportExternalMemory(&cuda_external_memory_, &handle_desc);
                if (import_status != cudaSuccess) {
                    close(fd);
                    LOG_WARN("Failed to import Vulkan scene texture memory into CUDA: {}",
                             cudaGetErrorString(import_status));
                    return false;
                }
#endif

                cudaExternalMemoryMipmappedArrayDesc mip_desc{};
                mip_desc.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
                mip_desc.extent = make_cudaExtent(static_cast<size_t>(size_.x), static_cast<size_t>(size_.y), 0);
                mip_desc.numLevels = 1;
                cudaError_t cuda_status = cudaExternalMemoryGetMappedMipmappedArray(&cuda_mipmap_,
                                                                                    cuda_external_memory_,
                                                                                    &mip_desc);
                if (cuda_status != cudaSuccess) {
                    LOG_WARN("Failed to map Vulkan scene texture memory as a CUDA array: {}",
                             cudaGetErrorString(cuda_status));
                    destroyCudaImageInterop();
                    return false;
                }

                cuda_status = cudaGetMipmappedArrayLevel(&cuda_array_, cuda_mipmap_, 0);
                if (cuda_status != cudaSuccess) {
                    LOG_WARN("Failed to get CUDA scene texture array level: {}", cudaGetErrorString(cuda_status));
                    destroyCudaImageInterop();
                    return false;
                }
                return true;
            }

            void destroyCudaImageInterop() {
                if (cuda_mipmap_ != nullptr || cuda_external_memory_ != nullptr) {
                    cudaDeviceSynchronize();
                }
                if (cuda_mipmap_ != nullptr) {
                    cudaFreeMipmappedArray(cuda_mipmap_);
                    cuda_mipmap_ = nullptr;
                }
                cuda_array_ = nullptr;
                if (cuda_external_memory_ != nullptr) {
                    cudaDestroyExternalMemory(cuda_external_memory_);
                    cuda_external_memory_ = nullptr;
                }
            }

            bool ensureCudaUploadBuffer(const size_t bytes) {
                if (bytes == 0) {
                    return false;
                }
                if (cuda_upload_buffer_ != nullptr && cuda_upload_buffer_size_ >= bytes) {
                    return true;
                }
                destroyCudaUploadBuffer();
                const cudaError_t alloc_status = cudaMalloc(&cuda_upload_buffer_, bytes);
                if (alloc_status != cudaSuccess) {
                    LOG_WARN("Failed to allocate CUDA viewport upload buffer: {}",
                             cudaGetErrorString(alloc_status));
                    cuda_upload_buffer_ = nullptr;
                    cuda_upload_buffer_size_ = 0;
                    return false;
                }
                cuda_upload_buffer_size_ = bytes;
                return true;
            }

            void destroyCudaUploadBuffer() {
                if (cuda_upload_buffer_ != nullptr) {
                    cudaDeviceSynchronize();
                    cudaFree(cuda_upload_buffer_);
                    cuda_upload_buffer_ = nullptr;
                    cuda_upload_buffer_size_ = 0;
                }
            }

            VkCommandBuffer beginSingleTimeCommands() {
                VkCommandBufferAllocateInfo alloc_info{};
                alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                alloc_info.commandPool = command_pool_;
                alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                alloc_info.commandBufferCount = 1;

                VkCommandBuffer command_buffer = VK_NULL_HANDLE;
                if (vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer) != VK_SUCCESS) {
                    LOG_ERROR("Failed to allocate Vulkan scene upload command buffer");
                    return VK_NULL_HANDLE;
                }

                VkCommandBufferBeginInfo begin_info{};
                begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
                    LOG_ERROR("Failed to begin Vulkan scene upload command buffer");
                    vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
                    return VK_NULL_HANDLE;
                }
                return command_buffer;
            }

            bool endSingleTimeCommands(VkCommandBuffer command_buffer) {
                if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
                    LOG_ERROR("Failed to end Vulkan scene upload command buffer");
                    vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
                    return false;
                }

                VkSubmitInfo submit_info{};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &command_buffer;
                const VkResult submit_status = vkQueueSubmit(graphics_queue_, 1, &submit_info, VK_NULL_HANDLE);
                if (submit_status == VK_SUCCESS) {
                    vkQueueWaitIdle(graphics_queue_);
                } else {
                    LOG_ERROR("Failed to submit Vulkan scene upload command buffer: {}", static_cast<int>(submit_status));
                }
                vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
                return submit_status == VK_SUCCESS;
            }

            void transitionImageLayout(VkCommandBuffer command_buffer,
                                       const VkImageLayout old_layout,
                                       const VkImageLayout new_layout) {
                VkImageMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.oldLayout = old_layout;
                barrier.newLayout = new_layout;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = image_;
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                barrier.subresourceRange.baseMipLevel = 0;
                barrier.subresourceRange.levelCount = 1;
                barrier.subresourceRange.baseArrayLayer = 0;
                barrier.subresourceRange.layerCount = 1;

                VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                if (old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
                    new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
                    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                } else if (old_layout == VK_IMAGE_LAYOUT_GENERAL &&
                           new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
                    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    src_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                } else if (new_layout == VK_IMAGE_LAYOUT_GENERAL) {
                    barrier.srcAccessMask = old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                ? VK_ACCESS_SHADER_READ_BIT
                                                : 0;
                    barrier.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                    src_stage = old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                    ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                    : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                    dst_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                } else if (old_layout == VK_IMAGE_LAYOUT_GENERAL &&
                           new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    src_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                    dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                } else if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                    dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                } else {
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                }

                vkCmdPipelineBarrier(command_buffer,
                                     src_stage,
                                     dst_stage,
                                     0,
                                     0,
                                     nullptr,
                                     0,
                                     nullptr,
                                     1,
                                     &barrier);
            }

            VkDevice device_ = VK_NULL_HANDLE;
            VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
            VkQueue graphics_queue_ = VK_NULL_HANDLE;
            uint32_t graphics_queue_family_ = 0;
            VkCommandPool command_pool_ = VK_NULL_HANDLE;
            VkImage image_ = VK_NULL_HANDLE;
            VkDeviceMemory image_memory_ = VK_NULL_HANDLE;
            VkDeviceSize image_memory_size_ = 0;
            VkImageView image_view_ = VK_NULL_HANDLE;
            VkSampler sampler_ = VK_NULL_HANDLE;
            VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
            VkImageLayout image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
            glm::ivec2 size_{0, 0};
            cudaExternalMemory_t cuda_external_memory_ = nullptr;
            cudaMipmappedArray_t cuda_mipmap_ = nullptr;
            cudaArray_t cuda_array_ = nullptr;
            void* cuda_upload_buffer_ = nullptr;
            size_t cuda_upload_buffer_size_ = 0;
            bool external_memory_interop_available_ = false;
            bool use_dedicated_external_memory_ = false;
            bool image_external_memory_ = false;
            bool cuda_interop_disabled_ = false;
#endif
            bool initialized_ = false;
    };

    namespace {

        [[nodiscard]] bool isTransformGizmoOverOrUsing() {
            return isBoundsGizmoHovered() ||
                   isBoundsGizmoActive() ||
                   isRotationGizmoHovered() ||
                   isRotationGizmoActive() ||
                   isScaleGizmoHovered() ||
                   isScaleGizmoActive() ||
                   isTranslationGizmoHovered() ||
                   isTranslationGizmoActive();
        }

        enum class DevResourceKind {
            None,
            Rml,
            Locale
        };

        [[nodiscard]] DevResourceKind devResourceKindForPath(const std::filesystem::path& path) {
            std::string ext = lfs::core::path_to_utf8(path.extension());
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });

            if (ext == ".json")
                return DevResourceKind::Locale;
            if (ext == ".rml" || ext == ".rcss")
                return DevResourceKind::Rml;
            return DevResourceKind::None;
        }

#ifndef LFS_BUILD_PORTABLE
        [[nodiscard]] bool envFlagEnabled(const char* name, const bool default_value) {
            const char* value = std::getenv(name);
            if (!value || !*value)
                return default_value;
            return std::string_view(value) != "0";
        }

        [[nodiscard]] bool envFlagEnabled(const char* name) {
            return envFlagEnabled(name, false);
        }
#endif

        std::string makeRmlTabDomId(const std::string& id) {
            std::string result = "rp-tab-";
            result.reserve(result.size() + id.size());
            for (const char ch : id) {
                const bool keep = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                                  (ch >= '0' && ch <= '9') || ch == '-' || ch == '_';
                result.push_back(keep ? ch : '-');
            }
            return result;
        }

        PanelInputState maskInputForBlockedUi(PanelInputState input) {
            input.mouse_x = -1.0e9f;
            input.mouse_y = -1.0e9f;
            for (auto& value : input.mouse_down)
                value = false;
            for (auto& value : input.mouse_clicked)
                value = false;
            for (auto& value : input.mouse_released)
                value = false;
            input.mouse_wheel = 0.0f;
            input.key_ctrl = false;
            input.key_shift = false;
            input.key_alt = false;
            input.key_super = false;
            input.viewport_keyboard_focus = false;
            input.keys_pressed.clear();
            input.keys_repeated.clear();
            input.keys_released.clear();
            input.text_codepoints.clear();
            input.text_inputs.clear();
            input.text_editing.clear();
            input.text_editing_start = -1;
            input.text_editing_length = -1;
            input.has_text_editing = false;
            return input;
        }

        void applyFrameInputCapture(RmlRightPanel* right_panel = nullptr) {
            const bool panel_hosts_want_keyboard = RmlPanelHost::consumeFrameWantsKeyboard();
            const bool panel_hosts_want_text_input = RmlPanelHost::consumeFrameWantsTextInput();
            if ((panel_hosts_want_keyboard || panel_hosts_want_text_input) && right_panel)
                right_panel->blurFocus();

            auto& focus = guiFocusState();
            if (panel_hosts_want_keyboard)
                focus.want_capture_keyboard = true;
            if (panel_hosts_want_text_input)
                focus.want_text_input = true;
        }

        void syncWindowTextInput(SDL_Window* window) {
            if (!window)
                return;

            const bool wants_text_input = guiFocusState().want_text_input;
            const bool text_input_active = SDL_TextInputActive(window);
            if (wants_text_input == text_input_active)
                return;

            if (wants_text_input)
                SDL_StartTextInput(window);
            else
                SDL_StopTextInput(window);
        }

        SDL_Cursor* systemCursorForImGuiCursor(const ImGuiMouseCursor cursor) {
            switch (cursor) {
            case ImGuiMouseCursor_TextInput: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_TEXT);
                return value;
            }
            case ImGuiMouseCursor_Hand: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_POINTER);
                return value;
            }
            case ImGuiMouseCursor_ResizeEW: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_EW_RESIZE);
                return value;
            }
            case ImGuiMouseCursor_ResizeNS: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NS_RESIZE);
                return value;
            }
            case ImGuiMouseCursor_ResizeNWSE: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NWSE_RESIZE);
                return value;
            }
            case ImGuiMouseCursor_ResizeNESW: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NESW_RESIZE);
                return value;
            }
            case ImGuiMouseCursor_ResizeAll: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_MOVE);
                return value;
            }
            case ImGuiMouseCursor_NotAllowed: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NOT_ALLOWED);
                return value;
            }
            default:
                return nullptr;
            }
        }

        void drawFrameTooltip(const std::string& tip, int screen_w, int screen_h) {
            if (tip.empty())
                return;

            const auto& p = lfs::vis::theme().palette;
            auto* font = ImGui::GetFont();
            const float font_size = ImGui::GetFontSize();
            const ImVec2 mouse = s_frame_input
                                     ? ImVec2(s_frame_input->mouse_x, s_frame_input->mouse_y)
                                     : ImVec2(0, 0);
            const ImVec2 pad(8, 6);
            const ImVec2 text_size = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, tip.c_str());
            const float box_w = text_size.x + pad.x * 2;
            const float box_h = text_size.y + pad.y * 2;

            const float sw = static_cast<float>(screen_w);
            const float sh = static_cast<float>(screen_h);
            ImVec2 box_min(mouse.x + 14, mouse.y + 18);
            if (box_min.x + box_w > sw)
                box_min.x = mouse.x - 14 - box_w;
            if (box_min.y + box_h > sh)
                box_min.y = mouse.y - 18 - box_h;

            const ImVec2 box_max(box_min.x + box_w, box_min.y + box_h);
            const ImU32 col_bg = ImGui::ColorConvertFloat4ToU32(p.surface_bright);
            const ImU32 col_border = ImGui::ColorConvertFloat4ToU32(p.border);
            const ImU32 col_text = ImGui::ColorConvertFloat4ToU32(p.text);

            ImDrawList dl(ImGui::GetDrawListSharedData());
            dl._ResetForNewFrame();
            dl.PushTextureID(ImGui::GetIO().Fonts->TexID);
            dl.PushClipRectFullScreen();
            dl.AddRectFilled(box_min, box_max, col_bg, 4.0f);
            dl.AddRect(box_min, box_max, col_border, 4.0f);
            dl.AddText(font, font_size,
                       ImVec2(box_min.x + pad.x, box_min.y + pad.y), col_text, tip.c_str());
            dl.PopClipRect();

            ImDrawData draw_data{};
            draw_data.DisplayPos = ImVec2(0.0f, 0.0f);
            draw_data.DisplaySize = ImVec2(sw, sh);
            draw_data.FramebufferScale = ImGui::GetIO().DisplayFramebufferScale;
            draw_data.Valid = true;
            draw_data.AddDrawList(&dl);
            ImGui_ImplOpenGL3_RenderDrawData(&draw_data);
        }

        SDL_Cursor* loadColorCursorFromAsset(const std::string& asset_name, int hot_x, int hot_y) {
            try {
                const auto path = lfs::vis::getAssetPath(asset_name);
                const std::string path_utf8 = lfs::core::path_to_utf8(path);
                std::unique_ptr<OIIO::ImageInput> in(OIIO::ImageInput::open(path_utf8));
                if (!in)
                    return nullptr;

                const OIIO::ImageSpec& spec = in->spec();
                const int width = spec.width;
                const int height = spec.height;
                const int channels = spec.nchannels;
                if (width <= 0 || height <= 0 || channels <= 0) {
                    in->close();
                    return nullptr;
                }

                const int read_channels = std::clamp(channels, 1, 4);
                std::vector<unsigned char> source_pixels(static_cast<size_t>(width) * height * read_channels);
                if (!in->read_image(0, 0, 0, read_channels, OIIO::TypeDesc::UINT8, source_pixels.data())) {
                    in->close();
                    return nullptr;
                }
                in->close();

                std::vector<unsigned char> rgba_pixels(static_cast<size_t>(width) * height * 4, 0);
                for (int i = 0; i < width * height; ++i) {
                    const size_t src = static_cast<size_t>(i) * read_channels;
                    const size_t dst = static_cast<size_t>(i) * 4;
                    switch (read_channels) {
                    case 1:
                        rgba_pixels[dst + 0] = source_pixels[src + 0];
                        rgba_pixels[dst + 1] = source_pixels[src + 0];
                        rgba_pixels[dst + 2] = source_pixels[src + 0];
                        rgba_pixels[dst + 3] = 255;
                        break;
                    case 2:
                        rgba_pixels[dst + 0] = source_pixels[src + 0];
                        rgba_pixels[dst + 1] = source_pixels[src + 0];
                        rgba_pixels[dst + 2] = source_pixels[src + 0];
                        rgba_pixels[dst + 3] = source_pixels[src + 1];
                        break;
                    case 3:
                        rgba_pixels[dst + 0] = source_pixels[src + 0];
                        rgba_pixels[dst + 1] = source_pixels[src + 1];
                        rgba_pixels[dst + 2] = source_pixels[src + 2];
                        rgba_pixels[dst + 3] = 255;
                        break;
                    default:
                        rgba_pixels[dst + 0] = source_pixels[src + 0];
                        rgba_pixels[dst + 1] = source_pixels[src + 1];
                        rgba_pixels[dst + 2] = source_pixels[src + 2];
                        rgba_pixels[dst + 3] = source_pixels[src + 3];
                        break;
                    }
                }

                SDL_Surface* surface = SDL_CreateSurfaceFrom(width, height, SDL_PIXELFORMAT_RGBA32,
                                                             rgba_pixels.data(), width * 4);
                if (!surface) {
                    return nullptr;
                }

                SDL_Cursor* cursor = SDL_CreateColorCursor(surface, hot_x, hot_y);
                SDL_DestroySurface(surface);
                return cursor;
            } catch (const std::exception& e) {
                LOG_WARN("Could not load cursor asset '{}': {}", asset_name, e.what());
                return nullptr;
            }
        }
    } // namespace

    GuiManager::GuiManager(VisualizerImpl* viewer)
        : viewer_(viewer),
          sequencer_ui_(viewer, sequencer_ui_state_, &rmlui_manager_),
          gizmo_manager_(viewer),
          async_tasks_(viewer) {

        panel_layout_.loadState();

        // Create components
        menu_bar_ = std::make_unique<MenuBar>();
        rml_modal_overlay_ = std::make_unique<RmlModalOverlay>(&rmlui_manager_);
        global_context_menu_ = std::make_unique<GlobalContextMenu>(&rmlui_manager_);
        lfs::python::set_global_context_menu(global_context_menu_.get());
        video_widget_ = lfs::gui::createVideoWidget();

        // Initialize window states
        window_states_["scene_panel"] = true;
        window_states_["system_console"] = false;
        window_states_["training_tab"] = false;
        window_states_["export_dialog"] = false;
        window_states_["python_console"] = false;

        lfs::python::set_modal_enqueue_callback(
            [this](lfs::core::ModalRequest req) { rml_modal_overlay_->enqueue(std::move(req)); });

        setupEventHandlers();
        async_tasks_.setupEvents();
        sequencer_ui_.setupEvents();
        gizmo_manager_.setupEvents();
        checkCudaVersionAndNotify();
    }

    void GuiManager::checkCudaVersionAndNotify() {
        using namespace lfs::core;
        const auto info = check_cuda_version();
        if (!info.query_failed && !info.supported) {
            pending_cuda_warning_ = info;
        }
    }

    void GuiManager::promptFileAssociation() {
#ifdef _WIN32
        if (file_association_checked_)
            return;
        file_association_checked_ = true;

        LayoutState state;
        state.load();
        if (state.file_association == "declined")
            return;
        if (areFileAssociationsRegistered())
            return;

        using namespace lichtfeld::Strings;
        lfs::core::ModalRequest req;
        req.title = LOC(FileAssociation::TITLE);
        req.body_rml = "<p>" + std::string(LOC(FileAssociation::MESSAGE)) + "</p>";
        req.style = lfs::core::ModalStyle::Info;
        req.buttons = {
            {LOC(FileAssociation::YES), "primary"},
            {LOC(FileAssociation::NOT_NOW), "secondary"},
            {LOC(FileAssociation::DONT_ASK), "secondary"},
        };
        req.on_result = [](const lfs::core::ModalResult& result) {
            LayoutState ls;
            ls.load();

            if (result.button_label == LOC(FileAssociation::YES)) {
                registerFileAssociations();
                openFileAssociationSettings();
                return;
            } else if (result.button_label == LOC(FileAssociation::DONT_ASK)) {
                ls.file_association = "declined";
            } else {
                return;
            }
            ls.save();
        };

        rml_modal_overlay_->enqueue(std::move(req));
#endif
    }

    GuiManager::~GuiManager() = default;

    void GuiManager::initCustomCursors() {
        if (!pipette_cursor_) {
            // The tip of the dropper sits near the lower-left corner in the 24x24 Tabler asset.
            pipette_cursor_ = loadColorCursorFromAsset("icon/color-picker.png", 4, 19);
            if (!pipette_cursor_)
                LOG_WARN("Could not create pipette cursor from icon/color-picker.png");
        }
    }

    void GuiManager::destroyCustomCursors() {
        if (pipette_cursor_) {
            SDL_SetCursor(SDL_GetDefaultCursor());
            SDL_DestroyCursor(pipette_cursor_);
            pipette_cursor_ = nullptr;
        }
    }

    void GuiManager::applyRmlCursorRequest(const RmlCursorRequest req) {
        if (req != RmlCursorRequest::Pipette && pipette_cursor_)
            SDL_SetCursor(SDL_GetDefaultCursor());

        switch (req) {
        case RmlCursorRequest::Arrow:
            ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);
            break;
        case RmlCursorRequest::TextInput:
            ImGui::SetMouseCursor(ImGuiMouseCursor_TextInput);
            break;
        case RmlCursorRequest::Hand:
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            break;
        case RmlCursorRequest::Pipette:
            ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);
            if (pipette_cursor_)
                SDL_SetCursor(pipette_cursor_);
            break;
        case RmlCursorRequest::ResizeEW:
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
            break;
        case RmlCursorRequest::ResizeNS:
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            break;
        case RmlCursorRequest::ResizeNWSE:
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNWSE);
            break;
        case RmlCursorRequest::ResizeNESW:
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNESW);
            break;
        case RmlCursorRequest::ResizeAll:
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
            break;
        case RmlCursorRequest::NotAllowed:
            ImGui::SetMouseCursor(ImGuiMouseCursor_NotAllowed);
            break;
        case RmlCursorRequest::None:
            break;
        }
    }

    void GuiManager::initMenuBar() {
        menu_bar_->setOnShowPythonConsole([this]() {
            window_states_["python_console"] = !window_states_["python_console"];
        });
    }

    FontSet GuiManager::buildFontSet() const {
        FontSet fs{font_regular_, font_bold_, font_heading_, font_small_, font_section_, font_monospace_};
        for (int i = 0; i < FontSet::MONO_SIZE_COUNT; ++i) {
            fs.monospace_sized[i] = mono_fonts_[i];
            fs.monospace_sizes[i] = mono_font_scales_[i];
        }
        return fs;
    }

    void GuiManager::rebuildFonts(float scale) {
        ImGuiIO& io = ImGui::GetIO();

        if (!vulkan_gui_)
            ImGui_ImplOpenGL3_DestroyDeviceObjects();
        io.Fonts->Clear();

        const auto& t = theme();
        try {
            const auto regular_path = lfs::vis::getAssetPath("fonts/" + t.fonts.regular_path);
            const auto bold_path = lfs::vis::getAssetPath("fonts/" + t.fonts.bold_path);
            const auto japanese_path = lfs::vis::getAssetPath("fonts/NotoSansJP-Regular.ttf");
            const auto korean_path = lfs::vis::getAssetPath("fonts/NotoSansKR-Regular.ttf");

            const auto is_font_valid = [](const std::filesystem::path& path) -> bool {
                constexpr size_t MIN_FONT_FILE_SIZE = 100;
                return std::filesystem::exists(path) && std::filesystem::file_size(path) >= MIN_FONT_FILE_SIZE;
            };

            const auto load_font_latin_only =
                [&](const std::filesystem::path& path, const float size) -> ImFont* {
                if (!is_font_valid(path))
                    return nullptr;
                const std::string path_utf8 = lfs::core::path_to_utf8(path);
                ImFontConfig config;
                config.PixelSnapH = true;
                return io.Fonts->AddFontFromFileTTF(path_utf8.c_str(), size, &config);
            };

            const auto merge_cjk = [&](const float size) {
                if (is_font_valid(japanese_path)) {
                    ImFontConfig config;
                    config.MergeMode = true;
                    config.OversampleH = 1;
                    config.PixelSnapH = true;
                    const std::string japanese_path_utf8 = lfs::core::path_to_utf8(japanese_path);
                    io.Fonts->AddFontFromFileTTF(japanese_path_utf8.c_str(), size, &config,
                                                 io.Fonts->GetGlyphRangesJapanese());
                    io.Fonts->AddFontFromFileTTF(japanese_path_utf8.c_str(), size, &config,
                                                 io.Fonts->GetGlyphRangesChineseSimplifiedCommon());
                }
                if (is_font_valid(korean_path)) {
                    ImFontConfig config;
                    config.MergeMode = true;
                    config.OversampleH = 1;
                    config.PixelSnapH = true;
                    const std::string korean_path_utf8 = lfs::core::path_to_utf8(korean_path);
                    io.Fonts->AddFontFromFileTTF(korean_path_utf8.c_str(), size, &config,
                                                 io.Fonts->GetGlyphRangesKorean());
                }
            };

            const auto load_font_with_cjk =
                [&](const std::filesystem::path& path, const float size) -> ImFont* {
                ImFont* font = load_font_latin_only(path, size);
                if (!font)
                    return nullptr;
                merge_cjk(size);
                return font;
            };

            font_regular_ = load_font_with_cjk(regular_path, t.fonts.base_size * scale);
            font_bold_ = load_font_with_cjk(bold_path, t.fonts.base_size * scale);
            font_heading_ = load_font_with_cjk(bold_path, t.fonts.heading_size * scale);
            font_small_ = load_font_with_cjk(regular_path, t.fonts.small_size * scale);
            font_section_ = load_font_with_cjk(bold_path, t.fonts.section_size * scale);

            const auto monospace_path = lfs::vis::getAssetPath("fonts/JetBrainsMono-Regular.ttf");
            if (is_font_valid(monospace_path)) {
                const std::string mono_path_utf8 = lfs::core::path_to_utf8(monospace_path);

                static constexpr ImWchar GLYPH_RANGES[] = {
                    0x0020,
                    0x00FF,
                    0x2190,
                    0x21FF,
                    0x2500,
                    0x257F,
                    0x2580,
                    0x259F,
                    0x25A0,
                    0x25FF,
                    0,
                };

                static constexpr float MONO_SCALES[] = {0.7f, 1.0f, 1.3f, 1.7f, 2.2f};
                static_assert(std::size(MONO_SCALES) == FontSet::MONO_SIZE_COUNT);

                for (int i = 0; i < FontSet::MONO_SIZE_COUNT; ++i) {
                    ImFontConfig config;
                    config.GlyphRanges = GLYPH_RANGES;
                    config.PixelSnapH = true;
                    const float size = t.fonts.base_size * scale * MONO_SCALES[i];
                    mono_fonts_[i] = io.Fonts->AddFontFromFileTTF(mono_path_utf8.c_str(), size, &config);
                    mono_font_scales_[i] = MONO_SCALES[i];
                }
                font_monospace_ = mono_fonts_[1];
            }
            if (!font_monospace_)
                font_monospace_ = font_regular_;

            const bool all_loaded = font_regular_ && font_bold_ && font_heading_ && font_small_ && font_section_;
            if (!all_loaded) {
                ImFont* const fallback = font_regular_ ? font_regular_ : io.Fonts->AddFontDefault();
                if (!font_regular_)
                    font_regular_ = fallback;
                if (!font_bold_)
                    font_bold_ = fallback;
                if (!font_heading_)
                    font_heading_ = fallback;
                if (!font_small_)
                    font_small_ = fallback;
                if (!font_section_)
                    font_section_ = fallback;
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Font loading failed: {}", e.what());
            ImFont* const fallback = io.Fonts->AddFontDefault();
            font_regular_ = font_bold_ = font_heading_ = font_small_ = font_section_ = fallback;
        }

        io.Fonts->TexMinWidth = 2048;
        if (!io.Fonts->Build()) {
            LOG_ERROR("Font atlas build failed — CJK glyphs may be missing");
        }
        if (!vulkan_gui_)
            ImGui_ImplOpenGL3_CreateDeviceObjects();
    }

    void GuiManager::applyUiScale(float scale) {
        scale = std::clamp(scale, 1.0f, 4.0f);
        const float previous_scale = current_ui_scale_;

        rmlui_manager_.setDpRatio(scale);
        lfs::vis::setThemeDpiScale(scale);
        lfs::python::set_shared_dpi_scale(scale);
        PanelRegistry::instance().rescale_floating_panels(previous_scale, scale);
        applyDefaultStyle();
        rebuildFonts(scale);
        current_ui_scale_ = scale;

        LOG_INFO("UI scale applied: {:.2f}", scale);
    }

    void GuiManager::loadImGuiSettings() {
        if (imgui_ini_path_.empty())
            return;

        try {
            if (!std::filesystem::exists(imgui_ini_path_))
                return;

            std::ifstream file;
            if (!lfs::core::open_file_for_read(imgui_ini_path_, std::ios::binary, file)) {
                LOG_WARN("Failed to open ImGui settings file: {}", lfs::core::path_to_utf8(imgui_ini_path_));
                return;
            }

            const std::string ini_data((std::istreambuf_iterator<char>(file)),
                                       std::istreambuf_iterator<char>());
            ImGui::LoadIniSettingsFromMemory(ini_data.c_str(), ini_data.size());
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load ImGui settings: {}", e.what());
        } catch (...) {
            LOG_WARN("Failed to load ImGui settings: unknown error");
        }
    }

    void GuiManager::saveImGuiSettings() const {
        if (imgui_ini_path_.empty() || !ImGui::GetCurrentContext())
            return;

        try {
            std::filesystem::create_directories(imgui_ini_path_.parent_path());

            size_t ini_size = 0;
            const char* ini_data = ImGui::SaveIniSettingsToMemory(&ini_size);

            std::ofstream file;
            if (!lfs::core::open_file_for_write(imgui_ini_path_,
                                                std::ios::binary | std::ios::trunc,
                                                file)) {
                LOG_WARN("Failed to open ImGui settings for writing: {}",
                         lfs::core::path_to_utf8(imgui_ini_path_));
                return;
            }

            file.write(ini_data, static_cast<std::streamsize>(ini_size));
            if (!file) {
                LOG_WARN("Failed to write ImGui settings: {}",
                         lfs::core::path_to_utf8(imgui_ini_path_));
            }
        } catch (const std::exception& e) {
            LOG_WARN("Failed to save ImGui settings: {}", e.what());
        } catch (...) {
            LOG_WARN("Failed to save ImGui settings: unknown error");
        }
    }

    void GuiManager::persistImGuiSettingsIfNeeded() {
        ImGuiIO& io = ImGui::GetIO();
        if (!io.WantSaveIniSettings)
            return;

        saveImGuiSettings();
        io.WantSaveIniSettings = false;
    }

    void GuiManager::init() {
        // ImGui initialization
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();

        // Share ImGui state with Python module across DLL boundaries
        ImGuiContext* const ctx = ImGui::GetCurrentContext();
        lfs::python::set_imgui_context(ctx);

        ImGuiMemAllocFunc alloc_fn{};
        ImGuiMemFreeFunc free_fn{};
        void* alloc_user_data{};
        ImGui::GetAllocatorFunctions(&alloc_fn, &free_fn, &alloc_user_data);
        lfs::python::set_imgui_allocator_functions(
            reinterpret_cast<void*>(alloc_fn),
            reinterpret_cast<void*>(free_fn),
            alloc_user_data);
        lfs::python::set_implot_context(ImPlot::GetCurrentContext());

        vulkan_gui_ = viewer_ && viewer_->getWindowManager() && viewer_->getWindowManager()->isVulkan();

        if (vulkan_gui_) {
            lfs::python::set_gl_texture_service(
                [](const unsigned char*, int, int, int) -> lfs::python::TextureResult {
                    return {0, 0, 0};
                },
                [](uint32_t) {},
                []() -> int {
                    constexpr int FALLBACK_MAX_TEXTURE_SIZE = 4096;
                    return FALLBACK_MAX_TEXTURE_SIZE;
                });
        } else {
            lfs::python::set_gl_texture_service(
                [](const unsigned char* data, const int w, const int h, const int channels) -> lfs::python::TextureResult {
                    if (!data || w <= 0 || h <= 0)
                        return {0, 0, 0};

                    GLuint tex = 0;
                    glGenTextures(1, &tex);
                    if (tex == 0)
                        return {0, 0, 0};

                    glBindTexture(GL_TEXTURE_2D, tex);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

                    GLenum format = GL_RGB;
                    GLenum internal_format = GL_RGB8;
                    if (channels == 1) {
                        format = GL_RED;
                        internal_format = GL_R8;
                    } else if (channels == 4) {
                        format = GL_RGBA;
                        internal_format = GL_RGBA8;
                    }

                    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, format, GL_UNSIGNED_BYTE, data);

                    if (channels == 1) {
                        GLint swizzle[] = {GL_RED, GL_RED, GL_RED, GL_ONE};
                        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzle);
                    }

                    glBindTexture(GL_TEXTURE_2D, 0);
                    return {tex, w, h};
                },
                [](const uint32_t tex) {
                    if (tex > 0) {
                        const auto gl_tex = static_cast<GLuint>(tex);
                        glDeleteTextures(1, &gl_tex);
                    }
                },
                []() -> int {
                    constexpr int FALLBACK_MAX_TEXTURE_SIZE = 4096;
                    GLint sz = 0;
                    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &sz);
                    return sz > 0 ? sz : FALLBACK_MAX_TEXTURE_SIZE;
                });
        }

        ImGuiIO& io = ImGui::GetIO();
        imgui_ini_path_ = LayoutState::getConfigDir() / "imgui.ini";
        io.IniFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
        io.ConfigWindowsMoveFromTitleBarOnly = true;
        io.ConfigDragClickToInputText = true;
        loadImGuiSettings();

        // Platform/Renderer initialization
        if (vulkan_gui_) {
            auto* vulkan_context = viewer_->getWindowManager()->getVulkanContext();
            if (!vulkan_context || !imgui_vulkan_backend_.init(viewer_->getWindow(), *vulkan_context)) {
                throw std::runtime_error("Failed to initialize ImGui Vulkan backend");
            }
        } else {
            ImGui_ImplSDL3_InitForOpenGL(viewer_->getWindow(), SDL_GL_GetCurrentContext());
            ImGui_ImplOpenGL3_Init("#version 430");
        }

        // Initialize localization system
        auto& loc = lfs::event::LocalizationManager::getInstance();
        std::filesystem::path locale_dir = lfs::core::getLocalesDir();
#ifdef LFS_DEV_LOCALE_SOURCE_DIR
        {
            const auto source_locale_dir = lfs::core::utf8_to_path(LFS_DEV_LOCALE_SOURCE_DIR);
            if (std::filesystem::exists(source_locale_dir) &&
                std::filesystem::is_directory(source_locale_dir)) {
                locale_dir = source_locale_dir;
                LOG_INFO("Localization dev source enabled: {}",
                         lfs::core::path_to_utf8(locale_dir));
            }
        }
#endif
        const std::string locale_path = lfs::core::path_to_utf8(locale_dir);
        if (!loc.initialize(locale_path)) {
            LOG_WARN("Failed to initialize localization system, using default strings");
        } else {
            LOG_INFO("Localization initialized with language: {}", loc.getCurrentLanguageName());
        }

        float saved_scale = lfs::vis::loadUiScalePreference();
        if (saved_scale <= 0.0f)
            saved_scale = SDL_GetWindowDisplayScale(viewer_->getWindow());
        current_ui_scale_ = std::clamp(saved_scale, 1.0f, 4.0f);

        lfs::python::set_shared_dpi_scale(current_ui_scale_);
        lfs::vis::setThemeDpiScale(current_ui_scale_);
        initCustomCursors();

        // Set application icon
        try {
            const auto icon_path = lfs::vis::getAssetPath("lichtfeld-icon.png");
            const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(icon_path);

            SDL_Surface* icon_surface = SDL_CreateSurfaceFrom(width, height, SDL_PIXELFORMAT_RGBA32, data, width * 4);
            if (icon_surface) {
                SDL_SetWindowIcon(viewer_->getWindow(), icon_surface);
                SDL_DestroySurface(icon_surface);
            }
            lfs::core::free_image(data);
        } catch (const std::exception& e) {
            LOG_WARN("Could not load application icon: {}", e.what());
        }

        applyDefaultStyle();
        rebuildFonts(current_ui_scale_);

        initMenuBar();

        if (!drag_drop_.init(viewer_->getWindow())) {
            LOG_WARN("Native drag-drop initialization failed, drag-drop will use SDL events only");
        }
        drag_drop_.setFileDropCallback([this](const std::vector<std::string>& paths) {
            LOG_INFO("Files dropped via native drag-drop: {} file(s)", paths.size());
            if (auto* const ic = viewer_->getInputController()) {
                ic->handleFileDrop(paths);
            } else {
                LOG_ERROR("InputController not available for file drop handling");
            }
        });

        if (vulkan_gui_) {
            auto* vulkan_context = viewer_->getWindowManager()->getVulkanContext();
            if (!vulkan_context || !rmlui_manager_.initVulkan(viewer_->getWindow(), *vulkan_context, current_ui_scale_)) {
                throw std::runtime_error("Failed to initialize RmlUI Vulkan backend");
            }
        } else if (!rmlui_manager_.init(viewer_->getWindow(), current_ui_scale_)) {
            throw std::runtime_error("Failed to initialize RmlUI OpenGL backend");
        }
        lfs::vis::setThemeChangeCallback([this](const std::string& theme_id) {
            rmlui_manager_.activateTheme(theme_id);
            if (auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr) {
                rendering->markDirty(DirtyFlag::OVERLAY);
            }
        });
        lfs::python::set_rml_manager(&rmlui_manager_);
        initDevResourceHotReload();

        startup_overlay_.init(&rmlui_manager_);
#ifdef LFS_BUILD_PORTABLE
        const bool startup_overlay_enabled = true;
#else
        const bool startup_overlay_enabled =
            viewer_->options_.show_startup_overlay && !envFlagEnabled("LFS_DISABLE_STARTUP_OVERLAY");
#endif
        if (!startup_overlay_enabled) {
            LOG_INFO("Startup overlay disabled");
            startup_overlay_.dismiss();
        }
        rml_shell_frame_.init(&rmlui_manager_);
        rml_right_panel_.init(&rmlui_manager_);
        rml_right_panel_.on_tab_changed = [this](const std::string& id) {
            panel_layout_.setActiveTab(id);
        };
        rml_right_panel_.on_splitter_delta = [this](float delta_y) {
            viewer_->getRenderingManager()->setViewportResizeActive(true);
            const auto* mvp = ImGui::GetMainViewport();
            ScreenState ss;
            ss.work_pos = {mvp->WorkPos.x, mvp->WorkPos.y};
            ss.work_size = {mvp->WorkSize.x, mvp->WorkSize.y};
            panel_layout_.adjustScenePanelRatio(delta_y, ss);
        };
        rml_right_panel_.on_splitter_end = [this]() {
            viewer_->getRenderingManager()->setViewportResizeActive(false);
        };
        rml_right_panel_.on_resize_delta = [this](float dx) {
            viewer_->getRenderingManager()->setViewportResizeActive(true);
            const auto* mvp = ImGui::GetMainViewport();
            ScreenState ss;
            ss.work_pos = {mvp->WorkPos.x, mvp->WorkPos.y};
            ss.work_size = {mvp->WorkSize.x, mvp->WorkSize.y};
            panel_layout_.applyResizeDelta(dx, ss);
        };
        rml_right_panel_.on_resize_end = [this]() {
            viewer_->getRenderingManager()->setViewportResizeActive(false);
        };
        rml_viewport_overlay_.init(&rmlui_manager_);
        rml_menu_bar_.init(&rmlui_manager_);
        rml_status_bar_.init(&rmlui_manager_);

        lfs::python::RmlPanelHostOps ops{};
        ops.create = [](void* mgr, const char* name, const char* rml,
                        const char* inline_rcss) -> void* {
            return new RmlPanelHost(static_cast<RmlUIManager*>(mgr),
                                    std::string(name), std::string(rml),
                                    inline_rcss ? std::string(inline_rcss) : std::string{});
        };
        ops.destroy = [](void* host) {
            if (lfs::python::on_gl_thread()) {
                delete static_cast<RmlPanelHost*>(host);
            } else {
                lfs::python::schedule_gl_callback([host]() {
                    delete static_cast<RmlPanelHost*>(host);
                });
            }
        };
        ops.draw = [](void* host, const void* ctx) {
            auto* h = static_cast<RmlPanelHost*>(host);
            float aw = ImGui::GetContentRegionAvail().x;
            float ah = ImGui::GetContentRegionAvail().y;
            ImVec2 pos = ImGui::GetCursorScreenPos();

            PanelInputState fallback;
            if (!h->hasInput() && s_frame_input) {
                fallback = buildPanelInputFromSDL(*s_frame_input);
                h->setInput(&fallback);
            }
            h->draw(*static_cast<const PanelDrawContext*>(ctx),
                    aw, ah, pos.x, pos.y);
            h->setInput(nullptr);
        };
        ops.draw_direct = [](void* host, float x, float y, float w, float h) {
            auto* hp = static_cast<RmlPanelHost*>(host);
            PanelInputState fallback;
            if (!hp->hasInput() && s_frame_input) {
                fallback = buildPanelInputFromSDL(*s_frame_input);
                auto* mvp = ImGui::GetMainViewport();
                fallback.bg_draw_list = ImGui::GetForegroundDrawList(mvp);
                fallback.fg_draw_list = ImGui::GetForegroundDrawList(mvp);
                hp->setInput(&fallback);
            }
            hp->drawDirect(x, y, w, h);
            hp->setInput(nullptr);
        };
        ops.prepare_direct = [](void* host, float w, float h) {
            auto* hp = static_cast<RmlPanelHost*>(host);
            PanelInputState fallback;
            if (!hp->hasInput() && s_frame_input) {
                fallback = buildPanelInputFromSDL(*s_frame_input);
                hp->setInput(&fallback);
            }
            hp->prepareDirect(w, h);
            hp->setInput(nullptr);
        };
        ops.prepare_layout = [](void* host, float w, float h) {
            static_cast<RmlPanelHost*>(host)->syncDirectLayout(w, h);
        };
        ops.get_document = [](void* host) -> void* {
            return static_cast<RmlPanelHost*>(host)->getDocument();
        };
        ops.is_loaded = [](void* host) -> bool {
            return static_cast<RmlPanelHost*>(host)->isDocumentLoaded();
        };
        ops.set_height_mode = [](void* host, int mode) {
            static_cast<RmlPanelHost*>(host)->setHeightMode(
                static_cast<PanelHeightMode>(mode));
        };
        ops.get_content_height = [](void* host) -> float {
            return static_cast<RmlPanelHost*>(host)->getContentHeight();
        };
        ops.ensure_context = [](void* host) -> bool {
            return static_cast<RmlPanelHost*>(host)->ensureContext();
        };
        ops.ensure_document = [](void* host) -> bool {
            return static_cast<RmlPanelHost*>(host)->ensureDocumentLoaded();
        };
        ops.reload_document = [](void* host) -> bool {
            return static_cast<RmlPanelHost*>(host)->reloadDocument();
        };
        ops.get_context = [](void* host) -> void* {
            return static_cast<RmlPanelHost*>(host)->getContext();
        };
        ops.set_foreground = [](void* host, bool fg) {
            static_cast<RmlPanelHost*>(host)->setForeground(fg);
        };
        ops.mark_content_dirty = [](void* host) {
            static_cast<RmlPanelHost*>(host)->markContentDirty();
        };
        ops.set_input_clip_y = [](void* host, float y_min, float y_max) {
            static_cast<RmlPanelHost*>(host)->setInputClipY(y_min, y_max);
        };
        ops.set_input = [](void* host, const void* input) {
            static_cast<RmlPanelHost*>(host)->setInput(
                static_cast<const PanelInputState*>(input));
        };
        ops.set_forced_height = [](void* host, float h) {
            static_cast<RmlPanelHost*>(host)->setForcedHeight(h);
        };
        ops.needs_animation = [](void* host) -> bool {
            return static_cast<RmlPanelHost*>(host)->needsAnimationFrame();
        };
        lfs::python::set_rml_panel_host_ops(ops);

        registerNativePanels();
    }

    void GuiManager::initDevResourceHotReload() {
        dev_resource_watch_ = {};

#if !defined(LFS_BUILD_PORTABLE) && (defined(LFS_DEV_RMLUI_SOURCE_DIR) || defined(LFS_DEV_LOCALE_SOURCE_DIR))
        if (!envFlagEnabled("LFS_RESOURCE_HOT_RELOAD", true))
            return;

#ifdef LFS_DEV_RMLUI_SOURCE_DIR
        {
            const auto dir = lfs::core::utf8_to_path(LFS_DEV_RMLUI_SOURCE_DIR);
            if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir))
                dev_resource_watch_.rml_dir = dir;
        }
#endif
#ifdef LFS_DEV_LOCALE_SOURCE_DIR
        {
            const auto dir = lfs::core::utf8_to_path(LFS_DEV_LOCALE_SOURCE_DIR);
            if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir))
                dev_resource_watch_.locale_dir = dir;
        }
#endif

        dev_resource_watch_.enabled =
            !dev_resource_watch_.rml_dir.empty() || !dev_resource_watch_.locale_dir.empty();
        if (!dev_resource_watch_.enabled)
            return;

        scanDevResourceFiles(false);
        dev_resource_watch_.next_scan = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        LOG_INFO("Resource hot reload enabled (RmlUI: '{}', locales: '{}')",
                 dev_resource_watch_.rml_dir.empty() ? std::string("<disabled>")
                                                     : lfs::core::path_to_utf8(dev_resource_watch_.rml_dir),
                 dev_resource_watch_.locale_dir.empty() ? std::string("<disabled>")
                                                        : lfs::core::path_to_utf8(dev_resource_watch_.locale_dir));
#endif
    }

    std::pair<bool, bool> GuiManager::scanDevResourceFiles(const bool detect_changes) {
        std::unordered_map<std::string, std::filesystem::file_time_type> next_times;
        bool rml_changed = false;
        bool locale_changed = false;

        const auto scan_dir =
            [&](const std::filesystem::path& dir, const bool locale_dir) {
                if (dir.empty())
                    return;

                std::error_code ec;
                if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec))
                    return;

                std::filesystem::recursive_directory_iterator it(
                    dir, std::filesystem::directory_options::skip_permission_denied, ec);
                const std::filesystem::recursive_directory_iterator end;
                for (; !ec && it != end; it.increment(ec)) {
                    std::error_code entry_ec;
                    if (!it->is_regular_file(entry_ec) || entry_ec)
                        continue;

                    const auto kind = devResourceKindForPath(it->path());
                    const bool watched = locale_dir
                                             ? kind == DevResourceKind::Locale
                                             : kind == DevResourceKind::Rml;
                    if (!watched)
                        continue;

                    const auto mtime = std::filesystem::last_write_time(it->path(), entry_ec);
                    if (entry_ec)
                        continue;

                    const std::string key = lfs::core::path_to_utf8(it->path().lexically_normal());
                    next_times[key] = mtime;

                    if (!detect_changes)
                        continue;

                    const auto old = dev_resource_watch_.file_times.find(key);
                    if (old == dev_resource_watch_.file_times.end() || old->second != mtime) {
                        if (locale_dir)
                            locale_changed = true;
                        else
                            rml_changed = true;
                    }
                }
            };

        scan_dir(dev_resource_watch_.rml_dir, false);
        scan_dir(dev_resource_watch_.locale_dir, true);

        if (detect_changes) {
            for (const auto& [key, unused] : dev_resource_watch_.file_times) {
                (void)unused;
                if (next_times.contains(key))
                    continue;

                const auto kind = devResourceKindForPath(lfs::core::utf8_to_path(key));
                if (kind == DevResourceKind::Locale)
                    locale_changed = true;
                else if (kind == DevResourceKind::Rml)
                    rml_changed = true;
            }
        }

        dev_resource_watch_.file_times = std::move(next_times);
        return {rml_changed, locale_changed};
    }

    bool GuiManager::reloadLocalizationResources() {
        if (dev_resource_watch_.locale_dir.empty())
            return false;

        auto& loc = lfs::event::LocalizationManager::getInstance();
        const std::string current_language = loc.getCurrentLanguage();
        const std::string locale_path = lfs::core::path_to_utf8(dev_resource_watch_.locale_dir);
        if (!loc.initialize(locale_path)) {
            LOG_WARN("Failed to reload localization resources from {}", locale_path);
            return false;
        }

        if (!current_language.empty() && current_language != loc.getCurrentLanguage()) {
            const auto available = loc.getAvailableLanguages();
            if (std::find(available.begin(), available.end(), current_language) != available.end())
                loc.setLanguage(current_language);
        }

        return true;
    }

    void GuiManager::reloadRmlResources() {
        rml_theme::invalidateBaseRcssCache();
        rml_theme::invalidateThemeMediaCache();

        startup_overlay_.reloadResources();
        rml_shell_frame_.reloadResources();
        rml_right_panel_.reloadResources();
        rml_viewport_overlay_.reloadResources();
        rml_menu_bar_.reloadResources();
        rml_status_bar_.reloadResources();
        sequencer_ui_.reloadRmlResources();
        PanelRegistry::instance().reload_rml_resources();

        if (rml_modal_overlay_)
            rml_modal_overlay_->reloadResources();
        if (global_context_menu_)
            global_context_menu_->reloadResources();

        if (auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr)
            rendering->markDirty(DirtyFlag::OVERLAY);
    }

    bool GuiManager::shouldDeferDevResourceHotReload() const {
        if (ImGui::GetCurrentContext()) {
            const ImGuiIO& io = ImGui::GetIO();
            if (io.WantTextInput || ImGui::IsAnyItemActive() ||
                ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
                ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
                ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
                return true;
            }
        }

        if (!ui_hidden_ && rml_menu_bar_.isOpen())
            return true;
        if (global_context_menu_ && global_context_menu_->isOpen())
            return true;
        if (rml_modal_overlay_ && rml_modal_overlay_->isOpen())
            return true;

        return false;
    }

    void GuiManager::pollDevResourceHotReload() {
        if (!dev_resource_watch_.enabled)
            return;

        const auto now = std::chrono::steady_clock::now();
        if (dev_resource_watch_.next_scan != std::chrono::steady_clock::time_point{} &&
            now < dev_resource_watch_.next_scan) {
            return;
        }
        dev_resource_watch_.next_scan = now + std::chrono::seconds(1);

        const auto [rml_changed, locale_changed] = scanDevResourceFiles(true);
        dev_resource_watch_.pending_rml_reload |= rml_changed;
        dev_resource_watch_.pending_locale_reload |= locale_changed;

        if (!dev_resource_watch_.pending_rml_reload &&
            !dev_resource_watch_.pending_locale_reload) {
            return;
        }

        if (shouldDeferDevResourceHotReload())
            return;

        const bool reload_rml = dev_resource_watch_.pending_rml_reload;
        const bool reload_locale = dev_resource_watch_.pending_locale_reload;
        dev_resource_watch_.pending_rml_reload = false;
        dev_resource_watch_.pending_locale_reload = false;

        if (reload_locale)
            reloadLocalizationResources();
        if (reload_rml || reload_locale)
            reloadRmlResources();

        LOG_INFO("Hot-reloaded dev resources{}{}",
                 reload_rml ? " (RmlUI)" : "",
                 reload_locale ? " (locales)" : "");
    }

    void GuiManager::shutdown() {
        panel_layout_.saveState();

        if (video_widget_)
            video_widget_->shutdown();

        async_tasks_.shutdown();

        const bool need_gil = lfs::python::get_main_thread_state() != nullptr;
        if (need_gil)
            lfs::python::acquire_gil_main_thread();

        lfs::python::shutdown_python_gl_resources();
        lfs::python::set_modal_enqueue_callback({});

        global_context_menu_->destroyGLResources();
        rml_modal_overlay_.reset();
        panels::ShutdownPythonConsoleRml();
        rml_status_bar_.shutdown();
        rml_menu_bar_.shutdown();
        rml_viewport_overlay_.shutdown();
        rml_right_panel_.shutdown();
        rml_shell_frame_.shutdown();
        startup_overlay_.shutdown();
        PanelRegistry::instance().unregister_all_non_native();
        rmlui_manager_.shutdown();

        if (need_gil)
            lfs::python::release_gil_main_thread();

        sequencer_ui_.destroyGLResources();
        drag_drop_.shutdown();
        destroyCustomCursors();
        vulkan_scene_uploaded_image_.reset();
        vulkan_scene_image_.reset();
        vulkan_scene_texture_.reset();

        if (ImGui::GetCurrentContext()) {
            saveImGuiSettings();
            if (vulkan_gui_) {
                imgui_vulkan_backend_.shutdown();
            } else {
                ImGui_ImplOpenGL3_Shutdown();
                ImGui_ImplSDL3_Shutdown();
            }
            ImPlot::DestroyContext();
            ImGui::DestroyContext();
        }
        vulkan_gui_ = false;
    }

    void GuiManager::registerNativePanels() {
        using namespace native_panels;
        auto& reg = PanelRegistry::instance();

        auto make_panel = [this](auto panel) -> std::shared_ptr<IPanel> {
            auto ptr = std::make_shared<decltype(panel)>(std::move(panel));
            native_panel_storage_.push_back(ptr);
            return ptr;
        };

        auto reg_panel = [&](const std::string& id, const std::string& label,
                             std::shared_ptr<IPanel> panel, PanelSpace space, int order,
                             uint32_t options = 0, float initial_width = 0, float initial_height = 0) {
            PanelInfo info;
            info.panel = std::move(panel);
            info.label = label;
            info.id = id;
            info.space = space;
            info.order = order;
            info.options = options;
            info.is_native = true;
            info.initial_width = initial_width;
            info.initial_height = initial_height;
            reg.register_panel(std::move(info));
        };

        // Floating panels (self-managed windows)
        {
            auto panel = std::static_pointer_cast<IPanel>(
                std::make_shared<NativeScenePanel>(&rmlui_manager_));
            native_panel_storage_.push_back(panel);
            reg_panel("lfs.scene", "Scene", panel, PanelSpace::SceneHeader, 0);
        }

        reg_panel("native.video_extractor", "Video Extractor",
                  make_panel(VideoExtractorPanel(video_widget_.get())),
                  PanelSpace::Floating, 11,
                  static_cast<uint32_t>(PanelOption::SELF_MANAGED),
                  750.0f);
        reg.set_panel_enabled("native.video_extractor", false);

        // Viewport overlays (ordered by draw priority)
        reg_panel("native.selection_overlay", "Selection Overlay",
                  make_panel(SelectionOverlayPanel(this)),
                  PanelSpace::ViewportOverlay, 200);

        reg_panel("native.node_transform_gizmo", "Node Transform",
                  make_panel(NodeTransformGizmoPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 300);

        reg_panel("native.cropbox_gizmo", "Crop Box",
                  make_panel(CropBoxGizmoPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 301);

        reg_panel("native.ellipsoid_gizmo", "Ellipsoid",
                  make_panel(EllipsoidGizmoPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 302);

        reg_panel("native.sequencer", "Sequencer",
                  make_panel(SequencerPanel(&sequencer_ui_, &panel_layout_)),
                  PanelSpace::BottomDock, 500,
                  0, 8192.0f);

        reg_panel("native.python_overlay", "Python Overlay",
                  make_panel(PythonOverlayPanel(this)),
                  PanelSpace::ViewportOverlay, 500);

        reg_panel("native.viewport_decorations", "Viewport Decorations",
                  make_panel(ViewportDecorationsPanel(this)),
                  PanelSpace::ViewportOverlay, 800);

        reg_panel("native.viewport_gizmo", "Viewport Gizmo",
                  make_panel(ViewportGizmoPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 900);

        reg_panel("native.pie_menu", "Pie Menu",
                  make_panel(PieMenuPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 950);

        reg_panel("native.startup_overlay", "Startup Overlay",
                  make_panel(StartupOverlayPanel(&startup_overlay_, &drag_drop_hovering_)),
                  PanelSpace::ViewportOverlay, 0);
    }

    void GuiManager::setVulkanSceneImage(std::shared_ptr<const lfs::core::Tensor> image,
                                         const glm::ivec2 size,
                                         const bool flip_y) {
        vulkan_scene_image_ = std::move(image);
        vulkan_scene_image_size_ = size;
        vulkan_scene_image_flip_y_ = flip_y;
        if (!vulkan_scene_image_) {
            vulkan_scene_uploaded_image_.reset();
            vulkan_scene_uploaded_size_ = {0, 0};
        }
    }

    void GuiManager::renderVulkan() {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        auto* window_manager = viewer_ ? viewer_->getWindowManager() : nullptr;
        auto* vulkan_context = window_manager ? window_manager->getVulkanContext() : nullptr;
        if (!vulkan_context || !imgui_vulkan_backend_.initialized())
            return;

        if (vulkan_scene_image_ && vulkan_scene_image_size_.x > 0 && vulkan_scene_image_size_.y > 0) {
            if (!vulkan_scene_texture_) {
                vulkan_scene_texture_ = std::make_unique<VulkanSceneTexture>();
            }
            bool texture_ready = vulkan_scene_texture_->init(*vulkan_context);
            const bool needs_upload =
                texture_ready &&
                (vulkan_scene_uploaded_image_.get() != vulkan_scene_image_.get() ||
                 vulkan_scene_uploaded_size_ != vulkan_scene_image_size_ ||
                 !vulkan_scene_texture_->valid());
            if (needs_upload) {
                const bool enable_cuda_interop = viewer_ && viewer_->options_.enable_cuda_interop;
                texture_ready = vulkan_scene_texture_->upload(
                    *vulkan_scene_image_,
                    vulkan_scene_image_size_,
                    enable_cuda_interop);
                if (texture_ready) {
                    vulkan_scene_uploaded_image_ = vulkan_scene_image_;
                    vulkan_scene_uploaded_size_ = vulkan_scene_image_size_;
                }
            }
            if (texture_ready && vulkan_scene_texture_->valid()) {
                ImGuiViewport* const viewport = ImGui::GetMainViewport();
                const bool has_viewport_layout =
                    viewport_layout_.size.x > 0.0f && viewport_layout_.size.y > 0.0f;
                const ImVec2 p0 = has_viewport_layout
                                       ? ImVec2(viewport_layout_.pos.x, viewport_layout_.pos.y)
                                       : (viewport ? viewport->Pos : ImVec2(0.0f, 0.0f));
                const ImVec2 size = has_viewport_layout
                                        ? ImVec2(viewport_layout_.size.x, viewport_layout_.size.y)
                                        : (viewport ? viewport->Size
                                                    : ImVec2(
                                                          static_cast<float>(vulkan_scene_image_size_.x),
                                                          static_cast<float>(vulkan_scene_image_size_.y)));
                const ImVec2 p1(p0.x + size.x, p0.y + size.y);
                const ImVec2 uv0(0.0f, vulkan_scene_image_flip_y_ ? 1.0f : 0.0f);
                const ImVec2 uv1(1.0f, vulkan_scene_image_flip_y_ ? 0.0f : 1.0f);
                ImDrawList* const draw_list = viewport ? ImGui::GetBackgroundDrawList(viewport)
                                                       : ImGui::GetBackgroundDrawList();
                draw_list->AddImage(
                    vulkan_scene_texture_->textureId(),
                    p0,
                    p1,
                    uv0,
                    uv1);
            }
        }
#else
        LOG_ERROR("Vulkan GUI render requested, but Vulkan viewer dependencies are disabled");
#endif
    }

    void GuiManager::render() {
        auto* window_manager = viewer_ ? viewer_->getWindowManager() : nullptr;
#ifdef LFS_VULKAN_VIEWER_ENABLED
        auto* vulkan_context = (vulkan_gui_ && window_manager) ? window_manager->getVulkanContext() : nullptr;
        if (vulkan_gui_ && (!vulkan_context || !imgui_vulkan_backend_.initialized()))
            return;
#endif

        if (!vulkan_gui_ && rmlui_manager_.getRenderInterface()) {
            auto* ri = rmlui_manager_.getRenderInterface();
            auto* sm = viewer_->getSceneManager();
            ri->set_scene_manager(sm);
            ri->process_pending_preview_uploads();
        }

        if (pending_cuda_warning_) {
            constexpr int MIN_MAJOR = lfs::core::MIN_CUDA_VERSION / 1000;
            constexpr int MIN_MINOR = (lfs::core::MIN_CUDA_VERSION % 1000) / 10;
            lfs::core::events::state::CudaVersionUnsupported{
                .major = pending_cuda_warning_->major,
                .minor = pending_cuda_warning_->minor,
                .min_major = MIN_MAJOR,
                .min_minor = MIN_MINOR}
                .emit();
            pending_cuda_warning_.reset();
        }

        promptFileAssociation();

        if (pending_ui_scale_ > 0.0f) {
            applyUiScale(pending_ui_scale_);
            pending_ui_scale_ = 0.0f;
        }

        drag_drop_.pollEvents();
        drag_drop_hovering_ = drag_drop_.isDragHovering();

        if (auto* input_controller = viewer_->getInputController()) {
            input_controller->getBindings().updateCapture();
        }

        // Start frame
        if (vulkan_gui_) {
            imgui_vulkan_backend_.newFrame();
            rmlui_manager_.clearVulkanQueue();
        } else {
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplSDL3_NewFrame();
        }
        const auto& sdl_input = viewer_->getWindowManager()->frameInput();

        // Check mouse state before ImGui::NewFrame() updates WantCaptureMouse
        const bool mouse_in_viewport = isPositionInViewport(sdl_input.mouse_x, sdl_input.mouse_y);

        ImGui::NewFrame();

        {
            auto& focus = guiFocusState();
            focus.reset();
            focus.want_capture_mouse = ImGui::GetIO().WantCaptureMouse;
            focus.want_capture_keyboard = ImGui::GetIO().WantCaptureKeyboard;
            focus.want_text_input = ImGui::GetIO().WantTextInput;
        }

        // Run queued Python/UI mutations before panel registries take draw snapshots.
        python::flush_gl_callbacks();

        rmlui_manager_.beginFrameCursorTracking();
        const bool modal_overlay_open = rml_modal_overlay_->isOpen();
        const bool context_menu_open = global_context_menu_ && global_context_menu_->isOpen();
        const bool block_underlay_input = modal_overlay_open || context_menu_open;

        if (ImGui::IsKeyPressed(ImGuiKey_Escape) && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId)) {
            auto* console_state = panels::PythonConsoleState::tryGetInstance();
            auto* editor = console_state ? console_state->getEditor() : nullptr;
            const bool editor_owns_escape =
                editor && (editor->isFocused() || editor->hasActiveCompletion());
            if (!editor_owns_escape) {
                widgets::RequestActiveEditCancel();
                ImGui::ClearActiveID();
                if (editor != nullptr) {
                    editor->unfocus();
                }
            }
        }

        // Check for async import completion (must happen on main thread)
        async_tasks_.pollImportCompletion();
        async_tasks_.pollMesh2SplatCompletion();
        async_tasks_.pollSplatSimplifyCompletion();

        // Poll UV package manager for async operations
        python::PackageManager::instance().poll();

        pollDevResourceHotReload();

        // Hot-reload themes (check once per second)
        {
            static auto last_check = std::chrono::steady_clock::now();
            const auto now = std::chrono::steady_clock::now();
            if (now - last_check > std::chrono::seconds(1)) {
                if (checkThemeFileChanges()) {
                    rml_theme::invalidateThemeMediaCache();
                }
                last_check = now;
            }
        }

        if (menu_bar_ && !ui_hidden_) {
            menu_bar_->render();

            if (menu_bar_->hasMenuEntries()) {
                auto entries = menu_bar_->getMenuEntries();
                std::vector<std::string> labels;
                std::vector<std::string> idnames;
                labels.reserve(entries.size());
                idnames.reserve(entries.size());
                for (const auto& entry : entries) {
                    labels.emplace_back(LOC(entry.label.c_str()));
                    idnames.emplace_back(entry.idname);
                }
                rml_menu_bar_.updateLabels(labels, idnames);
            } else {
                rml_menu_bar_.updateLabels({}, {});
            }

            // Reserve work area for the RML menu bar via ImGui's internal inset mechanism
            {
                auto* vp = static_cast<ImGuiViewportP*>(ImGui::GetMainViewport());
                float bar_h = rml_menu_bar_.barHeight();
                vp->BuildWorkInsetMin.y = ImMax(vp->BuildWorkInsetMin.y, bar_h);
                vp->WorkInsetMin.y = ImMax(vp->WorkInsetMin.y, bar_h);
                vp->UpdateWorkRect();
            }

            PanelInputState menu_input = buildPanelInputFromSDL(sdl_input);
            if (const ImGuiViewport* const main_viewport = ImGui::GetMainViewport()) {
                menu_input.screen_x = main_viewport->Pos.x;
                menu_input.screen_y = main_viewport->Pos.y;
                menu_input.screen_w = static_cast<int>(main_viewport->Size.x);
                menu_input.screen_h = static_cast<int>(main_viewport->Size.y);
            }
            if (block_underlay_input)
                menu_input = maskInputForBlockedUi(std::move(menu_input));

            rml_menu_bar_.processInput(menu_input);

            if (rml_menu_bar_.wantsInput())
                guiFocusState().want_capture_mouse = true;

            rml_menu_bar_.draw(menu_input.screen_w, menu_input.screen_h);
        } else {
            rml_menu_bar_.suspend();
        }

        PanelInputState frame_input = buildPanelInputFromSDL(sdl_input);
        updateInputOverrides(frame_input, mouse_in_viewport);
        if (auto* const wm = viewer_->getWindowManager()) {
            frame_input.viewport_keyboard_focus = wm->inputRouter().isViewportKeyboardFocused();
        }

        auto& reg = PanelRegistry::instance();

        if (!ui_hidden_) {
            const auto* mvp = ImGui::GetMainViewport();
            const float status_bar_h = PanelLayoutManager::STATUS_BAR_HEIGHT * current_ui_scale_;
            const float panel_h = mvp->WorkSize.y - status_bar_h;

            ShellRegions shell_regions;
            shell_regions.screen = {mvp->Pos.x, mvp->Pos.y, mvp->Size.x, mvp->Size.y};
            shell_regions.menu = {mvp->Pos.x, mvp->Pos.y,
                                  mvp->Size.x, mvp->WorkPos.y - mvp->Pos.y};

            if (show_main_panel_) {
                const float rpw = panel_layout_.getRightPanelWidth();
                shell_regions.right_panel = {
                    mvp->WorkPos.x + mvp->WorkSize.x - rpw,
                    mvp->WorkPos.y,
                    rpw,
                    panel_h,
                };
            }

            shell_regions.status = {
                mvp->WorkPos.x,
                mvp->WorkPos.y + mvp->WorkSize.y - status_bar_h,
                mvp->WorkSize.x,
                status_bar_h,
            };

            rml_shell_frame_.render(shell_regions);
        }

        // Update editor context state for this frame
        auto& editor_ctx = viewer_->getEditorContext();
        editor_ctx.update(viewer_->getSceneManager(), viewer_->getTrainerManager());

        // Create context for this frame
        UIContext ctx{
            .viewer = viewer_,
            .window_states = &window_states_,
            .editor = &editor_ctx,
            .sequencer_controller = &sequencer_ui_.controller(),
            .rml_manager = &rmlui_manager_,
            .fonts = buildFontSet()};

        // Build draw context for panel registry
        lfs::core::Scene* scene = nullptr;
        if (auto* sm = ctx.viewer->getSceneManager()) {
            scene = &sm->getScene();
        }
        PanelDrawContext draw_ctx;
        draw_ctx.ui = &ctx;
        draw_ctx.viewport = &viewport_layout_;
        draw_ctx.scene = scene;
        draw_ctx.ui_hidden = ui_hidden_;
        draw_ctx.frame_serial = ++panel_frame_serial_;
        draw_ctx.scene_generation = python::get_scene_generation();
        if (auto* sm = ctx.viewer->getSceneManager())
            draw_ctx.has_selection = sm->hasSelectedNode();
        if (auto* cc = lfs::event::command_center())
            draw_ctx.is_training = cc->snapshot().is_running;

        reg.preload_panels(PanelSpace::SceneHeader, draw_ctx);
        reg.preload_panels(PanelSpace::SidePanel, draw_ctx);

        auto* mvp_input = ImGui::GetMainViewport();
        s_frame_input = &sdl_input;
        PanelInputState panel_input = frame_input;
        panel_input.screen_x = mvp_input->Pos.x;
        panel_input.screen_y = mvp_input->Pos.y;
        panel_input.bg_draw_list = ImGui::GetBackgroundDrawList(mvp_input);
        panel_input.fg_draw_list = ImGui::GetForegroundDrawList(mvp_input);
        PanelInputState raw_panel_input = panel_input;
        if (block_underlay_input)
            panel_input = maskInputForBlockedUi(std::move(panel_input));
        RmlPanelHost::clearQueuedForegroundComposites();
        if (!modal_overlay_open)
            global_context_menu_->processInput(raw_panel_input);

        ScreenState screen;
        screen.work_pos = {mvp_input->WorkPos.x, mvp_input->WorkPos.y};
        screen.work_size = {mvp_input->WorkSize.x, mvp_input->WorkSize.y};
        screen.any_item_active = ImGui::IsAnyItemActive();

        constexpr uint8_t kUiLayoutSettleFrames = 3;
        const bool python_console_visible = window_states_["python_console"];
        const bool ui_layout_changed =
            std::abs(screen.work_pos.x - last_ui_layout_work_pos_.x) > 0.5f ||
            std::abs(screen.work_pos.y - last_ui_layout_work_pos_.y) > 0.5f ||
            std::abs(screen.work_size.x - last_ui_layout_work_size_.x) > 0.5f ||
            std::abs(screen.work_size.y - last_ui_layout_work_size_.y) > 0.5f ||
            std::abs(panel_layout_.getRightPanelWidth() - last_ui_layout_right_panel_w_) > 0.5f ||
            std::abs(panel_layout_.getScenePanelRatio() - last_ui_layout_scene_ratio_) > 0.0001f ||
            std::abs(panel_layout_.getPythonConsoleWidth() - last_ui_layout_python_console_w_) > 0.5f ||
            std::abs(panel_layout_.getBottomDockHeight() - last_ui_layout_bottom_dock_h_) > 0.5f ||
            show_main_panel_ != last_ui_layout_show_main_panel_ ||
            ui_hidden_ != last_ui_layout_ui_hidden_ ||
            python_console_visible != last_ui_layout_python_console_visible_ ||
            panel_layout_.isBottomDockVisible() != last_ui_layout_bottom_dock_visible_ ||
            panel_layout_.getActiveTab() != last_ui_layout_active_tab_;

        if (ui_layout_changed) {
            ui_layout_settle_frames_ = kUiLayoutSettleFrames;
            last_ui_layout_work_pos_ = screen.work_pos;
            last_ui_layout_work_size_ = screen.work_size;
            last_ui_layout_right_panel_w_ = panel_layout_.getRightPanelWidth();
            last_ui_layout_scene_ratio_ = panel_layout_.getScenePanelRatio();
            last_ui_layout_python_console_w_ = panel_layout_.getPythonConsoleWidth();
            last_ui_layout_bottom_dock_h_ = panel_layout_.getBottomDockHeight();
            last_ui_layout_show_main_panel_ = show_main_panel_;
            last_ui_layout_ui_hidden_ = ui_hidden_;
            last_ui_layout_python_console_visible_ = python_console_visible;
            last_ui_layout_bottom_dock_visible_ = panel_layout_.isBottomDockVisible();
            last_ui_layout_active_tab_ = panel_layout_.getActiveTab();
        }

        if (show_main_panel_ && !ui_hidden_) {
            const float sbh = PanelLayoutManager::STATUS_BAR_HEIGHT * current_ui_scale_;
            const float rpw = panel_layout_.getRightPanelWidth();
            const float ph = screen.work_size.y - sbh;
            const float splitter_h = PanelLayoutManager::SPLITTER_H * current_ui_scale_;
            const float avail_h = ph - 16.0f;
            const float scene_h = std::max(80.0f * current_ui_scale_,
                                           avail_h * panel_layout_.getScenePanelRatio() - splitter_h * 0.5f);

            RightPanelLayout rp_layout;
            rp_layout.pos = glm::vec2(screen.work_pos.x + screen.work_size.x - rpw, screen.work_pos.y);
            rp_layout.size = glm::vec2(rpw, ph);
            rp_layout.scene_h = scene_h + 8.0f;
            rp_layout.splitter_h = splitter_h;

            const bool float_blocks_rp = reg.isPositionOverFloatingPanel(
                panel_input.mouse_x, panel_input.mouse_y);
            if (float_blocks_rp) {
                PanelInputState masked_input = panel_input;
                masked_input.mouse_x = -1.0e9f;
                masked_input.mouse_y = -1.0e9f;
                for (auto& v : masked_input.mouse_clicked)
                    v = false;
                for (auto& v : masked_input.mouse_released)
                    v = false;
                for (auto& v : masked_input.mouse_down)
                    v = false;
                masked_input.mouse_wheel = 0;
                rml_right_panel_.processInput(rp_layout, masked_input);
            } else {
                rml_right_panel_.processInput(rp_layout, panel_input);
            }

            if (rml_right_panel_.wantsInput() && !float_blocks_rp)
                guiFocusState().want_capture_mouse = true;
            if (rml_right_panel_.wantsKeyboard())
                guiFocusState().want_capture_keyboard = true;

            const auto main_tabs = reg.get_panels_for_space(PanelSpace::MainPanelTab);
            panel_layout_.syncActiveTab(main_tabs, focus_panel_name_);
            std::vector<TabSnapshot> tab_snaps;
            tab_snaps.reserve(main_tabs.size());
            for (size_t i = 0; i < main_tabs.size(); ++i) {
                const auto& t = main_tabs[i];
                tab_snaps.push_back({
                    .id = t.id,
                    .label = t.label,
                    .dom_id = makeRmlTabDomId(t.id),
                });
            }

            rml_right_panel_.render(rp_layout, tab_snaps, panel_layout_.getActiveTab(),
                                    panel_input.screen_x, panel_input.screen_y,
                                    panel_input.screen_w, panel_input.screen_h);
        }

        panel_layout_.renderRightPanel(ctx, draw_ctx, show_main_panel_, ui_hidden_,
                                       window_states_, focus_panel_name_, panel_input, screen);
        panel_layout_.renderBottomDock(draw_ctx, show_main_panel_, ui_hidden_,
                                       panel_input, screen);

        applyFrameInputCapture(&rml_right_panel_);

        auto apply_cursor = [](CursorRequest req) {
            switch (req) {
            case CursorRequest::ResizeEW: ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW); break;
            case CursorRequest::ResizeNS: ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS); break;
            default: break;
            }
        };
        python::set_viewport_bounds(viewport_layout_.pos.x, viewport_layout_.pos.y,
                                    viewport_layout_.size.x, viewport_layout_.size.y);

        PanelInputState floating_input = panel_input;
        floating_input.bg_draw_list = ImGui::GetForegroundDrawList(ImGui::GetMainViewport());
        reg.draw_panels(PanelSpace::Floating, draw_ctx, &floating_input);

        applyFrameInputCapture(&rml_right_panel_);

        gizmo_manager_.updateToolState(ctx, ui_hidden_);
        gizmo_manager_.updateCropFlash();

        float primary_toolbar_x = 0.0f;
        float primary_toolbar_width = viewport_layout_.size.x;
        bool show_secondary_toolbar = false;
        float secondary_toolbar_x = 0.0f;
        float secondary_toolbar_width = 0.0f;
        if (auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr;
            rendering && rendering->isIndependentSplitViewActive()) {
            if (const auto primary_panel = rendering->resolveViewerPanel(
                    viewer_->getViewport(),
                    viewport_layout_.pos, viewport_layout_.size, std::nullopt, SplitViewPanelId::Left)) {
                primary_toolbar_x = primary_panel->x - viewport_layout_.pos.x;
                primary_toolbar_width = primary_panel->width;
            }
            if (const auto secondary_panel = rendering->resolveViewerPanel(
                    viewer_->getViewport(),
                    viewport_layout_.pos, viewport_layout_.size, std::nullopt, SplitViewPanelId::Right)) {
                show_secondary_toolbar = secondary_panel->valid();
                secondary_toolbar_x = secondary_panel->x - viewport_layout_.pos.x;
                secondary_toolbar_width = secondary_panel->width;
            }
        }

        rml_viewport_overlay_.setToolbarPanels(primary_toolbar_x,
                                               primary_toolbar_width,
                                               show_secondary_toolbar,
                                               secondary_toolbar_x,
                                               secondary_toolbar_width);
        rml_viewport_overlay_.setViewportBounds(
            viewport_layout_.pos, viewport_layout_.size,
            {panel_input.screen_x, panel_input.screen_y});
        RmlViewportOverlay::GTMetricsOverlayState gt_metrics_overlay;
        if (auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr) {
            const auto settings = rendering->getSettings();
            if (rendering->isGTComparisonActive() &&
                settings.camera_metrics_mode != RenderSettings::CameraMetricsMode::Off) {
                gt_metrics_overlay.visible = true;
                gt_metrics_overlay.psnr_text = "--";
                gt_metrics_overlay.show_ssim =
                    settings.camera_metrics_mode == RenderSettings::CameraMetricsMode::PSNRSSIM;
                gt_metrics_overlay.ssim_text = "--";

                const auto content_bounds = rendering->getContentBounds(glm::ivec2(
                    std::max(static_cast<int>(viewport_layout_.size.x), 0),
                    std::max(static_cast<int>(viewport_layout_.size.y), 0)));
                gt_metrics_overlay.x =
                    content_bounds.x + content_bounds.width * settings.split_position + 18.0f;
                gt_metrics_overlay.y = content_bounds.y + 18.0f;

                const int current_camera_id = rendering->getCurrentCameraId();
                if (const auto metrics = rendering->getLatestCameraMetrics();
                    metrics && metrics->camera_id == current_camera_id) {
                    gt_metrics_overlay.psnr_text = std::format("{:.2f}", metrics->psnr);
                    if (gt_metrics_overlay.show_ssim && metrics->ssim.has_value()) {
                        gt_metrics_overlay.ssim_text = std::format("{:.4f}", *metrics->ssim);
                    }
                }
            }
        }
        rml_viewport_overlay_.setGTMetricsOverlay(std::move(gt_metrics_overlay));
        startup_overlay_.setInput(&panel_input);
        if (startup_overlay_.isVisible()) {
            auto& focus = guiFocusState();
            focus.want_capture_mouse = true;
            focus.want_capture_keyboard = true;
        }
        rml_viewport_overlay_.processInput(panel_input);
        if (rml_viewport_overlay_.wantsInput() && panel_input.mouse_clicked[0]) {
            if (auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr;
                rendering && rendering->isIndependentSplitViewActive()) {
                if (const auto target_panel = rendering->resolveViewerPanel(
                        viewer_->getViewport(),
                        viewport_layout_.pos,
                        viewport_layout_.size,
                        glm::vec2(panel_input.mouse_x, panel_input.mouse_y))) {
                    if (auto* const input_controller = viewer_->getInputController()) {
                        input_controller->setFocusedSplitPanel(target_panel->panel);
                    } else {
                        rendering->setFocusedSplitPanel(target_panel->panel);
                    }
                }
            }
        }
        if (lfs::python::has_python_hooks("viewport_overlay", "draw")) {
            lfs::python::invoke_python_hooks("viewport_overlay", "draw", true);
            lfs::python::invoke_python_hooks("viewport_overlay", "draw", false);
        }
        if (vulkan_gui_)
            renderVulkan();
        reg.draw_panels(PanelSpace::ViewportOverlay, draw_ctx);

        rml_viewport_overlay_.render();

        applyFrameInputCapture();
        const std::string frame_tooltip = RmlPanelHost::consumeFrameTooltip();

        // Recompute viewport layout
        viewport_layout_ = panel_layout_.computeViewportLayout(
            show_main_panel_, ui_hidden_, window_states_["python_console"], screen);

        if (!ui_hidden_) {
            const float status_bar_h =
                PanelLayoutManager::STATUS_BAR_HEIGHT * lfs::python::get_shared_dpi_scale();
            rml_status_bar_.render(draw_ctx,
                                   screen.work_pos.x,
                                   screen.work_pos.y + screen.work_size.y - status_bar_h,
                                   screen.work_size.x,
                                   status_bar_h,
                                   panel_input.screen_w,
                                   panel_input.screen_h);
            reg.draw_panels(PanelSpace::StatusBar, draw_ctx, &panel_input);
        }

        python::draw_python_modals(scene);
        python::draw_python_popups(scene);

        rml_modal_overlay_->processInput(raw_panel_input);
        rml_viewport_overlay_.compositeToScreen(panel_input.screen_w, panel_input.screen_h);
        if (ImGui::GetMouseCursor() == ImGuiMouseCursor_Arrow)
            applyRmlCursorRequest(rmlui_manager_.consumeCursorRequest());
        apply_cursor(rml_right_panel_.getCursorRequest());
        apply_cursor(panel_layout_.getCursorRequest());
        if (SDL_Cursor* const cursor = systemCursorForImGuiCursor(ImGui::GetMouseCursor()))
            SDL_SetCursor(cursor);
        syncWindowTextInput(viewer_->getWindow());

        if (vulkan_gui_) {
            if (menu_bar_ && !ui_hidden_)
                rml_menu_bar_.draw(panel_input.screen_w, panel_input.screen_h);
            global_context_menu_->render(panel_input.screen_w, panel_input.screen_h,
                                         panel_input.screen_x, panel_input.screen_y);
            const auto* mvp_modal = ImGui::GetMainViewport();
            rml_modal_overlay_->render(static_cast<int>(mvp_modal->Size.x),
                                       static_cast<int>(mvp_modal->Size.y),
                                       mvp_modal->Pos.x, mvp_modal->Pos.y,
                                       viewport_layout_.pos.x, viewport_layout_.pos.y,
                                       viewport_layout_.size.x, viewport_layout_.size.y);
        }

        ImGui::Render();

        if (vulkan_gui_) {
#ifdef LFS_VULKAN_VIEWER_ENABLED
            guiFocusState().any_item_active |= ImGui::IsAnyItemActive();

            const auto& bg = lfs::vis::theme().menu_background();
            VkClearValue clear_value{};
            clear_value.color = VkClearColorValue{{bg.x, bg.y, bg.z, 1.0f}};

            VulkanContext::Frame frame{};
            if (vulkan_context && vulkan_context->beginFrame(clear_value, frame)) {
                if (rmlui_manager_.beginVulkanFrame(frame.command_buffer, frame.extent)) {
                    rmlui_manager_.renderQueuedVulkanContexts(false);
                    imgui_vulkan_backend_.renderDrawData(ImGui::GetDrawData(), frame.command_buffer);
                    rmlui_manager_.renderQueuedVulkanContexts(true);
                    rmlui_manager_.endVulkanFrame();
                } else {
                    rmlui_manager_.clearVulkanQueue();
                    imgui_vulkan_backend_.renderDrawData(ImGui::GetDrawData(), frame.command_buffer);
                }
                if (!vulkan_context->endFrame()) {
                    LOG_WARN("Vulkan GUI frame present failed: {}", vulkan_context->lastError());
                }
            } else if (vulkan_context) {
                rmlui_manager_.clearVulkanQueue();
                LOG_WARN("Vulkan GUI frame begin failed: {}", vulkan_context->lastError());
            }

            if (!ui_layout_changed && ui_layout_settle_frames_ > 0)
                --ui_layout_settle_frames_;

            persistImGuiSettingsIfNeeded();
            return;
#else
            return;
#endif
        }

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        guiFocusState().any_item_active |= ImGui::IsAnyItemActive();

        // Clean up GL state after ImGui rendering (ImGui can leave VAO/shader bindings corrupted)
        glBindVertexArray(0);
        glUseProgram(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        // Clear any errors ImGui might have generated
        while (glGetError() != GL_NO_ERROR) {}

        RmlPanelHost::flushQueuedForegroundComposites(panel_input.screen_w, panel_input.screen_h);
        sequencer_ui_.compositeOverlays(panel_input.screen_w, panel_input.screen_h);
        drawFrameTooltip(frame_tooltip, panel_input.screen_w, panel_input.screen_h);

        if (menu_bar_ && !ui_hidden_ && rml_menu_bar_.fbo().valid()) {
            const float menu_height = rml_menu_bar_.isOpen()
                                          ? static_cast<float>(panel_input.screen_h)
                                          : rml_menu_bar_.barHeight();
            rml_menu_bar_.fbo().blitToScreen(
                0.0f, 0.0f,
                static_cast<float>(panel_input.screen_w),
                menu_height,
                panel_input.screen_w, panel_input.screen_h);
        }

        global_context_menu_->render(panel_input.screen_w, panel_input.screen_h,
                                     panel_input.screen_x, panel_input.screen_y);

        {
            const auto* mvp_modal = ImGui::GetMainViewport();
            rml_modal_overlay_->render(static_cast<int>(mvp_modal->Size.x),
                                       static_cast<int>(mvp_modal->Size.y),
                                       mvp_modal->Pos.x, mvp_modal->Pos.y,
                                       viewport_layout_.pos.x, viewport_layout_.pos.y,
                                       viewport_layout_.size.x, viewport_layout_.size.y);
        }

        glBindVertexArray(0);
        glUseProgram(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        while (glGetError() != GL_NO_ERROR) {}

        // Update and Render additional Platform Windows (for multi-viewport)
        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            SDL_Window* backup_window = SDL_GL_GetCurrentWindow();
            SDL_GLContext backup_context = SDL_GL_GetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            SDL_GL_MakeCurrent(backup_window, backup_context);

            // Clean up GL state after multi-viewport rendering too
            glBindVertexArray(0);
            glUseProgram(0);
            glBindTexture(GL_TEXTURE_2D, 0);
            while (glGetError() != GL_NO_ERROR) {}
        }

        if (!ui_layout_changed && ui_layout_settle_frames_ > 0)
            --ui_layout_settle_frames_;

        persistImGuiSettingsIfNeeded();
    }

    void GuiManager::renderSelectionOverlays(const UIContext& ctx) {
        if (auto* const tool = ctx.viewer->getBrushTool(); tool && tool->isEnabled() && !ui_hidden_) {
            tool->renderUI(ctx, nullptr);
        }
        if (auto* const tool = ctx.viewer->getSelectionTool(); tool && tool->isEnabled() && !ui_hidden_) {
            tool->renderUI(ctx, nullptr);
        }

        const bool mouse_over_ui = guiFocusState().want_capture_mouse;
        if (!ui_hidden_ && !mouse_over_ui && viewport_layout_.size.x > 0 && viewport_layout_.size.y > 0) {
            auto* rm = ctx.viewer->getRenderingManager();
            auto* draw_list = ImGui::GetForegroundDrawList();
            const glm::ivec2 rendered_size = rm ? rm->getRenderedSize() : glm::ivec2(0);
            struct PreviewPanelContext {
                float x = 0.0f;
                float y = 0.0f;
                float width = 0.0f;
                float height = 0.0f;
                int render_width = 0;
                int render_height = 0;
                const Viewport* viewport = nullptr;
            };
            const auto resolve_preview_panel = [&](const std::optional<SplitViewPanelId> panel) {
                PreviewPanelContext panel_ctx{
                    .x = viewport_layout_.pos.x,
                    .y = viewport_layout_.pos.y,
                    .width = viewport_layout_.size.x,
                    .height = viewport_layout_.size.y,
                    .render_width =
                        rendered_size.x > 0 ? rendered_size.x : static_cast<int>(ctx.viewer->getViewport().windowSize.x),
                    .render_height =
                        rendered_size.y > 0 ? rendered_size.y : static_cast<int>(ctx.viewer->getViewport().windowSize.y),
                    .viewport = &ctx.viewer->getViewport(),
                };
                if (!rm || !panel || !rm->isIndependentSplitViewActive()) {
                    return panel_ctx;
                }

                const auto info = rm->resolveViewerPanel(
                    ctx.viewer->getViewport(),
                    {viewport_layout_.pos.x, viewport_layout_.pos.y},
                    {viewport_layout_.size.x, viewport_layout_.size.y},
                    std::nullopt,
                    panel);
                if (!info) {
                    return panel_ctx;
                }

                panel_ctx.x = info->x;
                panel_ctx.y = info->y;
                panel_ctx.width = info->width;
                panel_ctx.height = info->height;
                panel_ctx.render_width = info->render_width;
                panel_ctx.render_height = info->render_height;
                panel_ctx.viewport = info->viewport;
                return panel_ctx;
            };
            const auto render_to_screen = [&](const PreviewPanelContext& panel_ctx, const float x, const float y) {
                const float render_to_screen_x =
                    (panel_ctx.render_width > 0)
                        ? (panel_ctx.width / static_cast<float>(panel_ctx.render_width))
                        : (1.0f / std::max(rm ? rm->getSettings().render_scale : 1.0f, 0.001f));
                const float render_to_screen_y =
                    (panel_ctx.render_height > 0)
                        ? (panel_ctx.height / static_cast<float>(panel_ctx.render_height))
                        : (1.0f / std::max(rm ? rm->getSettings().render_scale : 1.0f, 0.001f));
                return ImVec2(panel_ctx.x + x * render_to_screen_x,
                              panel_ctx.y + y * render_to_screen_y);
            };
            // Keep preview overlays inside the live viewport region so docked panels stay in front.
            const auto push_preview_clip = [&](const PreviewPanelContext& panel_ctx) {
                const ImVec2 clip_min(panel_ctx.x, panel_ctx.y);
                float clip_bottom = panel_ctx.y + panel_ctx.height;
                const float bottom_dock_top = panel_layout_.bottomDockTopY();
                if (bottom_dock_top > 0.0f) {
                    clip_bottom = std::min(clip_bottom, bottom_dock_top);
                }

                const ImVec2 clip_max(panel_ctx.x + panel_ctx.width, clip_bottom);
                if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y) {
                    return false;
                }

                draw_list->PushClipRect(clip_min, clip_max, true);
                return true;
            };

            if (rm && rm->isCursorPreviewActive()) {
                const auto& t = theme();
                float bx, by, br;
                bool add_mode;
                rm->getCursorPreviewState(bx, by, br, add_mode);
                const auto panel_ctx = resolve_preview_panel(rm->getCursorPreviewPanel());

                const ImVec2 screen_pos = render_to_screen(panel_ctx, bx, by);
                const float screen_radius =
                    (panel_ctx.render_width > 0)
                        ? br * (panel_ctx.width / static_cast<float>(panel_ctx.render_width))
                        : br;

                const ImU32 brush_color = add_mode
                                              ? toU32WithAlpha(t.palette.success, 0.8f)
                                              : toU32WithAlpha(t.palette.error, 0.8f);
                if (push_preview_clip(panel_ctx)) {
                    draw_list->AddCircle(screen_pos, screen_radius, brush_color, 32, 2.0f);
                    draw_list->AddCircleFilled(screen_pos, 3.0f, brush_color);
                    draw_list->PopClipRect();
                }
            }

            if (rm && rm->isRectPreviewActive()) {
                const auto& t = theme();
                float rx0, ry0, rx1, ry1;
                bool add_mode;
                rm->getRectPreview(rx0, ry0, rx1, ry1, add_mode);
                const auto panel_ctx = resolve_preview_panel(rm->getRectPreviewPanel());

                const ImVec2 p0 = render_to_screen(panel_ctx, rx0, ry0);
                const ImVec2 p1 = render_to_screen(panel_ctx, rx1, ry1);

                const ImU32 fill_color = add_mode
                                             ? toU32WithAlpha(t.palette.success, 0.15f)
                                             : toU32WithAlpha(t.palette.error, 0.15f);
                const ImU32 border_color = add_mode
                                               ? toU32WithAlpha(t.palette.success, 0.8f)
                                               : toU32WithAlpha(t.palette.error, 0.8f);

                if (push_preview_clip(panel_ctx)) {
                    draw_list->AddRectFilled(p0, p1, fill_color);
                    draw_list->AddRect(p0, p1, border_color, 0.0f, 0, 2.0f);
                    draw_list->PopClipRect();
                }
            }

            if (rm && rm->isPolygonPreviewActive()) {
                const auto& t = theme();
                const auto& points = rm->getPolygonPoints();
                const auto& world_points = rm->getPolygonWorldPoints();
                const bool closed = rm->isPolygonClosed();
                const bool add_mode = rm->isPolygonAddMode();
                const auto panel_ctx = resolve_preview_panel(rm->getPolygonPreviewPanel());

                if (!points.empty() || !world_points.empty()) {
                    const ImU32 line_color = add_mode
                                                 ? toU32WithAlpha(t.palette.success, 0.8f)
                                                 : toU32WithAlpha(t.palette.error, 0.8f);
                    const ImU32 fill_color = add_mode
                                                 ? toU32WithAlpha(t.palette.success, 0.15f)
                                                 : toU32WithAlpha(t.palette.error, 0.15f);
                    const ImU32 vertex_color = t.polygon_vertex_u32();
                    const ImU32 vertex_hover_color = t.polygon_vertex_hover_u32();
                    const ImU32 close_hint_color = t.polygon_close_hint_u32();
                    const ImU32 line_to_mouse_color = add_mode
                                                          ? toU32WithAlpha(t.palette.success, 0.5f)
                                                          : toU32WithAlpha(t.palette.error, 0.5f);

                    std::vector<ImVec2> screen_points;
                    if (rm->isPolygonPreviewWorldSpace()) {
                        const auto render_settings = rm->getSettings();
                        screen_points.reserve(world_points.size());

                        if (!panel_ctx.viewport) {
                            screen_points.clear();
                        }
                        Viewport projection_viewport = panel_ctx.viewport ? *panel_ctx.viewport : ctx.viewer->getViewport();
                        projection_viewport.windowSize = {std::max(panel_ctx.render_width, 1),
                                                          std::max(panel_ctx.render_height, 1)};

                        bool all_visible = true;
                        for (const auto& world_point : world_points) {
                            const auto projected = lfs::rendering::projectWorldPoint(
                                projection_viewport.camera.R,
                                projection_viewport.camera.t,
                                projection_viewport.windowSize,
                                world_point,
                                render_settings.focal_length_mm,
                                render_settings.orthographic,
                                render_settings.ortho_scale);
                            if (!projected) {
                                all_visible = false;
                                break;
                            }

                            screen_points.emplace_back(
                                panel_ctx.x + projected->x * (panel_ctx.width / static_cast<float>(projection_viewport.windowSize.x)),
                                panel_ctx.y + projected->y * (panel_ctx.height / static_cast<float>(projection_viewport.windowSize.y)));
                        }

                        if (!all_visible) {
                            screen_points.clear();
                        }
                    } else {
                        screen_points.reserve(points.size());
                        for (const auto& [px, py] : points) {
                            screen_points.push_back(render_to_screen(panel_ctx, px, py));
                        }
                    }

                    if (push_preview_clip(panel_ctx)) {
                        if (closed && screen_points.size() >= 3) {
                            draw_list->AddConvexPolyFilled(screen_points.data(), static_cast<int>(screen_points.size()), fill_color);
                        }

                        for (size_t i = 0; i + 1 < screen_points.size(); ++i) {
                            draw_list->AddLine(screen_points[i], screen_points[i + 1], line_color, 2.0f);
                        }
                        if (closed && screen_points.size() >= 3) {
                            draw_list->AddLine(screen_points.back(), screen_points.front(), line_color, 2.0f);
                        }

                        const ImVec2 mouse_pos =
                            s_frame_input
                                ? ImVec2(s_frame_input->mouse_x, s_frame_input->mouse_y)
                                : ImVec2(viewport_layout_.pos.x, viewport_layout_.pos.y);
                        constexpr float CLOSE_THRESHOLD = 12.0f;
                        constexpr float VERTEX_RADIUS = 5.0f;
                        const auto distance_sq = [](const ImVec2 a, const ImVec2 b) {
                            const float dx = a.x - b.x;
                            const float dy = a.y - b.y;
                            return dx * dx + dy * dy;
                        };
                        const bool can_close = !closed && screen_points.size() >= 3 &&
                                               distance_sq(mouse_pos, screen_points.front()) <
                                                   CLOSE_THRESHOLD * CLOSE_THRESHOLD;
                        int hovered_idx = -1;
                        for (size_t i = 0; i < screen_points.size(); ++i) {
                            if (distance_sq(mouse_pos, screen_points[i]) <= VERTEX_RADIUS * VERTEX_RADIUS) {
                                hovered_idx = static_cast<int>(i);
                                break;
                            }
                        }

                        if (!closed && !screen_points.empty()) {
                            draw_list->AddLine(screen_points.back(), mouse_pos, line_to_mouse_color, 1.0f);

                            if (can_close) {
                                draw_list->AddCircle(screen_points.front(), 9.0f, close_hint_color, 16, 2.0f);
                            }
                        }

                        for (size_t i = 0; i < screen_points.size(); ++i) {
                            const ImU32 color = (static_cast<int>(i) == hovered_idx || (can_close && i == 0))
                                                    ? vertex_hover_color
                                                    : vertex_color;
                            draw_list->AddCircleFilled(screen_points[i], VERTEX_RADIUS, color);
                            draw_list->AddCircle(screen_points[i], VERTEX_RADIUS, line_color, 16, 1.5f);
                        }

                        if (!screen_points.empty()) {
                            const float initial_ring_radius = can_close ? 9.0f : 8.0f;
                            const float initial_ring_thickness = can_close ? 2.0f : 1.5f;
                            draw_list->AddCircle(screen_points.front(), initial_ring_radius,
                                                 close_hint_color, 24, initial_ring_thickness);
                        }

                        if (closed && screen_points.size() >= 3) {
                            float cx = 0.0f, cy = 0.0f;
                            for (const auto& sp : screen_points) {
                                cx += sp.x;
                                cy += sp.y;
                            }
                            cx /= static_cast<float>(screen_points.size());
                            cy /= static_cast<float>(screen_points.size());

                            const char* hint = "Enter to confirm\nShift-click edge: add\nCtrl-click vertex: remove";
                            const ImVec2 text_size = ImGui::CalcTextSize(hint);
                            draw_list->AddText(
                                ImVec2(cx - text_size.x * 0.5f, cy - text_size.y * 0.5f),
                                toU32WithAlpha(t.palette.text, 0.9f), hint);
                        }

                        draw_list->PopClipRect();
                    }
                }
            }

            if (rm && rm->isLassoPreviewActive()) {
                const auto& t = theme();
                const auto& points = rm->getLassoPoints();
                const bool add_mode = rm->isLassoAddMode();
                const auto panel_ctx = resolve_preview_panel(rm->getLassoPreviewPanel());

                if (points.size() >= 2) {
                    const ImU32 line_color = add_mode
                                                 ? toU32WithAlpha(t.palette.success, 0.8f)
                                                 : toU32WithAlpha(t.palette.error, 0.8f);

                    if (push_preview_clip(panel_ctx)) {
                        ImVec2 prev = render_to_screen(panel_ctx, points[0].first, points[0].second);
                        for (size_t i = 1; i < points.size(); ++i) {
                            ImVec2 curr = render_to_screen(panel_ctx, points[i].first, points[i].second);
                            draw_list->AddLine(prev, curr, line_color, 2.0f);
                            prev = curr;
                        }
                        draw_list->PopClipRect();
                    }
                }
            }
        }

        auto* align_tool = ctx.viewer->getAlignTool();
        if (align_tool && align_tool->isEnabled() && !ui_hidden_) {
            align_tool->renderUI(ctx, nullptr);
        }

        if (auto* const ic = ctx.viewer->getInputController();
            !ui_hidden_ && ic && ic->isNodeRectDragging()) {
            const auto start = ic->getNodeRectStart();
            const auto end = ic->getNodeRectEnd();
            const auto& t = theme();
            auto* const draw_list = ImGui::GetForegroundDrawList();
            draw_list->AddRectFilled({start.x, start.y}, {end.x, end.y}, toU32WithAlpha(t.palette.warning, 0.15f));
            draw_list->AddRect({start.x, start.y}, {end.x, end.y}, toU32WithAlpha(t.palette.warning, 0.85f), 0.0f, 0, 2.0f);
        }
    }

    void GuiManager::renderViewportDecorations() {
        if (!ui_hidden_ && viewport_layout_.size.x > 0 && viewport_layout_.size.y > 0) {
            const auto& t = theme();
            const float r = t.viewport.corner_radius;
            if (r > 0.0f) {
                auto* const dl = ImGui::GetBackgroundDrawList();
                const ImU32 bg = toU32(t.menu_background());
                const float x1 = viewport_layout_.pos.x, y1 = viewport_layout_.pos.y;
                const float x2 = x1 + viewport_layout_.size.x, y2 = y1 + viewport_layout_.size.y;

                constexpr int CORNER_ARC_SEGMENTS = 12;
                const auto maskCorner = [&](const ImVec2 corner, const ImVec2 edge,
                                            const ImVec2 center, const float a0, const float a1) {
                    dl->PathLineTo(corner);
                    dl->PathLineTo(edge);
                    dl->PathArcTo(center, r, a0, a1, CORNER_ARC_SEGMENTS);
                    dl->PathLineTo(corner);
                    dl->PathFillConvex(bg);
                };
                maskCorner({x1, y1}, {x1, y1 + r}, {x1 + r, y1 + r}, IM_PI, IM_PI * 1.5f);
                maskCorner({x2, y1}, {x2 - r, y1}, {x2 - r, y1 + r}, IM_PI * 1.5f, IM_PI * 2.0f);
                maskCorner({x1, y2}, {x1 + r, y2}, {x1 + r, y2 - r}, IM_PI * 0.5f, IM_PI);
                maskCorner({x2, y2}, {x2, y2 - r}, {x2 - r, y2 - r}, 0.0f, IM_PI * 0.5f);

                if (show_main_panel_) {
                    const float rpw = panel_layout_.getRightPanelWidth();
                    auto* mvp = ImGui::GetMainViewport();
                    const float px = mvp->WorkPos.x + mvp->WorkSize.x - rpw;
                    const float py1 = mvp->WorkPos.y;
                    const float py2 = py1 + mvp->WorkSize.y - PanelLayoutManager::STATUS_BAR_HEIGHT * current_ui_scale_;
                    maskCorner({px, py1}, {px, py1 + r}, {px + r, py1 + r}, IM_PI, IM_PI * 1.5f);
                    maskCorner({px, py2}, {px + r, py2}, {px + r, py2 - r}, IM_PI * 0.5f, IM_PI);
                }

                if (t.viewport.border_size > 0.0f) {
                    dl->AddRect({x1, y1}, {x2, y2}, t.viewport_border_u32(), r,
                                ImDrawFlags_RoundCornersAll, t.viewport.border_size);
                }
            }
        }

        auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr;
        if (!rendering || viewport_layout_.size.x <= 0.0f || viewport_layout_.size.y <= 0.0f) {
            return;
        }

        if (!rendering->isSplitViewActive()) {
            return;
        }

        const auto& t = theme();
        auto* const draw_list = ImGui::GetBackgroundDrawList(ImGui::GetMainViewport());
        const auto divider_x = rendering->getSplitDividerScreenX(viewport_layout_.pos, viewport_layout_.size);
        if (!divider_x) {
            return;
        }
        constexpr float kSplitDividerMinWidthPx = 10.0f;
        const float divider_width =
            std::max(kSplitDividerMinWidthPx * current_ui_scale_,
                     std::round(t.viewport.border_size * current_ui_scale_ * 4.0f));
        const float divider_left = std::round(*divider_x - divider_width * 0.5f);
        const float divider_right = std::round(*divider_x + divider_width * 0.5f);
        const ImU32 divider_fill_color = toU32(t.menu_background());

        draw_list->PushClipRect(
            ImVec2(viewport_layout_.pos.x, viewport_layout_.pos.y),
            ImVec2(viewport_layout_.pos.x + viewport_layout_.size.x,
                   viewport_layout_.pos.y + viewport_layout_.size.y),
            true);
        draw_list->AddRectFilled(
            ImVec2(divider_left, viewport_layout_.pos.y),
            ImVec2(divider_right, viewport_layout_.pos.y + viewport_layout_.size.y),
            divider_fill_color);
        draw_list->PopClipRect();
    }

    void GuiManager::updateInputOverrides(const PanelInputState& input,
                                          bool mouse_in_viewport) {
        if (rml_menu_bar_.wantsInput())
            return;

        auto& focus = guiFocusState();
        const bool any_popup_or_modal_open =
            ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) ||
            isModalWindowOpen() ||
            (global_context_menu_ && global_context_menu_->isOpen());
        const bool imgui_wants_input = focus.want_text_input || focus.want_capture_keyboard;

        if (isTransformGizmoOverOrUsing() && !any_popup_or_modal_open) {
            focus.want_capture_mouse = false;
            focus.want_capture_keyboard = false;
        }

        if (mouse_in_viewport && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) &&
            !any_popup_or_modal_open && !imgui_wants_input) {
            if (input.mouse_down[1] || input.mouse_down[2]) {
                focus.want_capture_mouse = false;
            }
            if (input.mouse_clicked[0] || input.mouse_clicked[1]) {
                ImGui::ClearActiveID();
                focus.want_capture_keyboard = false;
                auto* console_state = panels::PythonConsoleState::tryGetInstance();
                if (console_state != nullptr) {
                    auto* editor = console_state->getEditor();
                    if (editor != nullptr) {
                        editor->unfocus();
                    }
                }
            }
        }

        auto* rendering_manager = viewer_->getRenderingManager();
        if (rendering_manager) {
            const auto& settings = rendering_manager->getSettings();
            if (settings.point_cloud_mode && mouse_in_viewport &&
                !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) &&
                !any_popup_or_modal_open && !imgui_wants_input) {
                focus.want_capture_mouse = false;
                focus.want_capture_keyboard = false;
            }
        }
    }

    glm::vec2 GuiManager::getViewportPos() const {
        return viewport_layout_.pos;
    }

    glm::vec2 GuiManager::getViewportSize() const {
        return viewport_layout_.size;
    }

    bool GuiManager::isViewportFocused() const {
        return viewport_layout_.has_focus;
    }

    bool GuiManager::isPositionInViewport(double x, double y) const {
        return (x >= viewport_layout_.pos.x &&
                x < viewport_layout_.pos.x + viewport_layout_.size.x &&
                y >= viewport_layout_.pos.y &&
                y < viewport_layout_.pos.y + viewport_layout_.size.y);
    }

    bool GuiManager::isPositionOverFloatingPanel(const double x, const double y) const {
        return PanelRegistry::instance().isPositionOverFloatingPanel(x, y);
    }

    GuiHitTestResult GuiManager::hitTestPointer(const double x, const double y) const {
        if (isCapturingInput() || isModalWindowOpen() || startup_overlay_.isVisible() ||
            (global_context_menu_ && global_context_menu_->isOpen())) {
            return {.blocks_pointer = true, .takes_keyboard_focus = true};
        }

        if (panel_layout_.isResizingPanel() || isPositionOverFloatingPanel(x, y)) {
            return {.blocks_pointer = true, .takes_keyboard_focus = true};
        }

        if (sequencer_ui_.blocksPointer(x, y) || rml_viewport_overlay_.blocksPointer(x, y)) {
            return {.blocks_pointer = true, .takes_keyboard_focus = true};
        }

        if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
            return {.blocks_pointer = true, .takes_keyboard_focus = true};
        }

        return {};
    }

    GuiInputState GuiManager::inputState() const {
        const auto& focus = guiFocusState();
        const bool modal_open =
            isCapturingInput() ||
            isModalWindowOpen() ||
            startup_overlay_.isVisible() ||
            (global_context_menu_ && global_context_menu_->isOpen()) ||
            sequencer_ui_.blocksKeyboard();

        return {
            .has_keyboard_focus = focus.any_item_active || focus.want_capture_keyboard,
            .text_input_active = focus.want_text_input,
            .modal_open = modal_open,
        };
    }

    void GuiManager::setupEventHandlers() {
        using namespace lfs::core::events;

        ui::FileDropReceived::when([this](const auto&) {
            startup_overlay_.dismiss();
            drag_drop_.resetHovering();
        });

        cmd::ShowWindow::when([this](const auto& e) {
            showWindow(e.window_name, e.show);
        });

        cmd::GoToCamView::when([this](const auto& e) {
            if (auto* sm = viewer_->getSceneManager()) {
                const auto& scene = sm->getScene();
                for (const auto* node : scene.getNodes()) {
                    if (node->type == core::NodeType::CAMERA && node->camera_uid == e.cam_id) {
                        ui::NodeSelected{.path = node->name, .type = "Camera", .metadata = {}}.emit();
                        break;
                    }
                }
            }
        });

        ui::FocusTrainingPanel::when([this](const auto&) {
            focus_panel_name_ = "Training";
        });

        ui::ToggleUI::when([this](const auto&) {
            ui_hidden_ = !ui_hidden_;
        });

        ui::ToggleFullscreen::when([this](const auto&) {
            if (auto* wm = viewer_->getWindowManager()) {
                wm->toggleFullscreen();
            }
        });

        internal::DisplayScaleChanged::when([this](const auto& e) {
            if (lfs::vis::loadUiScalePreference() <= 0.0f) {
                pending_ui_scale_ = std::clamp(e.scale, 1.0f, 4.0f);
            }
        });

        internal::UiScaleChangeRequested::when([this](const auto& e) {
            if (e.scale <= 0.0f) {
                pending_ui_scale_ = std::clamp(SDL_GetWindowDisplayScale(viewer_->getWindow()), 1.0f, 4.0f);
            } else {
                pending_ui_scale_ = std::clamp(e.scale, 1.0f, 4.0f);
            }
        });

        state::DiskSpaceSaveFailed::when([this](const auto& e) {
            using namespace lichtfeld::Strings;
            if (!e.is_disk_space_error)
                return;

            auto formatBytes = [](size_t bytes) -> std::string {
                constexpr double KB = 1024.0;
                constexpr double MB = KB * 1024.0;
                constexpr double GB = MB * 1024.0;
                if (bytes >= static_cast<size_t>(GB))
                    return std::format("{:.2f} GB", static_cast<double>(bytes) / GB);
                if (bytes >= static_cast<size_t>(MB))
                    return std::format("{:.2f} MB", static_cast<double>(bytes) / MB);
                if (bytes >= static_cast<size_t>(KB))
                    return std::format("{:.2f} KB", static_cast<double>(bytes) / KB);
                return std::format("{} bytes", bytes);
            };

            const std::string subtitle = e.is_checkpoint
                                             ? std::format("{} {})", LOC(DiskSpaceDialog::CHECKPOINT_SAVE_FAILED), e.iteration)
                                             : std::string(LOC(DiskSpaceDialog::EXPORT_FAILED));

            std::string body;
            body += std::format("<div>{}</div>", LOC(DiskSpaceDialog::INSUFFICIENT_SPACE_PREFIX));
            body += std::format("<div class=\"content-row\"><span class=\"dim-text\">{} </span>{}</div>",
                                LOC(DiskSpaceDialog::LOCATION_LABEL), lfs::core::path_to_utf8(e.path.parent_path()));
            body += std::format("<div class=\"content-row\"><span class=\"dim-text\">{} </span>{}</div>",
                                LOC(DiskSpaceDialog::REQUIRED_LABEL), formatBytes(e.required_bytes));
            if (e.available_bytes > 0) {
                body += std::format("<div class=\"content-row\"><span class=\"dim-text\">{} </span>"
                                    "<span class=\"error-text\">{}</span></div>",
                                    LOC(DiskSpaceDialog::AVAILABLE_LABEL), formatBytes(e.available_bytes));
            }
            body += std::format("<div class=\"warning-text\">{}</div>", LOC(DiskSpaceDialog::INSTRUCTION));

            lfs::core::ModalRequest req;
            req.title = std::format("{} | {}", LOC(DiskSpaceDialog::ERROR_LABEL), subtitle);
            req.body_rml = body;
            req.style = lfs::core::ModalStyle::Error;
            req.width_dp = 480;
            req.buttons = {
                {LOC(DiskSpaceDialog::CANCEL), "secondary"},
                {LOC(DiskSpaceDialog::CHANGE_LOCATION), "warning"},
                {LOC(DiskSpaceDialog::RETRY), "primary"}};

            auto path = e.path;
            auto iteration = e.iteration;
            auto is_checkpoint = e.is_checkpoint;

            req.on_result = [this, path, iteration, is_checkpoint](const lfs::core::ModalResult& result) {
                if (result.button_label == LOC(DiskSpaceDialog::RETRY)) {
                    if (is_checkpoint) {
                        if (auto* tm = viewer_->getTrainerManager()) {
                            if (tm->isFinished() || !tm->isTrainingActive()) {
                                if (auto* trainer = tm->getTrainer()) {
                                    LOG_INFO("Retrying save at iteration {}", iteration);
                                    trainer->save_final_ply_and_checkpoint(iteration);
                                }
                            } else {
                                tm->requestSaveCheckpoint();
                            }
                        }
                    }
                } else if (result.button_label == LOC(DiskSpaceDialog::CHANGE_LOCATION)) {
                    std::filesystem::path new_location = PickFolderDialog(path.parent_path());
                    if (!new_location.empty() && is_checkpoint) {
                        if (auto* tm = viewer_->getTrainerManager()) {
                            if (auto* trainer = tm->getTrainer()) {
                                auto params = trainer->getParams();
                                params.dataset.output_path = new_location;
                                trainer->setParams(params);
                                LOG_INFO("Output path changed to: {}", lfs::core::path_to_utf8(new_location));
                                if (tm->isFinished() || !tm->isTrainingActive())
                                    trainer->save_final_ply_and_checkpoint(iteration);
                                else
                                    tm->requestSaveCheckpoint();
                            }
                        }
                    } else if (!new_location.empty()) {
                        LOG_INFO("Re-export manually using File > Export to: {}",
                                 lfs::core::path_to_utf8(new_location));
                    }
                } else {
                    if (is_checkpoint)
                        LOG_WARN("Checkpoint save cancelled by user");
                    else
                        LOG_INFO("Export cancelled by user");
                }
            };
            req.on_cancel = [is_checkpoint]() {
                if (is_checkpoint)
                    LOG_WARN("Checkpoint save cancelled by user");
                else
                    LOG_INFO("Export cancelled by user");
            };

            rml_modal_overlay_->enqueue(std::move(req));
        });

        state::DatasetLoadCompleted::when([this](const auto& e) {
            if (e.success) {
                focus_panel_name_ = "Training";
            }
        });

        internal::TrainerReady::when([this](const auto&) {
            focus_panel_name_ = "Training";
        });
    }

    bool GuiManager::isCapturingInput() const {
        if (auto* input_controller = viewer_->getInputController()) {
            return input_controller->getBindings().isCapturing();
        }
        return false;
    }

    bool GuiManager::isModalWindowOpen() const {
        return ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) ||
               rml_modal_overlay_->isOpen();
    }

    void GuiManager::captureKey(int physical_key, int logical_key, int mods) {
        if (auto* input_controller = viewer_->getInputController()) {
            input_controller->getBindings().captureKey(physical_key, logical_key, mods);
        }
    }

    void GuiManager::captureMouseButton(int button, int mods) {
        if (auto* input_controller = viewer_->getInputController()) {
            input_controller->getBindings().captureMouseButton(button, mods);
        }
    }

    void GuiManager::requestThumbnail(const std::string& video_id) {
        if (menu_bar_) {
            menu_bar_->requestThumbnail(video_id);
        }
    }

    void GuiManager::processThumbnails() {
        if (menu_bar_) {
            menu_bar_->processThumbnails();
        }
    }

    bool GuiManager::isThumbnailReady(const std::string& video_id) const {
        return menu_bar_ ? menu_bar_->isThumbnailReady(video_id) : false;
    }

    uint64_t GuiManager::getThumbnailTexture(const std::string& video_id) const {
        return menu_bar_ ? menu_bar_->getThumbnailTexture(video_id) : 0;
    }

    int GuiManager::getHighlightedCameraUid() const {
        if (auto* sm = viewer_->getSceneManager()) {
            return sm->getSelectedCameraUid();
        }
        return -1;
    }

    void GuiManager::applyDefaultStyle() {
        const std::string preferred_theme = loadThemePreferenceName();
        if (!setThemeByName(preferred_theme)) {
            setTheme(darkTheme());
        }
        rmlui_manager_.activateTheme(currentThemeId());
    }

    void GuiManager::showWindow(const std::string& name, bool show) {
        window_states_[name] = show;
    }

    bool GuiManager::needsAnimationFrame() const {
        if (startup_overlay_.needsAnimationFrame())
            return true;
        if (video_widget_ && video_widget_->isVideoPlaying())
            return true;
        if (ui_layout_settle_frames_ > 0)
            return true;
        if (rml_right_panel_.needsAnimationFrame())
            return true;
        if (PanelRegistry::instance().needsAnimationFrame())
            return true;
        return false;
    }

    void GuiManager::dismissStartupOverlay() {
        startup_overlay_.dismiss();
    }

    void GuiManager::requestExitConfirmation() {
        startup_overlay_.dismiss();
        lfs::core::events::cmd::RequestExit{}.emit();
    }

    bool GuiManager::isExitConfirmationPending() const {
        return lfs::python::is_exit_popup_open();
    }

} // namespace lfs::vis::gui
