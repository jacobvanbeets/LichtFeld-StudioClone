/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/rendering.hpp"
#include "core/camera.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "gs_rasterizer_tensor.hpp"
#include "image_layout.hpp"
#include "rendering/coordinate_conventions.hpp"
#include <OpenImageIO/imageio.h>
#include <algorithm>
#include <array>
#include <cuda_runtime.h>
#include <cmath>
#include <filesystem>
#include <format>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <limits>
#include <mutex>
#include <string_view>
#include <vector>

namespace lfs::rendering {

    namespace {
        struct RasterImageResult {
            Tensor image;
            Tensor depth;
            bool valid = false;
            bool flip_y = false;
            float far_plane = DEFAULT_FAR_PLANE;
            bool orthographic = false;
            bool color_has_alpha = false;
        };

        struct GaussianRasterResources {
            Tensor crop_box_transform_tensor;
            Tensor crop_box_min_tensor;
            Tensor crop_box_max_tensor;
            Tensor ellipsoid_transform_tensor;
            Tensor ellipsoid_radii_tensor;
            Tensor view_volume_transform_tensor;
            Tensor view_volume_min_tensor;
            Tensor view_volume_max_tensor;
        };

        struct GaussianRasterRequest {
            FrameView frame_view;
            float scaling_modifier = 1.0f;
            bool antialiasing = false;
            bool mip_filter = false;
            int sh_degree = 3;
            bool gut = false;
            bool equirectangular = false;
            GaussianSceneState scene;
            GaussianFilterState filters;
            GaussianOverlayState overlay;
            bool transparent_background = false;
            unsigned long long* hovered_depth_id = nullptr;
            Tensor* screen_positions_out = nullptr;
        };

        struct EnvironmentImage {
            std::filesystem::path path;
            int width = 0;
            int height = 0;
            std::vector<float> pixels;

            [[nodiscard]] bool valid() const {
                return width > 0 && height > 0 &&
                       pixels.size() == static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
            }
        };

        struct EnvironmentImageCache {
            std::mutex mutex;
            EnvironmentImage image;
            std::string last_error;
        };

        [[nodiscard]] EnvironmentImageCache& environmentImageCache() {
            static EnvironmentImageCache cache;
            return cache;
        }

        [[nodiscard]] std::filesystem::path resolveEnvironmentPath(const std::filesystem::path& requested) {
            if (requested.empty() || requested.is_absolute()) {
                return requested;
            }
            if (std::filesystem::exists(requested)) {
                return requested;
            }

            const std::array candidates{
                lfs::core::getAssetsDir() / requested,
                lfs::core::getExecutableDir() / requested,
                lfs::core::getExecutableDir() / "assets" / requested,
            };
            for (const auto& candidate : candidates) {
                if (std::filesystem::exists(candidate)) {
                    return candidate;
                }
            }
            return lfs::core::getAssetsDir() / requested;
        }

        Result<EnvironmentImage> loadEnvironmentImage(const std::filesystem::path& environment_path) {
            const auto resolved_path = resolveEnvironmentPath(environment_path);
            auto& cache = environmentImageCache();
            std::lock_guard lock(cache.mutex);
            if (cache.image.valid() && cache.image.path == resolved_path) {
                return cache.image;
            }

            cache.image = {};
            cache.last_error.clear();
            if (resolved_path.empty()) {
                cache.last_error = "Environment map path is empty";
                return std::unexpected(cache.last_error);
            }
            if (!std::filesystem::exists(resolved_path)) {
                cache.last_error = std::format("Environment map not found: {}", resolved_path.string());
                return std::unexpected(cache.last_error);
            }

            const std::string path_utf8 = lfs::core::path_to_utf8(resolved_path);
            std::unique_ptr<OIIO::ImageInput> input(OIIO::ImageInput::open(path_utf8));
            if (!input) {
                cache.last_error = std::format("Failed to open environment map {}: {}", path_utf8, OIIO::geterror());
                return std::unexpected(cache.last_error);
            }

            const auto& spec = input->spec();
            if (spec.width <= 0 || spec.height <= 0 || spec.nchannels <= 0) {
                input->close();
                cache.last_error = std::format("Invalid environment map dimensions for {}", path_utf8);
                return std::unexpected(cache.last_error);
            }

            std::vector<float> source_pixels(
                static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) *
                static_cast<size_t>(spec.nchannels));
            if (!input->read_image(0, 0, 0, spec.nchannels, OIIO::TypeDesc::FLOAT, source_pixels.data())) {
                cache.last_error = std::format("Failed to read environment map {}: {}", path_utf8, input->geterror());
                input->close();
                return std::unexpected(cache.last_error);
            }
            input->close();

            EnvironmentImage image{
                .path = resolved_path,
                .width = spec.width,
                .height = spec.height,
                .pixels = std::vector<float>(
                    static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) * 3u),
            };
            for (int y = 0; y < spec.height; ++y) {
                for (int x = 0; x < spec.width; ++x) {
                    const size_t src_index =
                        (static_cast<size_t>(y) * static_cast<size_t>(spec.width) + static_cast<size_t>(x)) *
                        static_cast<size_t>(spec.nchannels);
                    const size_t dst_index =
                        (static_cast<size_t>(y) * static_cast<size_t>(spec.width) + static_cast<size_t>(x)) * 3u;
                    if (spec.nchannels >= 3) {
                        image.pixels[dst_index + 0] = source_pixels[src_index + 0];
                        image.pixels[dst_index + 1] = source_pixels[src_index + 1];
                        image.pixels[dst_index + 2] = source_pixels[src_index + 2];
                    } else {
                        const float value = source_pixels[src_index];
                        image.pixels[dst_index + 0] = value;
                        image.pixels[dst_index + 1] = value;
                        image.pixels[dst_index + 2] = value;
                    }
                }
            }

            cache.image = image;
            LOG_INFO("Loaded tensor environment map {}", resolved_path.string());
            return image;
        }

        [[nodiscard]] glm::vec3 acesTonemap(const glm::vec3& value) {
            constexpr float a = 2.51f;
            constexpr float b = 0.03f;
            constexpr float c = 2.43f;
            constexpr float d = 0.59f;
            constexpr float e = 0.14f;
            return glm::clamp(
                (value * (a * value + glm::vec3(b))) /
                    (value * (c * value + glm::vec3(d)) + glm::vec3(e)),
                glm::vec3(0.0f),
                glm::vec3(1.0f));
        }

        [[nodiscard]] glm::vec3 rotateAroundY(const glm::vec3& value, const float radians) {
            const float c = std::cos(radians);
            const float s = std::sin(radians);
            return {
                c * value.x + s * value.z,
                value.y,
                -s * value.x + c * value.z,
            };
        }

        [[nodiscard]] glm::vec3 sampleEnvironmentBilinear(const EnvironmentImage& image,
                                                          float u,
                                                          float v) {
            if (!image.valid()) {
                return glm::vec3(0.0f);
            }
            u = u - std::floor(u);
            v = std::clamp(v, 0.0f, 1.0f);

            const float x = u * static_cast<float>(image.width - 1);
            const float y = v * static_cast<float>(image.height - 1);
            const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, image.width - 1);
            const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, image.height - 1);
            const int x1 = (x0 + 1) % image.width;
            const int y1 = std::clamp(y0 + 1, 0, image.height - 1);
            const float tx = x - static_cast<float>(x0);
            const float ty = y - static_cast<float>(y0);

            const auto fetch = [&](const int px, const int py) {
                const size_t index =
                    (static_cast<size_t>(py) * static_cast<size_t>(image.width) + static_cast<size_t>(px)) * 3u;
                return glm::vec3(
                    image.pixels[index + 0],
                    image.pixels[index + 1],
                    image.pixels[index + 2]);
            };

            const glm::vec3 top = glm::mix(fetch(x0, y0), fetch(x1, y0), tx);
            const glm::vec3 bottom = glm::mix(fetch(x0, y1), fetch(x1, y1), tx);
            return glm::mix(top, bottom, ty);
        }

        [[nodiscard]] glm::vec3 environmentDirectionForPixel(
            const FrameView& frame_view,
            const int x,
            const int y,
            const bool equirectangular_view) {
            const float width = static_cast<float>(std::max(frame_view.size.x, 1));
            const float height = static_cast<float>(std::max(frame_view.size.y, 1));
            const float tex_u = (static_cast<float>(x) + 0.5f) / width;
            const float tex_v = 1.0f - (static_cast<float>(y) + 0.5f) / height;

            glm::vec3 local_dir;
            if (equirectangular_view) {
                const float lon = (tex_u - 0.5f) * (2.0f * glm::pi<float>());
                const float lat = (tex_v - 0.5f) * glm::pi<float>();
                const float cos_lat = std::cos(lat);
                local_dir = glm::normalize(glm::vec3(
                    std::sin(lon) * cos_lat,
                    std::sin(lat),
                    -std::cos(lon) * cos_lat));
            } else {
                float focal_x = 0.0f;
                float focal_y = 0.0f;
                float center_x = width * 0.5f;
                float center_y = height * 0.5f;
                if (frame_view.intrinsics_override.has_value() && !frame_view.orthographic) {
                    const auto& intrinsics = *frame_view.intrinsics_override;
                    focal_x = intrinsics.focal_x;
                    focal_y = intrinsics.focal_y;
                    center_x = intrinsics.center_x;
                    center_y = intrinsics.center_y;
                } else {
                    const auto focal = computePixelFocalLengths(frame_view.size, frame_view.focal_length_mm);
                    focal_x = focal.first;
                    focal_y = focal.second;
                }
                const glm::vec2 pixel(tex_u * width, tex_v * height);
                local_dir = glm::normalize(glm::vec3(
                    (pixel.x - center_x) / std::max(focal_x, 1e-6f),
                    (pixel.y - center_y) / std::max(focal_y, 1e-6f),
                    -1.0f));
            }

            return glm::normalize(frame_view.rotation * local_dir);
        }

        Result<std::vector<float>> renderEnvironmentBackground(
            const VideoCompositeFrameRequest& request,
            const int width,
            const int height) {
            const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
            std::vector<float> image(3 * pixel_count, 0.0f);

            if (!request.environment.enabled) {
                for (size_t i = 0; i < pixel_count; ++i) {
                    image[i] = request.background_color.r;
                    image[pixel_count + i] = request.background_color.g;
                    image[2 * pixel_count + i] = request.background_color.b;
                }
                return image;
            }

            auto environment = loadEnvironmentImage(request.environment.map_path);
            if (!environment) {
                return std::unexpected(environment.error());
            }

            const float exposure = std::exp2(request.environment.exposure);
            const float rotation = glm::radians(request.environment.rotation_degrees);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    glm::vec3 world_dir = environmentDirectionForPixel(
                        request.frame_view, x, y, request.environment.equirectangular);
                    world_dir = glm::normalize(rotateAroundY(world_dir, rotation));

                    const float longitude = std::atan2(world_dir.x, -world_dir.z);
                    const float latitude = std::asin(std::clamp(world_dir.y, -1.0f, 1.0f));
                    const float u = longitude / (2.0f * glm::pi<float>()) + 0.5f;
                    const float v = 0.5f - latitude / glm::pi<float>();

                    glm::vec3 color = sampleEnvironmentBilinear(*environment, u, v) * exposure;
                    color = acesTonemap(color);
                    color = glm::pow(color, glm::vec3(1.0f / 2.2f));
                    color = glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f));

                    const size_t pixel = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
                    image[pixel] = color.r;
                    image[pixel_count + pixel] = color.g;
                    image[2 * pixel_count + pixel] = color.b;
                }
            }
            return image;
        }

        [[nodiscard]] bool tensorMatchesGaussianCount(const Tensor* const tensor,
                                                      const size_t gaussian_count) {
            return tensor == nullptr || !tensor->is_valid() || tensor->numel() == gaussian_count;
        }

        [[nodiscard]] glm::vec2 computeFov(const float vfov_rad, const int width, const int height) {
            const float aspect = static_cast<float>(width) / static_cast<float>(height);
            return glm::vec2(
                std::atan(std::tan(vfov_rad * 0.5f) * aspect) * 2.0f,
                vfov_rad);
        }

        Result<lfs::core::Camera> createRasterCamera(const FrameView& frame_view,
                                                     const bool gut,
                                                     const bool equirectangular) {
            const glm::mat3 camera_to_world =
                (gut || equirectangular)
                    ? dataCameraToWorldFromVisualizerRotation(frame_view.rotation)
                    : rasterCameraToWorldFromVisualizerRotation(frame_view.rotation);
            const glm::mat3 world_to_camera = glm::transpose(camera_to_world);
            const glm::vec3 translation = -world_to_camera * frame_view.translation;

            std::vector<float> rotation_data;
            rotation_data.reserve(9);
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    rotation_data.push_back(world_to_camera[col][row]);
                }
            }

            auto rotation_tensor = Tensor::from_vector(rotation_data, {3, 3}, lfs::core::Device::CPU);
            auto translation_tensor = Tensor::from_vector(
                std::vector<float>{translation.x, translation.y, translation.z},
                {3},
                lfs::core::Device::CPU);

            float focal_x = 0.0f;
            float focal_y = 0.0f;
            float center_x = 0.0f;
            float center_y = 0.0f;
            if (frame_view.intrinsics_override.has_value()) {
                const auto& intrinsics = *frame_view.intrinsics_override;
                focal_x = intrinsics.focal_x;
                focal_y = intrinsics.focal_y;
                center_x = intrinsics.center_x;
                center_y = intrinsics.center_y;
            } else {
                const glm::vec2 fov = computeFov(
                    focalLengthToVFovRad(frame_view.focal_length_mm),
                    frame_view.size.x,
                    frame_view.size.y);
                focal_x = lfs::core::fov2focal(fov.x, frame_view.size.x);
                focal_y = lfs::core::fov2focal(fov.y, frame_view.size.y);
                center_x = frame_view.size.x / 2.0f;
                center_y = frame_view.size.y / 2.0f;
            }

            try {
                return lfs::core::Camera(
                    rotation_tensor,
                    translation_tensor,
                    focal_x,
                    focal_y,
                    center_x,
                    center_y,
                    Tensor::empty({0}, lfs::core::Device::CPU, lfs::core::DataType::Float32),
                    Tensor::empty({0}, lfs::core::Device::CPU, lfs::core::DataType::Float32),
                    lfs::core::CameraModelType::PINHOLE,
                    "render_camera",
                    "none",
                    std::filesystem::path{},
                    frame_view.size.x,
                    frame_view.size.y,
                    -1);
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Failed to create camera: {}", e.what()));
            }
        }

        [[nodiscard]] std::unique_ptr<Tensor> makeModelTransformsTensor(
            const std::vector<glm::mat4>& transforms) {
            if (transforms.empty()) {
                return nullptr;
            }

            std::vector<float> transform_data(transforms.size() * 16);
            for (size_t i = 0; i < transforms.size(); ++i) {
                const auto& mat = transforms[i];
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        transform_data[i * 16 + row * 4 + col] = mat[col][row];
                    }
                }
            }

            return std::make_unique<Tensor>(
                Tensor::from_vector(
                    transform_data,
                    {transforms.size(), 4, 4},
                    lfs::core::Device::CPU)
                    .cuda());
        }

        [[nodiscard]] Tensor* cudaTensorPointer(const std::shared_ptr<Tensor>& tensor,
                                                std::unique_ptr<Tensor>& cuda_copy) {
            if (!tensor || !tensor->is_valid()) {
                return nullptr;
            }
            if (tensor->device() == lfs::core::Device::CUDA) {
                return tensor.get();
            }
            cuda_copy = std::make_unique<Tensor>(tensor->cuda());
            return cuda_copy.get();
        }

        void applyCropBoxToRaster(GaussianRasterRequest& request,
                                  GaussianRasterResources& resources) {
            if (!request.filters.crop_region.has_value()) {
                return;
            }

            const auto& crop = *request.filters.crop_region;
            const glm::mat4& world_to_box = crop.bounds.transform;
            std::vector<float> transform_data(16);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    transform_data[row * 4 + col] = world_to_box[col][row];
                }
            }

            resources.crop_box_transform_tensor =
                Tensor::from_vector(transform_data, {4, 4}, lfs::core::Device::CPU).cuda();
            resources.crop_box_min_tensor =
                Tensor::from_vector(
                    std::vector<float>{crop.bounds.min.x, crop.bounds.min.y, crop.bounds.min.z},
                    {3},
                    lfs::core::Device::CPU)
                    .cuda();
            resources.crop_box_max_tensor =
                Tensor::from_vector(
                    std::vector<float>{crop.bounds.max.x, crop.bounds.max.y, crop.bounds.max.z},
                    {3},
                    lfs::core::Device::CPU)
                    .cuda();
        }

        void applyEllipsoidToRaster(GaussianRasterRequest& request,
                                    GaussianRasterResources& resources) {
            if (!request.filters.ellipsoid_region.has_value()) {
                return;
            }

            const auto& ellipsoid = *request.filters.ellipsoid_region;
            const glm::mat4& world_to_ellipsoid = ellipsoid.bounds.transform;
            std::vector<float> transform_data(16);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    transform_data[row * 4 + col] = world_to_ellipsoid[col][row];
                }
            }

            resources.ellipsoid_transform_tensor =
                Tensor::from_vector(transform_data, {4, 4}, lfs::core::Device::CPU).cuda();
            resources.ellipsoid_radii_tensor =
                Tensor::from_vector(
                    std::vector<float>{
                        ellipsoid.bounds.radii.x,
                        ellipsoid.bounds.radii.y,
                        ellipsoid.bounds.radii.z},
                    {3},
                    lfs::core::Device::CPU)
                    .cuda();
        }

        void applyViewVolumeToRaster(GaussianRasterRequest& request,
                                     GaussianRasterResources& resources) {
            if (!request.filters.view_volume.has_value()) {
                return;
            }

            const auto& view_volume = *request.filters.view_volume;
            const glm::mat4& world_to_volume = view_volume.transform;
            std::vector<float> transform_data(16);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    transform_data[row * 4 + col] = world_to_volume[col][row];
                }
            }

            resources.view_volume_transform_tensor =
                Tensor::from_vector(transform_data, {4, 4}, lfs::core::Device::CPU).cuda();
            resources.view_volume_min_tensor =
                Tensor::from_vector(
                    std::vector<float>{view_volume.min.x, view_volume.min.y, view_volume.min.z},
                    {3},
                    lfs::core::Device::CPU)
                    .cuda();
            resources.view_volume_max_tensor =
                Tensor::from_vector(
                    std::vector<float>{view_volume.max.x, view_volume.max.y, view_volume.max.z},
                    {3},
                    lfs::core::Device::CPU)
                    .cuda();
        }

        [[nodiscard]] FrameMetadata makeFrameMetadata(const RasterImageResult& result) {
            return FrameMetadata{
                .depth_panels = {FramePanelMetadata{
                    .depth = result.depth.is_valid() ? std::make_shared<Tensor>(result.depth) : nullptr,
                    .start_position = 0.0f,
                    .end_position = 1.0f,
                }},
                .depth_panel_count = 1,
                .valid = result.valid,
                .flip_y = result.flip_y,
                .far_plane = result.far_plane,
                .orthographic = result.orthographic,
                .color_has_alpha = result.color_has_alpha};
        }

        [[nodiscard]] glm::mat3 makeAxisViewRotation(const int axis, const bool negative) {
            const float sign = negative ? -1.0f : 1.0f;

            glm::vec3 forward;
            glm::vec3 up;
            switch (axis) {
            case 0:
                forward = glm::vec3(-sign, 0.0f, 0.0f);
                up = glm::vec3(0.0f, 1.0f, 0.0f);
                break;
            case 1:
                forward = glm::vec3(0.0f, -sign, 0.0f);
                up = glm::vec3(0.0f, 0.0f, sign);
                break;
            case 2:
                forward = glm::vec3(0.0f, 0.0f, -sign);
                up = glm::vec3(0.0f, 1.0f, 0.0f);
                break;
            default:
                return glm::mat3(1.0f);
            }

            return makeVisualizerLookAtRotation(glm::vec3(0.0f), forward, up);
        }

        [[nodiscard]] FrameMetadata makePointCloudFrameMetadata(
            const RasterImageResult& result) {
            return FrameMetadata{
                .depth_panels = {FramePanelMetadata{
                    .depth = result.depth.is_valid() ? std::make_shared<Tensor>(result.depth) : nullptr,
                    .start_position = 0.0f,
                    .end_position = 1.0f,
                }},
                .depth_panel_count = 1,
                .valid = result.valid,
                .flip_y = result.flip_y,
                .far_plane = result.far_plane,
                .orthographic = result.orthographic,
                .color_has_alpha = result.color_has_alpha};
        }

        [[nodiscard]] int readTensorIndex(const Tensor* const tensor,
                                          const size_t index) {
            if (!tensor || !tensor->is_valid() || index >= tensor->numel()) {
                return 0;
            }

            switch (tensor->dtype()) {
            case lfs::core::DataType::Float32:
                return static_cast<int>(std::lround(tensor->ptr<float>()[index]));
            case lfs::core::DataType::Int32:
                return tensor->ptr<int>()[index];
            case lfs::core::DataType::Int64:
                return static_cast<int>(tensor->ptr<int64_t>()[index]);
            case lfs::core::DataType::UInt8:
            case lfs::core::DataType::Bool:
                return static_cast<int>(tensor->ptr<unsigned char>()[index]);
            default:
                return 0;
            }
        }

        [[nodiscard]] std::optional<glm::mat4> cameraVisualizerTransform(
            const lfs::core::Camera& camera,
            const glm::mat4& scene_transform) {
            auto rotation_tensor = camera.R();
            auto translation_tensor = camera.T();
            if (!rotation_tensor.is_valid() || !translation_tensor.is_valid()) {
                return std::nullopt;
            }
            if (rotation_tensor.device() != lfs::core::Device::CPU) {
                rotation_tensor = rotation_tensor.cpu();
            }
            if (translation_tensor.device() != lfs::core::Device::CPU) {
                translation_tensor = translation_tensor.cpu();
            }
            if (rotation_tensor.dtype() != lfs::core::DataType::Float32 ||
                translation_tensor.dtype() != lfs::core::DataType::Float32 ||
                rotation_tensor.numel() < 9 || translation_tensor.numel() < 3) {
                return std::nullopt;
            }

            glm::mat4 world_to_camera(1.0f);
            const float* const rotation = rotation_tensor.ptr<float>();
            const float* const translation = translation_tensor.ptr<float>();
            if (!rotation || !translation) {
                return std::nullopt;
            }
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    world_to_camera[col][row] = rotation[row * 3 + col];
                }
                world_to_camera[3][row] = translation[row];
            }

            return scene_transform * glm::inverse(world_to_camera) * DATA_TO_VISUALIZER_CAMERA_AXES_4;
        }

        [[nodiscard]] std::vector<glm::vec3> cameraFrustumWorldPoints(
            const lfs::core::Camera& camera,
            const glm::mat4& visualizer_camera_to_world,
            const float scale) {
            std::vector<glm::vec3> points;
            const int image_width = camera.image_width() > 0 ? camera.image_width() : camera.camera_width();
            const int image_height = camera.image_height() > 0 ? camera.image_height() : camera.camera_height();
            if (image_width <= 0 || image_height <= 0 || scale <= 0.0f) {
                return points;
            }

            const bool equirectangular =
                camera.camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR;
            if (equirectangular) {
                constexpr int SEGMENTS = 48;
                points.reserve(SEGMENTS * 3);
                for (int circle = 0; circle < 3; ++circle) {
                    for (int i = 0; i < SEGMENTS; ++i) {
                        const float a = static_cast<float>(i) / static_cast<float>(SEGMENTS) *
                                        2.0f * glm::pi<float>();
                        glm::vec3 local(0.0f);
                        if (circle == 0) {
                            local = {std::cos(a), std::sin(a), 0.0f};
                        } else if (circle == 1) {
                            local = {std::cos(a), 0.0f, std::sin(a)};
                        } else {
                            local = {0.0f, std::cos(a), std::sin(a)};
                        }
                        points.push_back(glm::vec3(
                            visualizer_camera_to_world * glm::vec4(local * scale, 1.0f)));
                    }
                }
                return points;
            }

            if (camera.focal_y() <= 0.0f) {
                return points;
            }

            const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
            const float fov_y = lfs::core::focal2fov(camera.focal_y(), image_height);
            const float half_height = std::tan(fov_y * 0.5f) * scale;
            const float half_width = half_height * aspect;

            const std::array local_points{
                glm::vec3(0.0f, 0.0f, 0.0f),
                glm::vec3(-half_width, half_height, -scale),
                glm::vec3(half_width, half_height, -scale),
                glm::vec3(half_width, -half_height, -scale),
                glm::vec3(-half_width, -half_height, -scale),
            };
            points.reserve(local_points.size());
            for (const glm::vec3& local : local_points) {
                points.push_back(glm::vec3(visualizer_camera_to_world * glm::vec4(local, 1.0f)));
            }
            return points;
        }

        [[nodiscard]] float pointSegmentDistance(
            const glm::vec2& point,
            const glm::vec2& a,
            const glm::vec2& b) {
            const glm::vec2 ab = b - a;
            const float denom = glm::dot(ab, ab);
            if (denom <= 1e-6f) {
                return glm::length(point - a);
            }
            const float t = std::clamp(glm::dot(point - a, ab) / denom, 0.0f, 1.0f);
            return glm::length(point - (a + ab * t));
        }

        [[nodiscard]] std::optional<glm::vec2> projectFrustumPoint(
            const glm::vec3& world_point,
            const CameraFrustumPickRequest& request) {
            const auto projected = projectWorldPoint(
                request.viewport.rotation,
                request.viewport.translation,
                request.viewport.size,
                world_point,
                request.viewport.focal_length_mm,
                request.viewport.orthographic,
                request.viewport.ortho_scale);
            if (!projected) {
                return std::nullopt;
            }

            const float scale_x = request.viewport_size.x /
                                  static_cast<float>(std::max(request.viewport.size.x, 1));
            const float scale_y = request.viewport_size.y /
                                  static_cast<float>(std::max(request.viewport.size.y, 1));
            return glm::vec2(
                request.viewport_pos.x + projected->x * scale_x,
                request.viewport_pos.y + projected->y * scale_y);
        }

        [[nodiscard]] glm::vec3 readPointColor(const float* const colors,
                                               const size_t point_index,
                                               const bool desaturate) {
            glm::vec3 color(
                std::clamp(colors[point_index * 3 + 0], 0.0f, 1.0f),
                std::clamp(colors[point_index * 3 + 1], 0.0f, 1.0f),
                std::clamp(colors[point_index * 3 + 2], 0.0f, 1.0f));

            if (desaturate) {
                const float gray = glm::dot(color, glm::vec3(0.299f, 0.587f, 0.114f));
                color = glm::mix(color, glm::vec3(gray), 0.75f);
            }
            return color;
        }

        [[nodiscard]] bool pointPassesCrop(const glm::vec3& world_pos,
                                           const PointCloudFilterState& filters,
                                           bool& desaturate) {
            desaturate = false;
            if (!filters.crop_box.has_value()) {
                return true;
            }

            const auto& crop = *filters.crop_box;
            const glm::vec3 local = glm::vec3(crop.transform * glm::vec4(world_pos, 1.0f));
            const bool inside =
                local.x >= crop.min.x && local.x <= crop.max.x &&
                local.y >= crop.min.y && local.y <= crop.max.y &&
                local.z >= crop.min.z && local.z <= crop.max.z;
            const bool visible = filters.crop_inverse ? !inside : inside;
            desaturate = filters.crop_desaturate && !visible;
            return visible || filters.crop_desaturate;
        }

        [[nodiscard]] std::optional<glm::vec3> projectPointToPixel(
            const glm::vec3& world_pos,
            const PointCloudRenderRequest& request,
            const glm::mat4& view,
            const glm::mat4& projection) {
            const glm::vec4 view_pos4 = view * glm::vec4(world_pos, 1.0f);
            const glm::vec3 view_pos(view_pos4);
            const int width = request.frame_view.size.x;
            const int height = request.frame_view.size.y;

            if (request.render.equirectangular) {
                const float len = glm::length(view_pos);
                if (len <= 1e-6f) {
                    return std::nullopt;
                }
                const glm::vec3 dir = view_pos / len;
                const float u = 0.5f + std::atan2(dir.x, -dir.z) / (2.0f * glm::pi<float>());
                const float v = 0.5f + std::asin(std::clamp(dir.y, -1.0f, 1.0f)) / glm::pi<float>();
                const float px = u * static_cast<float>(width - 1);
                const float py = v * static_cast<float>(height - 1);
                if (!std::isfinite(px) || !std::isfinite(py) ||
                    px < 0.0f || px >= static_cast<float>(width) ||
                    py < 0.0f || py >= static_cast<float>(height)) {
                    return std::nullopt;
                }
                return glm::vec3(px, py, len);
            }

            const glm::vec4 clip = projection * view_pos4;
            if (std::abs(clip.w) <= 1e-6f) {
                return std::nullopt;
            }
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            if (!std::isfinite(ndc.x) || !std::isfinite(ndc.y) || !std::isfinite(ndc.z) ||
                ndc.x < -1.0f || ndc.x > 1.0f ||
                ndc.y < -1.0f || ndc.y > 1.0f ||
                ndc.z < -1.0f || ndc.z > 1.0f) {
                return std::nullopt;
            }

            const float px = (ndc.x * 0.5f + 0.5f) * static_cast<float>(width - 1);
            const float py = (ndc.y * 0.5f + 0.5f) * static_cast<float>(height - 1);
            const float depth = request.frame_view.orthographic ? -view_pos.z : std::max(-view_pos.z, 0.0f);
            if (depth <= 0.0f && !request.frame_view.orthographic) {
                return std::nullopt;
            }
            return glm::vec3(px, py, depth);
        }

        [[nodiscard]] int pointRadiusPixels(const PointCloudRenderRequest& request,
                                            const float depth) {
            const float voxel = std::max(request.render.voxel_size * request.render.scaling_modifier, 1e-5f);
            if (request.frame_view.orthographic) {
                const float pixels_per_world =
                    static_cast<float>(request.frame_view.size.y) /
                    std::max(request.frame_view.ortho_scale, 1e-5f);
                return std::max(1, static_cast<int>(std::ceil(voxel * pixels_per_world * 0.5f)));
            }

            const float vfov = focalLengthToVFovRad(request.frame_view.focal_length_mm);
            const float focal_y = lfs::core::fov2focal(vfov, request.frame_view.size.y);
            return std::max(1, static_cast<int>(std::ceil(voxel * focal_y / std::max(depth, 1e-4f))));
        }

        void drawSoftwarePoint(std::vector<float>& image,
                               std::vector<float>& depth,
                               const int width,
                               const int height,
                               const int channels,
                               const glm::vec3& pixel_depth,
                               const glm::vec3& color,
                               const int radius) {
            const int cx = static_cast<int>(std::lround(pixel_depth.x));
            const int cy = static_cast<int>(std::lround(pixel_depth.y));
            const float point_depth = pixel_depth.z;
            const int r2 = radius * radius;

            for (int yy = cy - radius; yy <= cy + radius; ++yy) {
                if (yy < 0 || yy >= height) {
                    continue;
                }
                for (int xx = cx - radius; xx <= cx + radius; ++xx) {
                    if (xx < 0 || xx >= width) {
                        continue;
                    }
                    const int dx = xx - cx;
                    const int dy = yy - cy;
                    if (dx * dx + dy * dy > r2) {
                        continue;
                    }

                    const size_t pixel_index = static_cast<size_t>(yy) * width + xx;
                    if (point_depth >= depth[pixel_index]) {
                        continue;
                    }
                    depth[pixel_index] = point_depth;
                    image[pixel_index] = color.r;
                    image[static_cast<size_t>(height) * width + pixel_index] = color.g;
                    image[static_cast<size_t>(2) * height * width + pixel_index] = color.b;
                    if (channels == 4) {
                        image[static_cast<size_t>(3) * height * width + pixel_index] = 1.0f;
                    }
                }
            }
        }

        Result<RasterImageResult> renderSoftwarePointCloud(
            const Tensor& positions_source,
            const Tensor& colors_source,
            const PointCloudRenderRequest& request) {
            if (request.frame_view.size.x <= 0 || request.frame_view.size.y <= 0 ||
                request.frame_view.size.x > MAX_VIEWPORT_SIZE ||
                request.frame_view.size.y > MAX_VIEWPORT_SIZE) {
                return std::unexpected("Invalid viewport dimensions");
            }
            if (!positions_source.is_valid() || positions_source.ndim() != 2 || positions_source.size(1) != 3) {
                return std::unexpected("Point cloud positions must have shape [N, 3]");
            }
            if (!colors_source.is_valid() || colors_source.ndim() != 2 || colors_source.size(1) != 3 ||
                colors_source.size(0) != positions_source.size(0)) {
                return std::unexpected("Point cloud colors must have shape [N, 3]");
            }

            Tensor positions_cpu = positions_source.cpu().contiguous();
            Tensor colors_cpu = colors_source;
            if (colors_cpu.dtype() == lfs::core::DataType::UInt8) {
                colors_cpu = colors_cpu.to(lfs::core::DataType::Float32) / 255.0f;
            }
            colors_cpu = colors_cpu.cpu().contiguous();
            if (colors_cpu.dtype() != lfs::core::DataType::Float32) {
                colors_cpu = colors_cpu.to(lfs::core::DataType::Float32).cpu().contiguous();
            }

            Tensor transform_indices_cpu;
            const Tensor* transform_indices = nullptr;
            if (request.scene.transform_indices && request.scene.transform_indices->is_valid() &&
                request.scene.transform_indices->numel() == positions_source.size(0)) {
                transform_indices_cpu = request.scene.transform_indices->cpu().contiguous();
                transform_indices = &transform_indices_cpu;
            }

            const std::vector<glm::mat4>* const transforms_ptr = request.scene.model_transforms;
            const std::vector<glm::mat4> empty_transforms;
            const auto& transforms = transforms_ptr ? *transforms_ptr : empty_transforms;
            const glm::mat4 view = request.frame_view.getViewMatrix();
            const glm::mat4 projection = createProjectionMatrix(
                request.frame_view.size,
                focalLengthToVFov(request.frame_view.focal_length_mm),
                request.frame_view.orthographic,
                request.frame_view.ortho_scale,
                request.frame_view.near_plane,
                request.frame_view.far_plane);

            const int width = request.frame_view.size.x;
            const int height = request.frame_view.size.y;
            const int channels = request.transparent_background ? 4 : 3;
            const size_t pixel_count = static_cast<size_t>(width) * height;
            std::vector<float> image(static_cast<size_t>(channels) * pixel_count, 0.0f);
            std::vector<float> depth(pixel_count, request.frame_view.far_plane);
            for (size_t i = 0; i < pixel_count; ++i) {
                image[i] = request.frame_view.background_color.r;
                image[pixel_count + i] = request.frame_view.background_color.g;
                image[2 * pixel_count + i] = request.frame_view.background_color.b;
                if (channels == 4) {
                    image[3 * pixel_count + i] = request.transparent_background ? 0.0f : 1.0f;
                }
            }

            const float* const positions = positions_cpu.ptr<float>();
            const float* const colors = colors_cpu.ptr<float>();
            if (!positions || !colors) {
                return std::unexpected("Point cloud tensors have no readable storage");
            }

            const size_t count = positions_source.size(0);
            for (size_t i = 0; i < count; ++i) {
                int transform_index = readTensorIndex(transform_indices, i);
                if (transform_index < 0) {
                    transform_index = 0;
                }
                if (!request.scene.node_visibility_mask.empty() &&
                    static_cast<size_t>(transform_index) < request.scene.node_visibility_mask.size() &&
                    !request.scene.node_visibility_mask[static_cast<size_t>(transform_index)]) {
                    continue;
                }

                glm::vec3 position(
                    positions[i * 3 + 0],
                    positions[i * 3 + 1],
                    positions[i * 3 + 2]);
                if (!transforms.empty()) {
                    const size_t clamped_index =
                        std::min(static_cast<size_t>(transform_index), transforms.size() - 1);
                    position = glm::vec3(transforms[clamped_index] * glm::vec4(position, 1.0f));
                }

                bool desaturate = false;
                if (!pointPassesCrop(position, request.filters, desaturate)) {
                    continue;
                }

                const auto pixel_depth = projectPointToPixel(position, request, view, projection);
                if (!pixel_depth) {
                    continue;
                }

                const glm::vec3 color = readPointColor(colors, i, desaturate);
                const int radius = pointRadiusPixels(request, pixel_depth->z);
                drawSoftwarePoint(image, depth, width, height, channels, *pixel_depth, color, radius);
            }

            Tensor image_tensor = Tensor::from_vector(
                image,
                {static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CPU)
                                      .cuda();
            Tensor depth_tensor = Tensor::from_vector(
                depth,
                {static_cast<size_t>(1), static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CPU)
                                      .cuda();

            return RasterImageResult{
                .image = std::move(image_tensor),
                .depth = std::move(depth_tensor),
                .valid = true,
                .far_plane = request.frame_view.far_plane,
                .orthographic = request.frame_view.orthographic,
                .color_has_alpha = request.transparent_background};
        }

        [[nodiscard]] Result<Tensor> toCpuChwFloatTensor(const Tensor& image) {
            if (!image.is_valid() || image.ndim() != 3) {
                return std::unexpected("Invalid image tensor");
            }
            const auto layout = detectImageLayout(image);
            if (layout == ImageLayout::Unknown) {
                return std::unexpected("Unsupported image tensor layout");
            }
            Tensor formatted = image;
            if (formatted.dtype() == lfs::core::DataType::UInt8) {
                formatted = formatted.to(lfs::core::DataType::Float32) / 255.0f;
            } else if (formatted.dtype() != lfs::core::DataType::Float32) {
                formatted = formatted.to(lfs::core::DataType::Float32);
            }
            if (layout == ImageLayout::HWC) {
                formatted = formatted.permute({2, 0, 1}).contiguous();
            }
            return formatted.cpu().contiguous();
        }

        [[nodiscard]] std::optional<glm::vec3> projectMeshPoint(
            const glm::vec3& world_pos,
            const ViewportData& viewport,
            const glm::mat4& view,
            const glm::mat4& projection) {
            const glm::vec4 view_pos4 = view * glm::vec4(world_pos, 1.0f);
            const glm::vec4 clip = projection * view_pos4;
            if (std::abs(clip.w) <= 1e-6f) {
                return std::nullopt;
            }
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            if (!std::isfinite(ndc.x) || !std::isfinite(ndc.y) || !std::isfinite(ndc.z) ||
                ndc.x < -1.0f || ndc.x > 1.0f ||
                ndc.y < -1.0f || ndc.y > 1.0f ||
                ndc.z < -1.0f || ndc.z > 1.0f) {
                return std::nullopt;
            }
            const float x = (ndc.x * 0.5f + 0.5f) * static_cast<float>(viewport.size.x - 1);
            const float y = (ndc.y * 0.5f + 0.5f) * static_cast<float>(viewport.size.y - 1);
            const float z = viewport.orthographic ? -view_pos4.z : std::max(-view_pos4.z, 0.0f);
            if (z <= 0.0f && !viewport.orthographic) {
                return std::nullopt;
            }
            return glm::vec3(x, y, z);
        }

        [[nodiscard]] float edgeFunction(const glm::vec2& a,
                                         const glm::vec2& b,
                                         const glm::vec2& c) {
            return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
        }

        void drawMeshLine(std::vector<float>& image,
                          const int width,
                          const int height,
                          const glm::vec2& a,
                          const glm::vec2& b,
                          const glm::vec3& color,
                          const float thickness) {
            const glm::vec2 delta = b - a;
            const int steps = std::max(1, static_cast<int>(std::ceil(glm::length(delta))));
            const int radius = std::max(1, static_cast<int>(std::ceil(thickness * 0.5f)));
            const size_t pixel_count = static_cast<size_t>(width) * height;
            for (int i = 0; i <= steps; ++i) {
                const float t = static_cast<float>(i) / static_cast<float>(steps);
                const glm::vec2 p = glm::mix(a, b, t);
                const int cx = static_cast<int>(std::lround(p.x));
                const int cy = static_cast<int>(std::lround(p.y));
                for (int yy = cy - radius; yy <= cy + radius; ++yy) {
                    if (yy < 0 || yy >= height) {
                        continue;
                    }
                    for (int xx = cx - radius; xx <= cx + radius; ++xx) {
                        if (xx < 0 || xx >= width) {
                            continue;
                        }
                        const size_t pixel = static_cast<size_t>(yy) * width + xx;
                        image[pixel] = color.r;
                        image[pixel_count + pixel] = color.g;
                        image[2 * pixel_count + pixel] = color.b;
                    }
                }
            }
        }

        void rasterizeMeshTriangle(std::vector<float>& image,
                                   std::vector<float>& depth,
                                   const int width,
                                   const int height,
                                   const std::array<glm::vec3, 3>& screen,
                                   const glm::vec3& color) {
            const glm::vec2 p0(screen[0]);
            const glm::vec2 p1(screen[1]);
            const glm::vec2 p2(screen[2]);
            const float area = edgeFunction(p0, p1, p2);
            if (std::abs(area) <= 1e-6f) {
                return;
            }

            const int min_x = std::clamp(
                static_cast<int>(std::floor(std::min({p0.x, p1.x, p2.x}))), 0, width - 1);
            const int max_x = std::clamp(
                static_cast<int>(std::ceil(std::max({p0.x, p1.x, p2.x}))), 0, width - 1);
            const int min_y = std::clamp(
                static_cast<int>(std::floor(std::min({p0.y, p1.y, p2.y}))), 0, height - 1);
            const int max_y = std::clamp(
                static_cast<int>(std::ceil(std::max({p0.y, p1.y, p2.y}))), 0, height - 1);
            const size_t pixel_count = static_cast<size_t>(width) * height;

            for (int y = min_y; y <= max_y; ++y) {
                for (int x = min_x; x <= max_x; ++x) {
                    const glm::vec2 p(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
                    const float w0 = edgeFunction(p1, p2, p) / area;
                    const float w1 = edgeFunction(p2, p0, p) / area;
                    const float w2 = edgeFunction(p0, p1, p) / area;
                    if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) {
                        continue;
                    }
                    const float z = w0 * screen[0].z + w1 * screen[1].z + w2 * screen[2].z;
                    const size_t pixel = static_cast<size_t>(y) * width + x;
                    if (z >= depth[pixel]) {
                        continue;
                    }
                    depth[pixel] = z;
                    image[pixel] = color.r;
                    image[pixel_count + pixel] = color.g;
                    image[2 * pixel_count + pixel] = color.b;
                }
            }
        }

        Result<Tensor> renderSoftwareVideoComposite(
            const std::shared_ptr<lfs::core::Tensor>& primary_image,
            const FrameMetadata* primary_metadata,
            const VideoCompositeFrameRequest& request) {
            const int width = request.frame_view.size.x > 0 ? request.frame_view.size.x : request.viewport.size.x;
            const int height = request.frame_view.size.y > 0 ? request.frame_view.size.y : request.viewport.size.y;
            if (width <= 0 || height <= 0) {
                return std::unexpected("Invalid video composite dimensions");
            }

            const size_t pixel_count = static_cast<size_t>(width) * height;
            auto background = renderEnvironmentBackground(request, width, height);
            if (!background) {
                return std::unexpected(background.error());
            }
            std::vector<float> image = std::move(*background);
            std::vector<float> depth(pixel_count, request.frame_view.far_plane);

            if (primary_image && primary_image->is_valid()) {
                auto cpu_image = toCpuChwFloatTensor(*primary_image);
                if (!cpu_image) {
                    return std::unexpected(cpu_image.error());
                }
                const auto& img = *cpu_image;
                const auto layout = detectImageLayout(img);
                const int src_w = imageWidth(img, layout);
                const int src_h = imageHeight(img, layout);
                const int channels = imageChannels(img, layout);
                const float* src = img.ptr<float>();
                for (int y = 0; y < height; ++y) {
                    const int sy = std::clamp(static_cast<int>(
                                                  static_cast<float>(y) * src_h / std::max(height, 1)),
                                              0, src_h - 1);
                    for (int x = 0; x < width; ++x) {
                        const int sx = std::clamp(static_cast<int>(
                                                      static_cast<float>(x) * src_w / std::max(width, 1)),
                                                  0, src_w - 1);
                        const size_t dst = static_cast<size_t>(y) * width + x;
                        const size_t src_pixel = static_cast<size_t>(sy) * src_w + sx;
                        const float src_r = src[src_pixel];
                        const float src_g = src[static_cast<size_t>(1) * src_h * src_w + src_pixel];
                        const float src_b = src[static_cast<size_t>(2) * src_h * src_w + src_pixel];
                        if (channels == 4) {
                            const float alpha = src[static_cast<size_t>(3) * src_h * src_w + src_pixel];
                            image[dst] = glm::mix(image[dst], src_r, alpha);
                            image[pixel_count + dst] = glm::mix(image[pixel_count + dst], src_g, alpha);
                            image[2 * pixel_count + dst] = glm::mix(image[2 * pixel_count + dst], src_b, alpha);
                        } else {
                            image[dst] = src_r;
                            image[pixel_count + dst] = src_g;
                            image[2 * pixel_count + dst] = src_b;
                        }
                    }
                }

                if (primary_metadata && primary_metadata->primaryDepth() &&
                    primary_metadata->primaryDepth()->is_valid()) {
                    Tensor depth_cpu = primary_metadata->primaryDepth()->cpu().contiguous();
                    if (depth_cpu.ndim() == 3 && depth_cpu.dtype() == lfs::core::DataType::Float32) {
                        const int depth_h = static_cast<int>(depth_cpu.size(1));
                        const int depth_w = static_cast<int>(depth_cpu.size(2));
                        const float* depth_src = depth_cpu.ptr<float>();
                        for (int y = 0; y < height; ++y) {
                            const int sy = std::clamp(static_cast<int>(
                                                          static_cast<float>(y) * depth_h / std::max(height, 1)),
                                                      0, depth_h - 1);
                            for (int x = 0; x < width; ++x) {
                                const int sx = std::clamp(static_cast<int>(
                                                              static_cast<float>(x) * depth_w / std::max(width, 1)),
                                                          0, depth_w - 1);
                                depth[static_cast<size_t>(y) * width + x] =
                                    depth_src[static_cast<size_t>(sy) * depth_w + sx];
                            }
                        }
                    }
                }
            }

            const glm::mat4 view = request.viewport.getViewMatrix();
            const glm::mat4 projection = request.viewport.getProjectionMatrix();
            const glm::vec3 light_dir = glm::normalize(glm::length(request.meshes.empty()
                                                                       ? glm::vec3(0.3f, 1.0f, 0.5f)
                                                                       : request.meshes.front().options.light_dir) > 1e-5f
                                                           ? (request.meshes.empty()
                                                                  ? glm::vec3(0.3f, 1.0f, 0.5f)
                                                                  : request.meshes.front().options.light_dir)
                                                           : glm::vec3(0.3f, 1.0f, 0.5f));

            for (const auto& item : request.meshes) {
                if (!item.mesh || !item.mesh->vertices.is_valid() || !item.mesh->indices.is_valid()) {
                    continue;
                }
                Tensor vertices_cpu = item.mesh->vertices.cpu().contiguous();
                Tensor indices_cpu = item.mesh->indices.cpu().contiguous();
                if (vertices_cpu.dtype() != lfs::core::DataType::Float32 ||
                    indices_cpu.dtype() != lfs::core::DataType::Int32 ||
                    vertices_cpu.ndim() != 2 || vertices_cpu.size(1) != 3 ||
                    indices_cpu.ndim() != 2 || indices_cpu.size(1) != 3) {
                    continue;
                }

                const float* vertices = vertices_cpu.ptr<float>();
                const int* indices = indices_cpu.ptr<int>();
                const size_t face_count = indices_cpu.size(0);
                const glm::vec3 base_color =
                    item.mesh->materials.empty()
                        ? glm::vec3(0.8f)
                        : glm::vec3(item.mesh->materials.front().base_color);

                for (size_t face = 0; face < face_count; ++face) {
                    std::array<glm::vec3, 3> world{};
                    std::array<glm::vec3, 3> screen{};
                    bool visible = true;
                    for (int corner = 0; corner < 3; ++corner) {
                        const int idx = indices[face * 3 + corner];
                        if (idx < 0 || static_cast<size_t>(idx) >= vertices_cpu.size(0)) {
                            visible = false;
                            break;
                        }
                        const glm::vec3 local(
                            vertices[static_cast<size_t>(idx) * 3 + 0],
                            vertices[static_cast<size_t>(idx) * 3 + 1],
                            vertices[static_cast<size_t>(idx) * 3 + 2]);
                        world[corner] = glm::vec3(item.transform * glm::vec4(local, 1.0f));
                        const auto projected = projectMeshPoint(world[corner], request.viewport, view, projection);
                        if (!projected) {
                            visible = false;
                            break;
                        }
                        screen[corner] = *projected;
                    }
                    if (!visible) {
                        continue;
                    }

                    const glm::vec3 normal = glm::normalize(glm::cross(world[1] - world[0], world[2] - world[0]));
                    if (item.options.backface_culling && glm::dot(normal, glm::vec3(view[0][2], view[1][2], view[2][2])) >= 0.0f) {
                        continue;
                    }
                    const float diffuse = std::max(glm::dot(normal, glm::normalize(-light_dir)), 0.0f);
                    glm::vec3 color = base_color * std::clamp(item.options.ambient + diffuse * item.options.light_intensity, 0.0f, 1.5f);
                    if (item.options.dim_non_emphasized && !item.options.is_emphasized) {
                        const float gray = glm::dot(color, glm::vec3(0.299f, 0.587f, 0.114f));
                        color = glm::mix(color, glm::vec3(gray), 0.75f);
                    }
                    if (item.options.flash_intensity > 0.0f && item.options.is_emphasized) {
                        color = glm::mix(color, glm::vec3(1.0f), std::clamp(item.options.flash_intensity, 0.0f, 1.0f));
                    }
                    rasterizeMeshTriangle(image, depth, width, height, screen, glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f)));

                    if (item.options.wireframe_overlay) {
                        drawMeshLine(image, width, height, screen[0], screen[1], item.options.wireframe_color, item.options.wireframe_width);
                        drawMeshLine(image, width, height, screen[1], screen[2], item.options.wireframe_color, item.options.wireframe_width);
                        drawMeshLine(image, width, height, screen[2], screen[0], item.options.wireframe_color, item.options.wireframe_width);
                    }
                }
            }

            return Tensor::from_vector(
                image,
                {static_cast<size_t>(3), static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CPU)
                .cuda();
        }
    } // namespace

    class RasterOnlyRenderingEngine final : public RenderingEngine {
    public:
        ~RasterOnlyRenderingEngine() override {
            shutdown();
        }

        Result<void> initialize() override {
            return initializeRasterOnly();
        }

        Result<void> initializeRasterOnly() override {
            if (!background_.is_valid()) {
                background_ = Tensor::zeros({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            }
            raster_initialized_ = true;
            return {};
        }

        void shutdown() override {
            raster_initialized_ = false;
            if (hovered_depth_id_device_) {
                cudaFree(hovered_depth_id_device_);
                hovered_depth_id_device_ = nullptr;
            }
            if (hovered_depth_id_host_) {
                cudaFreeHost(hovered_depth_id_host_);
                hovered_depth_id_host_ = nullptr;
            }
        }

        bool isInitialized() const override {
            return raster_initialized_;
        }

        bool isRasterInitialized() const override {
            return raster_initialized_;
        }

        Result<GaussianGpuFrameResult> renderGaussiansGpuFrame(
            const lfs::core::SplatData& splat_data,
            const ViewportRenderRequest& request) override {
            auto image_result = renderGaussiansImage(splat_data, request);
            if (!image_result || !image_result->image) {
                return std::unexpected(image_result ? "Gaussian GPU-frame render returned no image"
                                                    : image_result.error());
            }

            return GaussianGpuFrameResult{
                .frame = cacheTensorFrame(image_result->image, image_result->metadata, request.frame_view.size),
                .metadata = std::move(image_result->metadata)};
        }

        Result<GaussianImageResult> renderGaussiansImage(
            const lfs::core::SplatData& splat_data,
            const ViewportRenderRequest& request) override {
            auto result = renderRaster(
                splat_data,
                GaussianRasterRequest{
                    .frame_view = request.frame_view,
                    .scaling_modifier = request.scaling_modifier,
                    .antialiasing = request.antialiasing,
                    .mip_filter = request.mip_filter,
                    .sh_degree = request.sh_degree,
                    .gut = request.gut,
                    .equirectangular = request.equirectangular,
                    .scene = request.scene,
                    .filters = request.filters,
                    .overlay = request.overlay,
                    .transparent_background = request.transparent_background});
            if (!result) {
                return std::unexpected(result.error());
            }

            return GaussianImageResult{
                .image = std::make_shared<Tensor>(std::move(result->image)),
                .metadata = makeFrameMetadata(*result)};
        }

        Result<DualGaussianImageResult> renderGaussiansImagePair(
            const lfs::core::SplatData& splat_data,
            const std::array<ViewportRenderRequest, 2>& requests) override {
            DualGaussianImageResult pair_result;
            for (size_t i = 0; i < pair_result.size(); ++i) {
                auto single = renderGaussiansImage(splat_data, requests[i]);
                if (!single) {
                    return std::unexpected(single.error());
                }
                pair_result[i] = std::move(*single);
            }
            return pair_result;
        }

        Result<std::optional<int>> queryHoveredGaussianId(
            const lfs::core::SplatData& splat_data,
            const HoveredGaussianQueryRequest& request) override {
            if (!ensureHoveredDepthQueryBuffersAllocated()) {
                return std::unexpected("Failed to allocate hovered-depth query buffers");
            }

            constexpr auto NO_HOVERED_RESULT = std::numeric_limits<unsigned long long>::max();
            if (cudaMemset(hovered_depth_id_device_, 0xFF, sizeof(unsigned long long)) != cudaSuccess) {
                return std::unexpected("Failed to reset hovered-depth query buffer");
            }

            auto render_result = renderRaster(
                splat_data,
                GaussianRasterRequest{
                    .frame_view = request.frame_view,
                    .scaling_modifier = request.scaling_modifier,
                    .mip_filter = request.mip_filter,
                    .sh_degree = request.sh_degree,
                    .gut = request.gut,
                    .equirectangular = request.equirectangular,
                    .scene = request.scene,
                    .filters = request.filters,
                    .overlay =
                        GaussianOverlayState{
                            .cursor =
                                {.enabled = true,
                                 .cursor = request.cursor}},
                    .hovered_depth_id = hovered_depth_id_device_});
            if (!render_result) {
                return std::unexpected(render_result.error());
            }

            if (cudaMemcpy(hovered_depth_id_host_, hovered_depth_id_device_,
                           sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) {
                return std::unexpected("Failed to read back hovered-depth query result");
            }

            const unsigned long long packed = *hovered_depth_id_host_;
            if (packed == NO_HOVERED_RESULT) {
                return std::optional<int>{};
            }
            return std::optional<int>{static_cast<int>(packed & 0xFFFFFFFFu)};
        }

        Result<std::shared_ptr<lfs::core::Tensor>> renderGaussianScreenPositions(
            const lfs::core::SplatData& splat_data,
            const ScreenPositionRenderRequest& request) override {
            Tensor screen_positions;
            auto render_result = renderRaster(
                splat_data,
                GaussianRasterRequest{
                    .frame_view = request.frame_view,
                    .sh_degree = 0,
                    .equirectangular = request.equirectangular,
                    .scene = request.scene,
                    .screen_positions_out = &screen_positions});
            if (!render_result) {
                return std::unexpected(render_result.error());
            }
            if (!screen_positions.is_valid()) {
                return std::unexpected("Screen-position render returned no screen positions");
            }
            return std::make_shared<Tensor>(std::move(screen_positions));
        }

        Result<GpuFrame> renderPointCloudGpuFrame(
            const lfs::core::SplatData& splat_data,
            const PointCloudRenderRequest& request) override {
            auto image_result = renderPointCloudImage(splat_data, request);
            if (!image_result || !image_result->image) {
                return std::unexpected(image_result ? "Point-cloud GPU-frame render returned no image"
                                                    : image_result.error());
            }
            return cacheTensorFrame(image_result->image, image_result->metadata, request.frame_view.size);
        }

        Result<PointCloudImageResult> renderPointCloudImage(
            const lfs::core::SplatData& splat_data,
            const PointCloudRenderRequest& request) override {
            constexpr float SH_C0 = 0.28209479177387814f;
            Tensor colors;
            try {
                colors = (splat_data.sh0_raw().slice(1, 0, 1).squeeze(1) * SH_C0 + 0.5f).clamp(0.0f, 1.0f);
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Failed to derive point colors from SH data: {}", e.what()));
            }

            auto result = renderSoftwarePointCloud(splat_data.get_means(), colors, request);
            if (!result) {
                return std::unexpected(result.error());
            }

            return PointCloudImageResult{
                .image = std::make_shared<Tensor>(std::move(result->image)),
                .metadata = makePointCloudFrameMetadata(*result)};
        }

        Result<PointCloudImageResult> renderPointCloudImage(
            const lfs::core::PointCloud& point_cloud,
            const PointCloudRenderRequest& request) override {
            auto result = renderSoftwarePointCloud(point_cloud.means, point_cloud.colors, request);
            if (!result) {
                return std::unexpected(result.error());
            }

            return PointCloudImageResult{
                .image = std::make_shared<Tensor>(std::move(result->image)),
                .metadata = makePointCloudFrameMetadata(*result)};
        }

        Result<GpuFrame> renderPointCloudGpuFrame(
            const lfs::core::PointCloud& point_cloud,
            const PointCloudRenderRequest& request) override {
            auto image_result = renderPointCloudImage(point_cloud, request);
            if (!image_result || !image_result->image) {
                return std::unexpected(image_result ? "Raw point-cloud GPU-frame render returned no image"
                                                    : image_result.error());
            }
            return cacheTensorFrame(image_result->image, image_result->metadata, request.frame_view.size);
        }

        Result<GpuFrame> materializeGpuFrame(
            const std::shared_ptr<lfs::core::Tensor>& image,
            const FrameMetadata& metadata,
            const glm::ivec2& viewport_size) override {
            if (!image || !image->is_valid()) {
                return std::unexpected("Cannot materialize an empty tensor frame");
            }
            return cacheTensorFrame(image, metadata, viewport_size);
        }

        Result<std::shared_ptr<lfs::core::Tensor>> readbackGpuFrameColor(
            const GpuFrame& frame) override {
            if (!frame.valid() || frame.color.id != cached_tensor_frame_id_ || !cached_tensor_frame_image_) {
                return std::unexpected("Tensor-backed GPU frame is no longer available");
            }
            return cached_tensor_frame_image_;
        }

        Result<lfs::core::Tensor> renderVideoCompositeFrame(
            const std::optional<GpuFrame>& primary_frame,
            const VideoCompositeFrameRequest& request) override {
            std::shared_ptr<lfs::core::Tensor> primary_image;
            const FrameMetadata* primary_metadata = nullptr;
            if (primary_frame) {
                auto image = readbackGpuFrameColor(*primary_frame);
                if (image && *image) {
                    primary_image = *image;
                    primary_metadata = &cached_tensor_frame_metadata_;
                } else {
                    return std::unexpected(image.error());
                }
            }

            auto composite = renderSoftwareVideoComposite(primary_image, primary_metadata, request);
            if (!composite) {
                return std::unexpected(composite.error());
            }
            return std::move(*composite);
        }

        Result<void> renderScreenSpaceVignette(
            const glm::ivec2&,
            ScreenSpaceVignette) override {
            return {};
        }

        Result<void> renderGrid(const ViewportData&, GridPlane, float) override { return {}; }
        Result<void> renderBoundingBox(const BoundingBox&, const ViewportData&, const glm::vec3&, float) override { return {}; }
        Result<void> renderEllipsoid(const Ellipsoid&, const ViewportData&, const glm::vec3&, float) override { return {}; }
        Result<void> renderCoordinateAxes(const ViewportData&, float, const std::array<bool, 3>&, bool) override { return {}; }
        Result<void> renderPivot(const ViewportData&, const glm::vec3&, float, float) override { return {}; }
        Result<void> renderViewportGizmo(const glm::mat3&, const glm::vec2&, const glm::vec2&) override { return {}; }
        int hitTestViewportGizmo(const glm::vec2&, const glm::vec2&, const glm::vec2&) const override { return -1; }
        void setViewportGizmoHover(int) override {}

        Result<void> renderCameraFrustums(
            const std::vector<std::shared_ptr<const lfs::core::Camera>>&,
            const CameraFrustumRenderRequest&) override {
            return {};
        }

        Result<int> pickCameraFrustum(
            const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
            const CameraFrustumPickRequest& request) override {
            if (cameras.empty() || request.viewport_size.x <= 0.0f || request.viewport_size.y <= 0.0f ||
                request.viewport.size.x <= 0 || request.viewport.size.y <= 0) {
                return -1;
            }

            constexpr float HIT_RADIUS_PIXELS = 12.0f;
            int best_uid = -1;
            float best_score = HIT_RADIUS_PIXELS;
            float best_depth = std::numeric_limits<float>::max();
            const glm::vec2 mouse = request.mouse_pos;
            const glm::vec3 viewer_position = request.viewport.translation;

            for (size_t i = 0; i < cameras.size(); ++i) {
                const auto& camera = cameras[i];
                if (!camera) {
                    continue;
                }
                glm::mat4 scene_transform = request.scene_transform;
                if (i < request.scene_transforms.size()) {
                    scene_transform = request.scene_transforms[i];
                }

                const auto transform = cameraVisualizerTransform(*camera, scene_transform);
                if (!transform) {
                    continue;
                }
                const auto points = cameraFrustumWorldPoints(*camera, *transform, request.scale);
                if (points.empty()) {
                    continue;
                }

                float camera_best = HIT_RADIUS_PIXELS;
                if (camera->camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR) {
                    constexpr int SEGMENTS = 48;
                    for (int circle = 0; circle < 3; ++circle) {
                        const int offset = circle * SEGMENTS;
                        if (offset + SEGMENTS > static_cast<int>(points.size())) {
                            break;
                        }
                        for (int segment = 0; segment < SEGMENTS; ++segment) {
                            const auto a = projectFrustumPoint(points[offset + segment], request);
                            const auto b = projectFrustumPoint(points[offset + ((segment + 1) % SEGMENTS)], request);
                            if (!a || !b) {
                                continue;
                            }
                            camera_best = std::min(camera_best, pointSegmentDistance(mouse, *a, *b));
                        }
                    }
                } else if (points.size() >= 5) {
                    constexpr std::array<std::pair<int, int>, 8> EDGES{{
                        {0, 1}, {0, 2}, {0, 3}, {0, 4},
                        {1, 2}, {2, 3}, {3, 4}, {4, 1},
                    }};
                    for (const auto& [a_index, b_index] : EDGES) {
                        const auto a = projectFrustumPoint(points[static_cast<size_t>(a_index)], request);
                        const auto b = projectFrustumPoint(points[static_cast<size_t>(b_index)], request);
                        if (!a || !b) {
                            continue;
                        }
                        camera_best = std::min(camera_best, pointSegmentDistance(mouse, *a, *b));
                    }
                }

                const glm::vec3 camera_position = glm::vec3((*transform)[3]);
                const float depth = glm::length(camera_position - viewer_position);
                if (camera_best < best_score ||
                    (std::abs(camera_best - best_score) <= 1e-3f && depth < best_depth)) {
                    best_score = camera_best;
                    best_depth = depth;
                    best_uid = camera->uid();
                }
            }

            return best_uid;
        }

        void clearFrustumCache() override {}
        void setFrustumImageLoader(std::shared_ptr<lfs::io::PipelinedImageLoader>, bool) override {}

    private:
        GpuFrame cacheTensorFrame(std::shared_ptr<lfs::core::Tensor> image,
                                  const FrameMetadata& metadata,
                                  const glm::ivec2& viewport_size) {
            cached_tensor_frame_image_ = std::move(image);
            cached_tensor_frame_metadata_ = metadata;
            cached_tensor_frame_id_ = next_tensor_frame_id_++;
            if (cached_tensor_frame_id_ == 0) {
                cached_tensor_frame_id_ = next_tensor_frame_id_++;
            }

            TextureHandle depth_handle{};
            if (metadata.primaryDepth() && metadata.primaryDepth()->is_valid()) {
                depth_handle = {
                    .id = cached_tensor_frame_id_,
                    .size = viewport_size,
                    .texcoord_scale = metadata.depth_texcoord_scale};
            }

            return GpuFrame{
                .color =
                    {.id = cached_tensor_frame_id_,
                     .size = viewport_size,
                     .texcoord_scale = glm::vec2(1.0f)},
                .depth = depth_handle,
                .flip_y = metadata.flip_y,
                .depth_is_ndc = metadata.depth_is_ndc,
                .color_has_alpha = metadata.color_has_alpha,
                .near_plane = metadata.near_plane,
                .far_plane = metadata.far_plane,
                .orthographic = metadata.orthographic};
        }

        Result<RasterImageResult> renderRaster(
            const lfs::core::SplatData& splat_data,
            GaussianRasterRequest request) {
            if (!isRasterInitialized()) {
                return std::unexpected("Rendering raster pipeline is not initialized");
            }
            if (request.frame_view.size.x <= 0 || request.frame_view.size.y <= 0 ||
                request.frame_view.size.x > MAX_VIEWPORT_SIZE ||
                request.frame_view.size.y > MAX_VIEWPORT_SIZE) {
                return std::unexpected("Invalid viewport dimensions");
            }

            if (background_.is_valid()) {
                if (auto* bg_data = background_.ptr<float>();
                    bg_data && background_.device() == lfs::core::Device::CUDA) {
                    const float bg_values[3] = {
                        request.frame_view.background_color.r,
                        request.frame_view.background_color.g,
                        request.frame_view.background_color.b};
                    cudaMemcpy(bg_data, bg_values, sizeof(bg_values), cudaMemcpyHostToDevice);
                }
            }

            auto camera = createRasterCamera(request.frame_view, request.gut, request.equirectangular);
            if (!camera) {
                return std::unexpected(camera.error());
            }

            const size_t gaussian_count = static_cast<size_t>(splat_data.size());
            const int effective_sh_degree = std::clamp(request.sh_degree, 0, splat_data.get_max_sh_degree());

            auto model_transforms_tensor =
                makeModelTransformsTensor(request.scene.model_transforms ? *request.scene.model_transforms
                                                                         : std::vector<glm::mat4>{});

            std::unique_ptr<Tensor> transform_indices_cuda;
            Tensor* transform_indices_ptr = cudaTensorPointer(request.scene.transform_indices, transform_indices_cuda);
            if (!tensorMatchesGaussianCount(transform_indices_ptr, gaussian_count)) {
                LOG_WARN("Ignoring transform_indices with stale size: model has {}, tensor has {}",
                         gaussian_count, transform_indices_ptr->numel());
                transform_indices_ptr = nullptr;
                transform_indices_cuda.reset();
            }

            std::unique_ptr<Tensor> selection_mask_cuda;
            Tensor* selection_mask_ptr = cudaTensorPointer(request.overlay.emphasis.mask, selection_mask_cuda);
            if (!tensorMatchesGaussianCount(selection_mask_ptr, gaussian_count)) {
                LOG_WARN("Ignoring selection_mask with stale size: model has {}, tensor has {}",
                         gaussian_count, selection_mask_ptr->numel());
                selection_mask_ptr = nullptr;
                selection_mask_cuda.reset();
            }

            Tensor* preview_selection_ptr = request.overlay.emphasis.transient_mask.mask;
            if (preview_selection_ptr && !preview_selection_ptr->is_valid()) {
                preview_selection_ptr = nullptr;
            }
            if (!tensorMatchesGaussianCount(preview_selection_ptr, gaussian_count)) {
                LOG_WARN("Ignoring preview_selection_tensor with stale size: model has {}, tensor has {}",
                         gaussian_count, preview_selection_ptr->numel());
                preview_selection_ptr = nullptr;
            }

            GaussianRasterResources resources;
            applyCropBoxToRaster(request, resources);
            applyEllipsoidToRaster(request, resources);
            applyViewVolumeToRaster(request, resources);

            try {
                if (request.gut || request.equirectangular) {
                    const auto camera_model = request.equirectangular
                                                  ? GutCameraModel::EQUIRECTANGULAR
                                                  : GutCameraModel::PINHOLE;
                    auto render_output = gut_rasterize_tensor(
                        *camera,
                        splat_data,
                        background_,
                        effective_sh_degree,
                        request.scaling_modifier,
                        camera_model,
                        model_transforms_tensor.get(),
                        transform_indices_ptr,
                        request.scene.node_visibility_mask,
                        request.transparent_background);
                    return RasterImageResult{
                        .image = std::move(render_output.image),
                        .depth = std::move(render_output.depth),
                        .valid = true,
                        .flip_y = true,
                        .far_plane = request.frame_view.far_plane,
                        .orthographic = request.frame_view.orthographic,
                        .color_has_alpha = request.transparent_background};
                }

                auto [image, depth] = rasterize_tensor(
                    *camera,
                    splat_data,
                    background_,
                    effective_sh_degree,
                    request.overlay.markers.show_rings,
                    request.overlay.markers.ring_width,
                    model_transforms_tensor.get(),
                    transform_indices_ptr,
                    selection_mask_ptr,
                    request.screen_positions_out,
                    request.overlay.cursor.enabled,
                    request.overlay.cursor.cursor.x,
                    request.overlay.cursor.cursor.y,
                    request.overlay.cursor.radius,
                    request.overlay.emphasis.transient_mask.additive,
                    preview_selection_ptr,
                    request.overlay.cursor.saturation_preview,
                    request.overlay.cursor.saturation_amount,
                    request.overlay.markers.show_center_markers,
                    resources.crop_box_transform_tensor.is_valid() ? &resources.crop_box_transform_tensor : nullptr,
                    resources.crop_box_min_tensor.is_valid() ? &resources.crop_box_min_tensor : nullptr,
                    resources.crop_box_max_tensor.is_valid() ? &resources.crop_box_max_tensor : nullptr,
                    request.filters.crop_region ? request.filters.crop_region->inverse : false,
                    request.filters.crop_region ? request.filters.crop_region->desaturate : false,
                    request.filters.crop_region ? request.filters.crop_region->parent_node_index : -1,
                    resources.ellipsoid_transform_tensor.is_valid() ? &resources.ellipsoid_transform_tensor : nullptr,
                    resources.ellipsoid_radii_tensor.is_valid() ? &resources.ellipsoid_radii_tensor : nullptr,
                    request.filters.ellipsoid_region ? request.filters.ellipsoid_region->inverse : false,
                    request.filters.ellipsoid_region ? request.filters.ellipsoid_region->desaturate : false,
                    request.filters.ellipsoid_region ? request.filters.ellipsoid_region->parent_node_index : -1,
                    resources.view_volume_transform_tensor.is_valid() ? &resources.view_volume_transform_tensor : nullptr,
                    resources.view_volume_min_tensor.is_valid() ? &resources.view_volume_min_tensor : nullptr,
                    resources.view_volume_max_tensor.is_valid() ? &resources.view_volume_max_tensor : nullptr,
                    request.filters.cull_outside_view_volume,
                    nullptr,
                    request.hovered_depth_id,
                    request.overlay.emphasis.focused_gaussian_id,
                    request.frame_view.far_plane,
                    request.overlay.emphasis.emphasized_node_mask,
                    request.overlay.emphasis.dim_non_emphasized,
                    request.scene.node_visibility_mask,
                    request.overlay.emphasis.flash_intensity,
                    request.frame_view.orthographic,
                    request.frame_view.ortho_scale,
                    request.mip_filter,
                    request.transparent_background);

                return RasterImageResult{
                    .image = std::move(image),
                    .depth = std::move(depth),
                    .valid = true,
                    .far_plane = request.frame_view.far_plane,
                    .orthographic = request.frame_view.orthographic,
                    .color_has_alpha = request.transparent_background};
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Rasterization failed: {}", e.what()));
            }
        }

        [[nodiscard]] bool ensureHoveredDepthQueryBuffersAllocated() {
            if (!hovered_depth_id_device_ &&
                cudaMalloc(&hovered_depth_id_device_, sizeof(unsigned long long)) != cudaSuccess) {
                hovered_depth_id_device_ = nullptr;
                return false;
            }
            if (!hovered_depth_id_host_ &&
                cudaMallocHost(&hovered_depth_id_host_, sizeof(unsigned long long)) != cudaSuccess) {
                if (hovered_depth_id_device_) {
                    cudaFree(hovered_depth_id_device_);
                    hovered_depth_id_device_ = nullptr;
                }
                hovered_depth_id_host_ = nullptr;
                return false;
            }
            return true;
        }

        Tensor background_;
        bool raster_initialized_ = false;
        unsigned long long* hovered_depth_id_device_ = nullptr;
        unsigned long long* hovered_depth_id_host_ = nullptr;
        unsigned int next_tensor_frame_id_ = 1;
        unsigned int cached_tensor_frame_id_ = 0;
        std::shared_ptr<lfs::core::Tensor> cached_tensor_frame_image_;
        FrameMetadata cached_tensor_frame_metadata_{};
    };

    std::unique_ptr<RenderingEngine> RenderingEngine::create() {
        LOG_DEBUG("Creating default raster-only RenderingEngine instance");
        return createRasterOnly();
    }

    std::unique_ptr<RenderingEngine> RenderingEngine::createRasterOnly() {
        LOG_DEBUG("Creating raster-only RenderingEngine instance");
        return std::make_unique<RasterOnlyRenderingEngine>();
    }

    glm::mat3 RenderingEngine::getAxisViewRotation(const int axis, const bool negative) {
        return makeAxisViewRotation(axis, negative);
    }

} // namespace lfs::rendering
