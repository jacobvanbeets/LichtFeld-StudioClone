/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vksplat_rasterizer.hpp"

#include "core/logger.hpp"
#include "kernels/vksplat_interop.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <format>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(LFS_HAS_VKSPLAT)
#include "gs_trainer.h"
#endif

namespace lfs::training {

    namespace {
        [[nodiscard]] bool has_background_image(const core::Tensor& bg_image) {
            return bg_image.is_valid() && !bg_image.is_empty();
        }

#if defined(LFS_HAS_VKSPLAT)
        TrainerConfig make_vksplat_config(const core::param::OptimizationParameters& params,
                                          const core::SplatData& model) {
            TrainerConfig config{};
            config.strategy = params.strategy == std::string(core::param::kStrategyMCMC)
                                  ? TrainerConfig::Strategy::MCMC
                                  : TrainerConfig::Strategy::Default;
            config.max_steps = static_cast<int>(params.iterations);
            config.ssim_lambda = params.lambda_dssim;
            config.means_lr = params.means_lr;
            config.means_lr_final = params.means_lr_end;
            config.features_dc_lr = params.shs_lr;
            config.features_rest_lr = params.shs_lr / 20.0f;
            config.opacities_lr = params.opacity_lr;
            config.scales_lr = params.scaling_lr;
            config.quats_lr = params.rotation_lr;
            config.scale_reg = params.scale_reg;
            config.opacity_reg = params.opacity_reg;

            config.refine_start_iter = static_cast<int>(params.start_refine);
            config.refine_stop_iter = static_cast<int>(params.stop_refine);
            config.refine_every = std::max(1, static_cast<int>(params.refine_every));
            config.prune_opa = params.prune_opacity;
            config.grow_grad2d = params.grad_threshold;
            config.grow_scale3d = params.grow_scale3d;
            config.grow_scale2d = params.grow_scale2d;
            config.prune_scale3d = params.prune_scale3d;
            config.prune_scale2d = params.prune_scale2d;
            config.refine_scale2d_stop_iter = static_cast<int>(params.stop_refine);
            config.reset_every = std::max(1, static_cast<int>(params.reset_every));
            config.stop_reset_at = static_cast<int>(params.stop_refine);
            config.pause_refine_after_reset = static_cast<int>(params.pause_refine_after_reset);

            config.noise_lr = params.means_lr * model.get_scene_scale();
            config.min_opacity = params.min_opacity;
            config.grow_factor = 1.05f;
            config.cap_max = params.max_cap;
            return config;
        }

        std::map<std::string, std::string> make_spirv_paths() {
            static constexpr auto kShaderNames = std::to_array<const char*>({
                "projection_forward",
                "generate_keys",
                "compute_tile_ranges",
                "rasterize_forward",
                "rasterize_backward_0",
                "rasterize_backward_1",
                "rasterize_backward_2",
                "rasterize_backward_3",
                "rasterize_backward_4",
                "cumsum_single_pass",
                "cumsum_block_scan",
                "cumsum_scan_block_sums",
                "cumsum_add_block_offsets",
                "radix_sort/upsweep",
                "radix_sort/spine",
                "radix_sort/downsweep",
                "ssim_forward",
                "ssim_backward",
                "fused_projection_backward_optimizer",
                "sum",
                "where",
                "default_update_state",
                "default_compute_grow_mask",
                "default_duplicate",
                "default_split",
                "default_compute_prune_mask",
                "default_prune",
                "default_prune_mean",
                "default_prune_sh",
                "default_reset_opa",
                "mcmc_inject_noise",
                "mcmc_compute_probs",
                "mcmc_compute_relocation_index_map",
                "mcmc_compute_relocation",
                "mcmc_update_relocation",
                "mcmc_compute_add_index_map",
                "mcmc_compute_add",
                "mcmc_update_add",
                "morton_sort_compute_stats",
                "morton_sort_generate_keys",
                "morton_sort_apply_indices",
                "morton_sort_apply_indices_sh",
                "morton_sort_update_buffer",
                "morton_sort_update_buffer_sh",
            });

            std::map<std::string, std::string> paths;
            const std::filesystem::path shader_dir = std::filesystem::path(LFS_VKSPLAT_SHADER_DIR);
            for (const char* name : kShaderNames) {
                const std::string key{name};
                if (key.find('/') == std::string::npos) {
                    paths[key] = (shader_dir / "generated" / (key + ".spv")).string();
                } else {
                    paths[key] = (shader_dir / (key + ".spv")).string();
                }
            }
            return paths;
        }

        [[nodiscard]] std::vector<float> tensor_to_cpu_vector(core::Tensor tensor, std::string_view label) {
            if (!tensor.is_valid() || tensor.numel() == 0) {
                return {};
            }
            LOG_DEBUG("VkSplat staging {}: shape={}, dtype={}, device={}, contiguous={}, lazy={}, numel={}",
                      std::string(label),
                      tensor.shape().str(),
                      core::dtype_name(tensor.dtype()),
                      core::device_name(tensor.device()),
                      tensor.is_contiguous(),
                      tensor.has_lazy_expr(),
                      tensor.numel());
            const core::Tensor* current = &tensor;
            core::Tensor float_tensor;
            if (tensor.dtype() != core::DataType::Float32) {
                float_tensor = current->to(core::DataType::Float32);
                current = &float_tensor;
            }
            core::Tensor cpu_tensor;
            if (current->device() != core::Device::CPU) {
                cpu_tensor = current->to(core::Device::CPU);
                current = &cpu_tensor;
            }
            core::Tensor contiguous_tensor;
            if (!current->is_contiguous()) {
                contiguous_tensor = current->contiguous();
                current = &contiguous_tensor;
            }
            if (current->dtype() != core::DataType::Float32) {
                throw std::runtime_error(std::format("VkSplat expected float32 tensor for {}", label));
            }
            if (current->device() != core::Device::CPU) {
                throw std::runtime_error(std::format("VkSplat failed to stage {} on CPU", label));
            }
            const float* ptr = current->ptr<float>();
            if (!ptr) {
                throw std::runtime_error(std::format("VkSplat got null CPU pointer for {}", label));
            }
            LOG_DEBUG("VkSplat staged {} on CPU: shape={}, dtype={}, device={}, bytes={}, ptr={}",
                     std::string(label),
                     current->shape().str(),
                     core::dtype_name(current->dtype()),
                     core::device_name(current->device()),
                     current->bytes(),
                     static_cast<const void*>(ptr));
            std::vector<float> result(current->numel());
            std::memcpy(result.data(), ptr, result.size() * sizeof(float));
            return result;
        }

        void copy_distortion(const core::Tensor& src, VulkanGSRendererUniforms& uniforms, int max_count) {
            for (int i = 0; i < 4; ++i) {
                uniforms.dist_coeffs[i] = 0.0f;
            }
            if (!src.is_valid() || src.numel() == 0) {
                return;
            }
            auto values = tensor_to_cpu_vector(src, "camera.distortion");
            const int count = std::min<int>(max_count, static_cast<int>(values.size()));
            for (int i = 0; i < count; ++i) {
                uniforms.dist_coeffs[i] = values[static_cast<size_t>(i)];
            }
        }

        struct NativeVKSplatBackend {
            VulkanGSTrainer trainer;
            VulkanGSPipelineBuffers buffers;
            VulkanGSRendererUniforms uniforms{};
            TrainerConfig config{};
            const core::SplatData* resident_model = nullptr;
            bool initialized = false;
            bool model_uploaded = false;
            bool model_dirty = false;

            void ensure_initialized() {
                if (initialized) {
                    return;
                }
                trainer.initialize(make_spirv_paths(), 0);
                initialized = true;
                LOG_INFO("VkSplat native Vulkan backend initialized");
            }

            void configure_uniforms(const core::Camera& camera,
                                    const core::SplatData& model,
                                    int tile_x_offset,
                                    int tile_y_offset,
                                    int tile_width,
                                    int tile_height) {
                const int full_width = camera.image_width();
                const int full_height = camera.image_height();
                uniforms.image_width = static_cast<uint32_t>((tile_width > 0) ? tile_width : full_width);
                uniforms.image_height = static_cast<uint32_t>((tile_height > 0) ? tile_height : full_height);
                uniforms.grid_width = _CEIL_DIV(uniforms.image_width, TILE_WIDTH);
                uniforms.grid_height = _CEIL_DIV(uniforms.image_height, TILE_HEIGHT);
                uniforms.num_splats = static_cast<uint32_t>(buffers.num_splats);
                uniforms.active_sh = static_cast<uint32_t>(model.get_active_sh_degree());
                uniforms.step = 0;

                const auto [fx, fy, cx, cy] = camera.get_intrinsics();
                uniforms.fx = fx;
                uniforms.fy = fy;
                uniforms.cx = cx - static_cast<float>(tile_x_offset);
                uniforms.cy = cy - static_cast<float>(tile_y_offset);

                switch (camera.camera_model_type()) {
                case core::CameraModelType::PINHOLE:
                    uniforms.camera_model = 0;
                    copy_distortion(camera.radial_distortion(), uniforms, 2);
                    if (camera.tangential_distortion().is_valid() && camera.tangential_distortion().numel() > 0) {
                        const auto tangential = tensor_to_cpu_vector(camera.tangential_distortion(), "camera.tangential_distortion");
                        if (!tangential.empty()) {
                            uniforms.dist_coeffs[2] = tangential[0];
                        }
                        if (tangential.size() > 1) {
                            uniforms.dist_coeffs[3] = tangential[1];
                        }
                    }
                    if ((camera.radial_distortion().is_valid() && camera.radial_distortion().numel() > 0) ||
                        (camera.tangential_distortion().is_valid() && camera.tangential_distortion().numel() > 0)) {
                        uniforms.camera_model = 1;
                    }
                    break;
                case core::CameraModelType::FISHEYE:
                    uniforms.camera_model = 2;
                    copy_distortion(camera.radial_distortion(), uniforms, 4);
                    break;
                default:
                    throw std::runtime_error("VkSplat currently supports pinhole and fisheye training cameras only");
                }

                auto w2c = camera.world_view_transform();
                if (w2c.ndim() == 3 && w2c.shape()[0] == 1) {
                    w2c = w2c.squeeze(0);
                }
                auto w2c_cpu = tensor_to_cpu_vector(w2c, "camera.world_view_transform");
                if (w2c_cpu.size() != 16) {
                    throw std::runtime_error("VkSplat expected a 4x4 world-view transform");
                }
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        uniforms.world_view_transform[4 * row + col] =
                            w2c_cpu[static_cast<size_t>(4 * col + row)];
                    }
                }
            }

            void sync_model_to_vulkan(core::SplatData& model,
                                      const core::param::OptimizationParameters& params) {
                const size_t n = static_cast<size_t>(model.size());
                if (n == 0) {
                    throw std::runtime_error("VkSplat cannot rasterize an empty model");
                }

                config = make_vksplat_config(params, model);
                trainer.set_scene_scale(model.get_scene_scale());
                buffers.num_splats = n;
                buffers.num_indices = 0;
                buffers.is_unsorted_1 = true;

                const auto means = tensor_to_cpu_vector(model.means(), "model.means");
                buffers.xyz_ws.assign(means.begin(), means.end());
                const auto rotations = tensor_to_cpu_vector(model.get_rotation(), "model.rotation");
                buffers.rotations.assign(rotations.begin(), rotations.end());
                const auto scales = tensor_to_cpu_vector(model.get_scaling(), "model.scaling");
                const auto opacities = tensor_to_cpu_vector(model.get_opacity(), "model.opacity");
                buffers.assignScalesOpacs(buffers.scales_opacs, n, scales.data(), opacities.data());

                buffers.sh_coeffs.assign(n * 16 * 3, 0.0f);
                const auto sh0 = tensor_to_cpu_vector(model.sh0(), "model.sh0");
                for (size_t i = 0; i < n; ++i) {
                    const size_t sh0_base = (model.sh0().ndim() == 3) ? i * 3 : i * 3;
                    for (size_t c = 0; c < 3 && sh0_base + c < sh0.size(); ++c) {
                        buffers.sh_coeffs[(i * 16) * 3 + c] = sh0[sh0_base + c];
                    }
                }
                if (model.shN().is_valid() && model.shN().numel() > 0) {
                    const auto shn = tensor_to_cpu_vector(model.shN(), "model.shN");
                    const size_t source_rest = model.shN().shape()[1];
                    const size_t rest = std::min<size_t>(15, source_rest);
                    for (size_t i = 0; i < n; ++i) {
                        for (size_t k = 0; k < rest; ++k) {
                            for (size_t c = 0; c < 3; ++c) {
                                buffers.sh_coeffs[((i * 16) + (k + 1)) * 3 + c] =
                                    shn[(i * source_rest + k) * 3 + c];
                            }
                        }
                    }
                }
                buffers.reorderSH(buffers.sh_coeffs);

                if (config.strategy == TrainerConfig::Strategy::MCMC && config.cap_max > 0) {
                    const size_t cap = std::max(n, static_cast<size_t>(config.cap_max));
                    trainer.resizeDeviceBuffer(buffers.xyz_ws, 3 * cap);
                    trainer.resizeDeviceBuffer(buffers.sh_coeffs, 12 * 4 * cap);
                    trainer.resizeDeviceBuffer(buffers.rotations, 4 * cap);
                    trainer.resizeDeviceBuffer(buffers.scales_opacs, 4 * cap);
                }
                trainer.copyToDevice(buffers.xyz_ws);
                trainer.copyToDevice(buffers.sh_coeffs);
                trainer.copyToDevice(buffers.rotations);
                trainer.copyToDevice(buffers.scales_opacs);
                resident_model = &model;
                model_uploaded = true;
                model_dirty = false;
            }

            void ensure_model_resident(core::SplatData& model,
                                       const core::param::OptimizationParameters& params) {
                const bool needs_upload =
                    !model_uploaded ||
                    resident_model != &model ||
                    buffers.num_splats == 0;
                config = make_vksplat_config(params, model);
                trainer.set_scene_scale(model.get_scene_scale());
                if (needs_upload) {
                    sync_model_to_vulkan(model, params);
                }
            }

            bool sync_model_from_vulkan(core::SplatData& model) {
                if (!model_uploaded || !model_dirty) {
                    return false;
                }
                trainer.copyFromDevice(buffers.xyz_ws);
                trainer.copyFromDevice(buffers.sh_coeffs);
                trainer.copyFromDevice(buffers.rotations);
                trainer.copyFromDevice(buffers.scales_opacs);
                buffers.undoReorderSH(buffers.sh_coeffs, buffers.num_splats);

                const size_t n = buffers.num_splats;
                if (buffers.xyz_ws.size() < 3 * n ||
                    buffers.rotations.size() < 4 * n ||
                    buffers.scales_opacs.size() < 4 * n ||
                    buffers.sh_coeffs.size() < 16 * 3 * n) {
                    throw std::runtime_error("VkSplat copied an incomplete model snapshot from Vulkan");
                }

                std::vector<float> active_means(buffers.xyz_ws.begin(), buffers.xyz_ws.begin() + static_cast<std::ptrdiff_t>(3 * n));
                std::vector<float> active_rotations(buffers.rotations.begin(), buffers.rotations.begin() + static_cast<std::ptrdiff_t>(4 * n));
                buffers.sh_coeffs.resize(16 * 3 * n);

                model.means() = core::Tensor::from_vector(active_means, {n, 3}, core::Device::CUDA);
                model.rotation_raw() = core::Tensor::from_vector(active_rotations, {n, 4}, core::Device::CUDA);

                std::vector<float> raw_scales(n * 3);
                std::vector<float> raw_opacities(n);
                for (size_t i = 0; i < n; ++i) {
                    const float* so = &buffers.scales_opacs[4 * i];
                    raw_scales[3 * i + 0] = std::log(std::max(so[0], 1e-8f));
                    raw_scales[3 * i + 1] = std::log(std::max(so[1], 1e-8f));
                    raw_scales[3 * i + 2] = std::log(std::max(so[2], 1e-8f));
                    const float opacity = std::clamp(so[3], 1e-6f, 1.0f - 1e-6f);
                    raw_opacities[i] = std::log(opacity / (1.0f - opacity));
                }
                model.scaling_raw() = core::Tensor::from_vector(raw_scales, {n, 3}, core::Device::CUDA);
                model.opacity_raw() = core::Tensor::from_vector(raw_opacities, {n, 1}, core::Device::CUDA);

                std::vector<float> sh0(n * 3);
                for (size_t i = 0; i < n; ++i) {
                    for (size_t c = 0; c < 3; ++c) {
                        sh0[i * 3 + c] = buffers.sh_coeffs[(i * 16) * 3 + c];
                    }
                }
                model.sh0() = core::Tensor::from_vector(sh0, {n, 1, 3}, core::Device::CUDA);

                const size_t rest = model.shN().is_valid() && model.shN().ndim() >= 2
                                        ? model.shN().shape()[1]
                                        : 15;
                std::vector<float> shn(n * rest * 3, 0.0f);
                for (size_t i = 0; i < n; ++i) {
                    for (size_t k = 0; k < std::min<size_t>(15, rest); ++k) {
                        for (size_t c = 0; c < 3; ++c) {
                            shn[(i * rest + k) * 3 + c] =
                                buffers.sh_coeffs[((i * 16) + (k + 1)) * 3 + c];
                        }
                    }
                }
                model.shN() = core::Tensor::from_vector(shn, {n, rest, 3}, core::Device::CUDA);
                model._densification_info = core::Tensor::zeros({2, n}, core::Device::CUDA);
                resident_model = &model;
                model_dirty = false;
                return true;
            }

            RenderOutput read_render_output(const core::Tensor& bg_color,
                                            const core::Tensor& bg_image) {
                const size_t width = uniforms.image_width;
                const size_t height = uniforms.image_height;
                const size_t pixels = width * height;

                if (buffers.num_indices > 0) {
                    if (bg_color.device() == core::Device::CUDA &&
                        (!has_background_image(bg_image) || bg_image.device() == core::Device::CUDA)) {
                        RenderOutput output;
                        output.image = core::Tensor::empty({3, height, width}, core::Device::CUDA);
                        output.alpha = core::Tensor::empty({1, height, width}, core::Device::CUDA);
                        output.width = static_cast<int>(width);
                        output.height = static_cast<int>(height);

                        auto* pixel_state = static_cast<const float*>(
                            trainer.getCudaMappedPointer(buffers.pixel_state.deviceBuffer));
                        const float* bg_image_ptr = has_background_image(bg_image)
                                                        ? bg_image.ptr<float>()
                                                        : nullptr;
                        kernels::launch_vksplat_compose_pixel_state(
                            pixel_state,
                            bg_color.ptr<float>(),
                            bg_image_ptr,
                            output.image.ptr<float>(),
                            output.alpha.ptr<float>(),
                            static_cast<int>(height),
                            static_cast<int>(width),
                            output.image.stream());
                        return output;
                    }

                    trainer.copyFromDevice(buffers.pixel_state);
                } else {
                    buffers.pixel_state.assign(4 * pixels, 0.0f);
                    for (size_t i = 0; i < pixels; ++i) {
                        buffers.pixel_state[4 * i + 3] = 1.0f;
                    }
                }

                std::vector<float> image(3 * pixels);
                std::vector<float> alpha(pixels);
                const auto bg = tensor_to_cpu_vector(bg_color, "background.color");
                const auto bg_pixels = has_background_image(bg_image) ? tensor_to_cpu_vector(bg_image, "background.image") : std::vector<float>{};

                for (size_t i = 0; i < pixels; ++i) {
                    const float r = buffers.pixel_state[4 * i + 0];
                    const float g = buffers.pixel_state[4 * i + 1];
                    const float b = buffers.pixel_state[4 * i + 2];
                    const float transmittance = buffers.pixel_state[4 * i + 3];
                    alpha[i] = 1.0f - transmittance;

                    const float bg_r = bg_pixels.empty() ? bg[0] : bg_pixels[0 * pixels + i];
                    const float bg_g = bg_pixels.empty() ? bg[1] : bg_pixels[1 * pixels + i];
                    const float bg_b = bg_pixels.empty() ? bg[2] : bg_pixels[2 * pixels + i];
                    image[0 * pixels + i] = r + transmittance * bg_r;
                    image[1 * pixels + i] = g + transmittance * bg_g;
                    image[2 * pixels + i] = b + transmittance * bg_b;
                }

                RenderOutput output;
                output.image = core::Tensor::from_vector(image, {3, height, width}, core::Device::CUDA);
                output.alpha = core::Tensor::from_vector(alpha, {1, height, width}, core::Device::CUDA);
                output.width = static_cast<int>(width);
                output.height = static_cast<int>(height);
                return output;
            }

            void write_grad_to_vulkan(const core::Tensor& grad_image,
                                      const core::Tensor& bg_color,
                                      const core::Tensor& bg_image,
                                      const core::Tensor& grad_alpha_extra) {
                const size_t width = uniforms.image_width;
                const size_t height = uniforms.image_height;
                const size_t pixels = width * height;
                const bool grad_chw = grad_image.ndim() == 3 && grad_image.shape()[0] == 3;

                if (grad_image.device() == core::Device::CUDA &&
                    bg_color.device() == core::Device::CUDA &&
                    (!has_background_image(bg_image) || bg_image.device() == core::Device::CUDA) &&
                    (!grad_alpha_extra.is_valid() || grad_alpha_extra.numel() == 0 ||
                     grad_alpha_extra.device() == core::Device::CUDA)) {
                    trainer.resizeCudaExportableDeviceBuffer(buffers.v_pixel_state, 4 * pixels);
                    auto* v_pixel_state = static_cast<float*>(
                        trainer.getCudaMappedPointer(buffers.v_pixel_state.deviceBuffer));
                    const float* bg_image_ptr = has_background_image(bg_image)
                                                    ? bg_image.ptr<float>()
                                                    : nullptr;
                    const float* grad_alpha_ptr = grad_alpha_extra.is_valid() && grad_alpha_extra.numel() > 0
                                                      ? grad_alpha_extra.ptr<float>()
                                                      : nullptr;
                    kernels::launch_vksplat_pack_grad_pixel_state(
                        grad_image.ptr<float>(),
                        bg_color.ptr<float>(),
                        bg_image_ptr,
                        grad_alpha_ptr,
                        v_pixel_state,
                        static_cast<int>(height),
                        static_cast<int>(width),
                        grad_chw,
                        grad_alpha_extra.is_valid() ? static_cast<int>(grad_alpha_extra.numel()) : 0,
                        grad_image.stream());
                    const cudaError_t sync_error = cudaStreamSynchronize(grad_image.stream());
                    if (sync_error != cudaSuccess) {
                        throw std::runtime_error(std::format(
                            "VkSplat CUDA gradient interop sync failed: {}",
                            cudaGetErrorString(sync_error)));
                    }
                    return;
                }

                const auto grad = tensor_to_cpu_vector(grad_image, "grad.image");
                const auto bg = tensor_to_cpu_vector(bg_color, "grad.background_color");
                const auto bg_pixels = has_background_image(bg_image) ? tensor_to_cpu_vector(bg_image, "grad.background_image") : std::vector<float>{};
                const auto extra_alpha = grad_alpha_extra.is_valid() && grad_alpha_extra.numel() > 0
                                             ? tensor_to_cpu_vector(grad_alpha_extra, "grad.alpha")
                                             : std::vector<float>{};

                buffers.v_pixel_state.assign(4 * pixels, 0.0f);
                for (size_t i = 0; i < pixels; ++i) {
                    const float dr = grad_chw ? grad[0 * pixels + i] : grad[3 * i + 0];
                    const float dg = grad_chw ? grad[1 * pixels + i] : grad[3 * i + 1];
                    const float db = grad_chw ? grad[2 * pixels + i] : grad[3 * i + 2];
                    const float bg_r = bg_pixels.empty() ? bg[0] : bg_pixels[0 * pixels + i];
                    const float bg_g = bg_pixels.empty() ? bg[1] : bg_pixels[1 * pixels + i];
                    const float bg_b = bg_pixels.empty() ? bg[2] : bg_pixels[2 * pixels + i];
                    float d_transmittance = dr * bg_r + dg * bg_g + db * bg_b;
                    if (!extra_alpha.empty()) {
                        const size_t alpha_idx = extra_alpha.size() == pixels ? i : std::min(i, extra_alpha.size() - 1);
                        d_transmittance -= extra_alpha[alpha_idx];
                    }

                    buffers.v_pixel_state[4 * i + 0] = dr;
                    buffers.v_pixel_state[4 * i + 1] = dg;
                    buffers.v_pixel_state[4 * i + 2] = db;
                    buffers.v_pixel_state[4 * i + 3] = d_transmittance;
                }
                trainer.copyToDevice(buffers.v_pixel_state);
            }
        };

        NativeVKSplatBackend& backend() {
            static NativeVKSplatBackend instance;
            return instance;
        }

        std::mutex& backend_mutex() {
            static std::mutex mutex;
            return mutex;
        }
#endif
    } // namespace

    bool vksplat_backend_available() {
#if defined(LFS_HAS_VKSPLAT)
        return true;
#else
        return false;
#endif
    }

    std::expected<std::pair<RenderOutput, VKSplatRasterizeContext>, std::string> vksplat_rasterize_forward(
        const core::Camera& viewpoint_camera,
        core::SplatData& gaussian_model,
        core::Tensor& bg_color,
        const core::param::OptimizationParameters& params,
        int tile_x_offset,
        int tile_y_offset,
        int tile_width,
        int tile_height,
        const core::Tensor& bg_image) {
#if defined(LFS_HAS_VKSPLAT)
        try {
            std::lock_guard lock(backend_mutex());
            auto& vk = backend();
            vk.ensure_initialized();
            vk.ensure_model_resident(gaussian_model, params);
            vk.configure_uniforms(viewpoint_camera, gaussian_model, tile_x_offset, tile_y_offset, tile_width, tile_height);

            {
                DeviceGuard guard(&vk.trainer);
                vk.trainer.executeProjectionForward(vk.uniforms, vk.buffers);
                vk.trainer.executeCalculateIndexBufferOffset(vk.buffers);
                if (vk.buffers.num_indices > 0) {
                    vk.trainer.executeGenerateKeys(vk.uniforms, vk.buffers);
                    vk.trainer.executeSort(vk.uniforms, vk.buffers, -1);
                    vk.trainer.executeComputeTileRanges(vk.uniforms, vk.buffers);
                    vk.trainer.executeRasterizeForward(vk.uniforms, vk.buffers);
                }
            }

            auto output = vk.read_render_output(bg_color, bg_image);
            VKSplatRasterizeContext ctx;
            ctx.width = output.width;
            ctx.height = output.height;
            ctx.has_visible_splats = vk.buffers.num_indices > 0;
            ctx.bg_color = bg_color;
            ctx.bg_image = bg_image;
            return std::pair{std::move(output), std::move(ctx)};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat forward failed: {}", e.what()));
        }
#else
        (void)viewpoint_camera;
        (void)gaussian_model;
        (void)bg_color;
        (void)params;
        (void)tile_x_offset;
        (void)tile_y_offset;
        (void)tile_width;
        (void)tile_height;
        (void)bg_image;
        return std::unexpected("VkSplat backend was not built. Install the vcpkg vulkan port and reconfigure.");
#endif
    }

    void vksplat_rasterize_backward(
        VKSplatRasterizeContext& ctx,
        const core::Tensor& grad_image,
        core::SplatData& gaussian_model,
        const core::param::OptimizationParameters& params,
        const core::Tensor& grad_alpha_extra,
        int iteration) {
#if defined(LFS_HAS_VKSPLAT)
        if (!ctx.has_visible_splats) {
            return;
        }
        std::lock_guard lock(backend_mutex());
        auto& vk = backend();
        vk.write_grad_to_vulkan(grad_image, ctx.bg_color, ctx.bg_image, grad_alpha_extra);
        {
            DeviceGuard guard(&vk.trainer);
            vk.trainer.executeRasterizeBackward(vk.uniforms, vk.buffers);
            vk.trainer.executeFusedProjectionBackwardOptimizerStep(vk.config, vk.uniforms, vk.buffers, iteration + 1);
            if (vk.config.strategy == TrainerConfig::Strategy::MCMC) {
                vk.trainer.executeMCMCPostBackward(vk.config, vk.uniforms, vk.buffers, iteration);
            } else {
                vk.trainer.executeDefaultPostBackward(vk.config, vk.uniforms, vk.buffers, iteration);
            }
        }
        vk.model_dirty = true;
#else
        (void)ctx;
        (void)grad_image;
        (void)gaussian_model;
        (void)params;
        (void)grad_alpha_extra;
        (void)iteration;
        throw std::runtime_error("VkSplat backend was not built. Install the vcpkg vulkan port and reconfigure.");
#endif
    }

    std::expected<bool, std::string> vksplat_flush_model(core::SplatData& gaussian_model) {
#if defined(LFS_HAS_VKSPLAT)
        try {
            std::lock_guard lock(backend_mutex());
            auto& vk = backend();
            return vk.sync_model_from_vulkan(gaussian_model);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat model flush failed: {}", e.what()));
        }
#else
        (void)gaussian_model;
        return std::unexpected("VkSplat backend was not built. Install the vcpkg vulkan port and reconfigure.");
#endif
    }

    size_t vksplat_resident_splat_count() {
#if defined(LFS_HAS_VKSPLAT)
        std::lock_guard lock(backend_mutex());
        return backend().buffers.num_splats;
#else
        return 0;
#endif
    }

    RenderOutput vksplat_rasterize(
        const core::Camera& viewpoint_camera,
        core::SplatData& gaussian_model,
        core::Tensor& bg_color,
        const core::param::OptimizationParameters& params,
        const core::Tensor& bg_image) {
        auto result = vksplat_rasterize_forward(viewpoint_camera, gaussian_model, bg_color, params, 0, 0, 0, 0, bg_image);
        if (!result) {
            throw std::runtime_error(result.error());
        }
        return std::move(result->first);
    }

} // namespace lfs::training
