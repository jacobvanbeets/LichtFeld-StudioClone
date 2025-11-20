/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training_setup.hpp"
#include "core_new/logger.hpp"
#include "core_new/point_cloud.hpp"
#include "loader_new/loader.hpp"
#include "strategies/default_strategy.hpp"
#include "strategies/mcmc.hpp"
#include <format>

namespace lfs::training {
    std::expected<TrainingSetup, std::string> setupTraining(const lfs::core::param::TrainingParameters& params) {
        // 1. Create loader
        auto data_loader = lfs::loader::Loader::create();

        // 2. Set up load options
        lfs::loader::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .max_width = params.dataset.max_width,
            .images_folder = params.dataset.images,
            .validate_only = false,
            .progress = [](float percentage, const std::string& message) {
                LOG_DEBUG("[{:5.1f}%] {}", percentage, message);
            }};

        // 3. Load the dataset
        LOG_INFO("Loading dataset from: {}", params.dataset.data_path.string());
        auto load_result = data_loader->load(params.dataset.data_path, load_options);
        if (!load_result) {
            return std::unexpected(std::format("Failed to load dataset: {}", load_result.error()));
        }

        LOG_INFO("Dataset loaded successfully using {} loader", load_result->loader_used);

        // 4. Handle the loaded data based on type
        return std::visit([&](auto&& data) -> std::expected<TrainingSetup, std::string> {
            using T = std::decay_t<decltype(data)>;

            if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
                // Direct PLY load - not supported for training
                return std::unexpected(
                    "Direct PLY loading is not supported for training. Please use a dataset format (COLMAP or Blender).");
            } else if constexpr (std::is_same_v<T, lfs::loader::LoadedScene>) {
                // Full scene data - set up training

                // Initialize model directly with point cloud
                std::expected<lfs::core::SplatData, std::string> splat_result;
                int max_cap = params.optimization.max_cap;
                if (params.init_ply.has_value()) {
                    // Using PLY for initialization - free the COLMAP point cloud immediately
                    if (data.point_cloud && data.point_cloud->size() > 0) {
                        LOG_INFO("Freeing unused COLMAP point cloud ({} points) since --init-ply is used",
                                 data.point_cloud->size());
                        data.point_cloud.reset();  // Free the shared_ptr and release memory
                    }

                    auto ply_loader = lfs::loader::Loader::create();
                    auto ply_load_result = ply_loader->load(params.init_ply.value());

                    if (!ply_load_result) {
                        splat_result = std::unexpected(std::format(
                            "Failed to load initialization PLY file '{}': {}",
                            params.init_ply.value(),
                            ply_load_result.error()));
                    } else {
                        try {
                            splat_result = std::move(*std::get<std::shared_ptr<lfs::core::SplatData>>(ply_load_result->data));
                        } catch (const std::bad_variant_access&) {
                            splat_result = std::unexpected(std::format(
                                "Initialization PLY file '{}' did not contain valid SplatData",
                                params.init_ply.value()));
                        }
                    }

                } else {
                    // Get point cloud or generate random one
                    lfs::core::PointCloud point_cloud_to_use;
                    if (data.point_cloud && data.point_cloud->size() > 0) {
                        if (max_cap > 0) {
                            // Move point cloud to CPU to avoid pool allocations (init will be on CPU)
                            LOG_INFO("Moving point cloud to CPU to avoid pool allocations ({} points)", data.point_cloud->size());
                            auto& means_tensor = data.point_cloud->means;
                            auto& colors_tensor = data.point_cloud->colors;
                            void* means_ptr = means_tensor.template ptr<float>();
                            void* colors_ptr = colors_tensor.template ptr<uint8_t>();
                            LOG_DEBUG("  Original point cloud: means.device={}, means.is_valid={}, means.shape={}, means.ptr={}",
                                      means_tensor.device() == lfs::core::Device::CUDA ? "CUDA" : "CPU",
                                      means_tensor.is_valid(),
                                      means_tensor.shape().str(),
                                      means_ptr);
                            LOG_DEBUG("  Original point cloud: colors.device={}, colors.is_valid={}, colors.shape={}, colors.ptr={}",
                                      colors_tensor.device() == lfs::core::Device::CUDA ? "CUDA" : "CPU",
                                      colors_tensor.is_valid(),
                                      colors_tensor.shape().str(),
                                      colors_ptr);

                            point_cloud_to_use = *data.point_cloud;
                            auto& pc_means = point_cloud_to_use.means;
                            void* pc_means_ptr = pc_means.template ptr<float>();
                            LOG_DEBUG("  After copy: point_cloud_to_use.means.device={}, is_valid={}, ptr={}",
                                      pc_means.device() == lfs::core::Device::CUDA ? "CUDA" : "CPU",
                                      pc_means.is_valid(),
                                      pc_means_ptr);

                            point_cloud_to_use.means = point_cloud_to_use.means.cpu();
                            const void* pc_means_cpu_ptr = pc_means.template ptr<float>();
                            LOG_DEBUG("  After .cpu() on means: device={}, is_valid={}, ptr={}, shape={}",
                                      pc_means.device() == lfs::core::Device::CUDA ? "CUDA" : "CPU",
                                      pc_means.is_valid(),
                                      pc_means_cpu_ptr,
                                      pc_means.shape().str());

                            point_cloud_to_use.colors = point_cloud_to_use.colors.cpu();
                            auto& pc_colors = point_cloud_to_use.colors;
                            const void* pc_colors_cpu_ptr = pc_colors.template ptr<uint8_t>();
                            LOG_DEBUG("  After .cpu() on colors: device={}, is_valid={}, ptr={}, shape={}",
                                      pc_colors.device() == lfs::core::Device::CUDA ? "CUDA" : "CPU",
                                      pc_colors.is_valid(),
                                      pc_colors_cpu_ptr,
                                      pc_colors.shape().str());

                            // Free the original CUDA point cloud to eliminate pool allocations
                            data.point_cloud->means = lfs::core::Tensor();  // Clear CUDA tensor
                            data.point_cloud->colors = lfs::core::Tensor(); // Clear CUDA tensor
                            LOG_DEBUG("Cleared original CUDA point cloud from pool");
                        } else {
                            point_cloud_to_use = *data.point_cloud;
                            LOG_INFO("Using point cloud with {} points", point_cloud_to_use.size());
                        }
                    } else {
                        // Generate random point cloud if needed
                        LOG_INFO("No point cloud provided, using random initialization");
                        size_t numInitGaussian = 10000;
                        uint64_t seed = 8128;

                        if (max_cap > 0) {
                            // Generate on CPU to avoid pool allocations
                            auto positions = lfs::core::Tensor::rand({numInitGaussian, 3}, lfs::core::Device::CPU);
                            positions = positions * 2.0f - 1.0f;
                            auto colors = lfs::core::Tensor::randint({numInitGaussian, 3}, 0, 256, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                            point_cloud_to_use = lfs::core::PointCloud(positions, colors);
                        } else {
                            // Generate on CUDA (uses pool)
                            auto positions = lfs::core::Tensor::rand({numInitGaussian, 3}, lfs::core::Device::CUDA);
                            positions = positions * 2.0f - 1.0f;
                            auto colors = lfs::core::Tensor::randint({numInitGaussian, 3}, 0, 256, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                            point_cloud_to_use = lfs::core::PointCloud(positions, colors);
                        }
                    }
                    // Move scene_center to CPU if using capacity (to avoid pool allocations)
                    auto scene_center = max_cap > 0 ? load_result->scene_center.cpu() : load_result->scene_center;
                    if (max_cap > 0) {
                        load_result->scene_center = lfs::core::Tensor(); // Clear original CUDA tensor
                    }

                    splat_result = lfs::core::SplatData::init_model_from_pointcloud(
                        params,
                        scene_center,
                        point_cloud_to_use,
                        max_cap);  // Pass capacity to use zeros_direct() and avoid pool
                }

                if (!splat_result) {
                    return std::unexpected(
                        std::format("Failed to initialize model: {}", splat_result.error()));
                }
                if (max_cap < splat_result->size()) {
                    LOG_WARN("Max cap is less than to {} initial splats {}. Choosing randomly {} splats", max_cap, splat_result->size(), max_cap);
                    splat_result->random_choose(max_cap);
                }

                // 5. Create strategy
                std::unique_ptr<IStrategy> strategy;
                if (params.optimization.strategy == "mcmc") {
                    strategy = std::make_unique<MCMC>(std::move(*splat_result));
                    LOG_DEBUG("Created MCMC strategy");
                } else {
                    strategy = std::make_unique<DefaultStrategy>(std::move(*splat_result));
                    LOG_DEBUG("Created default strategy");
                }

                // Create trainer (without parameters)
                // Note: provided_splits not available in new loader, pass std::nullopt
                auto trainer = std::make_unique<Trainer>(
                    data.cameras,
                    std::move(strategy), std::nullopt);

                return TrainingSetup{
                    .trainer = std::move(trainer),
                    .dataset = data.cameras,
                    .scene_center = load_result->scene_center};
            } else {
                return std::unexpected("Unknown data type returned from loader");
            }
        },
                          load_result->data);
    }
} // namespace lfs::training
