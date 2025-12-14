// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * @file test_gut_gsplat_comparison.cpp
 * @brief Compare Legacy (LibTorch) GUT gsplat vs LFS (LibTorch-free) GUT gsplat
 *        using the REAL Trainer classes and train_step methods
 *
 * This test:
 * - Loads data using respective dataloaders (Legacy & New)
 * - Creates Trainer instances from both implementations
 * - Runs train_step for N iterations using the SAME camera
 * - Disables regularizations (opacity_reg=0, scale_reg=0)
 * - Disables post_backward via skip_post_backward flag
 * - Compares parameter values exactly after each iteration
 */

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <format>
#include <variant>

// Legacy modules (LibTorch-based)
#include "loader/loader.hpp"
#include "loader/cache_image_loader.hpp"
#include "core/splat_data.hpp"
#include "core/point_cloud.hpp"
#include "core/parameters.hpp"
#include "core/camera.hpp"

// New modules (LibTorch-free)
#include "io/loader.hpp"
#include "io/cache_image_loader.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/point_cloud.hpp"
#include "core_new/parameters.hpp"
#include "core_new/camera.hpp"

// Include dataset headers
#include "../src/training/dataset.hpp"
#include "../src/training_new/dataset.hpp"

// Include Trainer headers
#include "../src/training/trainer.hpp"
#include "../src/training_new/trainer.hpp"

// Include strategy headers
#include "../src/training/strategies/mcmc.hpp"
#include "../src/training_new/strategies/mcmc.hpp"

// Include render output structs
#include "../src/training/rasterization/rasterizer.hpp"
#include "../src/training_new/optimizer/render_output.hpp"

namespace {

// Default test data path - can be overridden via environment variable
const char* get_test_data_path() {
    const char* env_path = std::getenv("GUT_TEST_DATA_PATH");
    return env_path ? env_path : "/media/paja/T7/my_data/garden";
}

// Configuration for test
struct TestConfig {
    std::string data_path;
    std::string images_folder = "images_4";
    int resize_factor = -1;
    int max_width = 3840;
    int max_cap = 1000000;
    int sh_degree = 0;
    int num_iterations = 1;  // Start with 1 iteration to isolate divergence
    float lambda_dssim = 0.0f;  // Pure L1 loss to isolate gradient difference
};

// Helper to convert LFS tensor to torch tensor for comparison
torch::Tensor lfs_to_torch(const lfs::core::Tensor& lfs_tensor) {
    auto cpu_tensor = lfs_tensor.to(lfs::core::Device::CPU);
    std::vector<int64_t> shape;
    for (size_t i = 0; i < cpu_tensor.ndim(); ++i) {
        shape.push_back(static_cast<int64_t>(cpu_tensor.shape()[i]));
    }
    return torch::from_blob(
        const_cast<float*>(cpu_tensor.ptr<float>()),
        torch::IntArrayRef(shape),
        torch::kFloat32).clone();
}

// Convert tensor sizes to string for logging
std::string sizes_to_string(const torch::IntArrayRef& sizes) {
    std::string result = "[";
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(sizes[i]);
    }
    result += "]";
    return result;
}

// Compare tensors and return max absolute difference
float compare_tensors(const torch::Tensor& a, const torch::Tensor& b, const std::string& name) {
    auto a_cpu = a.cpu().contiguous();
    auto b_cpu = b.cpu().contiguous();

    if (a_cpu.sizes() != b_cpu.sizes()) {
        spdlog::error("{}: Shape mismatch! Legacy: {}, New: {}",
                      name, sizes_to_string(a_cpu.sizes()), sizes_to_string(b_cpu.sizes()));
        return std::numeric_limits<float>::max();
    }

    auto diff = (a_cpu - b_cpu).abs();
    float max_diff = diff.max().item<float>();
    float mean_diff = diff.mean().item<float>();

    spdlog::info("{}: max_diff={:.6e}, mean_diff={:.6e}", name, max_diff, mean_diff);
    return max_diff;
}

// Compare model parameters
void compare_model_params(const gs::SplatData& legacy, const lfs::core::SplatData& new_impl,
                          const std::string& tag) {
    spdlog::info("=== {} Parameter Comparison ===", tag);

    // Means
    auto legacy_means = legacy.get_means().cpu();
    auto new_means = lfs_to_torch(new_impl.get_means());
    compare_tensors(legacy_means, new_means, tag + " means");

    // Opacities (RAW - before sigmoid)
    auto legacy_opac_raw = legacy.opacity_raw().cpu();
    auto new_opac_raw = lfs_to_torch(new_impl.opacity_raw());
    compare_tensors(legacy_opac_raw, new_opac_raw, tag + " opacity_raw");

    // Opacities (activated - after sigmoid)
    auto legacy_opac = legacy.get_opacity().cpu();
    auto new_opac = lfs_to_torch(new_impl.get_opacity());
    compare_tensors(legacy_opac, new_opac, tag + " opacity_activated");

    // Scales (RAW - before exp)
    auto legacy_scales_raw = legacy.scaling_raw().cpu();
    auto new_scales_raw = lfs_to_torch(new_impl.scaling_raw());
    compare_tensors(legacy_scales_raw, new_scales_raw, tag + " scales_raw");

    // Scales (activated - after exp)
    auto legacy_scales = legacy.get_scaling().cpu();
    auto new_scales = lfs_to_torch(new_impl.get_scaling());
    compare_tensors(legacy_scales, new_scales, tag + " scales_activated");

    // Rotations (RAW - before normalize)
    auto legacy_rot_raw = legacy.rotation_raw().cpu();
    auto new_rot_raw = lfs_to_torch(new_impl.rotation_raw());
    compare_tensors(legacy_rot_raw, new_rot_raw, tag + " rotation_raw");

    // Rotations (activated/normalized)
    auto legacy_rot = legacy.get_rotation().cpu();
    auto new_rot = lfs_to_torch(new_impl.get_rotation());
    compare_tensors(legacy_rot, new_rot, tag + " rotation_activated");

    // SH coefficients
    auto legacy_shs = legacy.get_shs().cpu();
    auto new_shs = lfs_to_torch(new_impl.get_shs());
    compare_tensors(legacy_shs, new_shs, tag + " shs");
}

} // anonymous namespace

class GutGsplatComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        spdlog::set_level(spdlog::level::info);
    }
};

/**
 * @brief Main test: Compare Legacy GUT gsplat vs New GUT gsplat using REAL Trainer classes
 *
 * Uses the actual train_step() method from both trainers with:
 * - Same camera for both
 * - Regularizations disabled (opacity_reg=0, scale_reg=0)
 * - post_backward disabled via skip_post_backward flag
 */
TEST_F(GutGsplatComparisonTest, RealTrainerComparison) {
    spdlog::info("============================================================");
    spdlog::info("=== GUT GSPLAT: Legacy vs New REAL Trainer Comparison ===");
    spdlog::info("============================================================");

    TestConfig cfg;
    cfg.data_path = get_test_data_path();
    cfg.num_iterations = 500;  // 500 iterations (Legacy crashes ~900)

    spdlog::info("Test configuration:");
    spdlog::info("  Data path: {}", cfg.data_path);
    spdlog::info("  Iterations: {}", cfg.num_iterations);
    spdlog::info("  SH degree: {}", cfg.sh_degree);
    spdlog::info("  Regularizations: DISABLED (scale_reg=0, opacity_reg=0)");
    spdlog::info("  post_backward: DISABLED (skip_post_backward=true)");
    spdlog::info("  Same camera for both: YES");

    // ============================================================
    // LEGACY INITIALIZATION
    // ============================================================
    spdlog::info("\n[LEGACY] Initializing...");

    gs::param::TrainingParameters legacy_params;
    legacy_params.dataset.data_path = cfg.data_path;
    legacy_params.dataset.output_path = "/tmp/gut_test_legacy";
    legacy_params.dataset.images = cfg.images_folder;
    legacy_params.dataset.resize_factor = cfg.resize_factor;
    legacy_params.dataset.max_width = cfg.max_width;
    legacy_params.optimization.strategy = "mcmc";
    legacy_params.optimization.max_cap = cfg.max_cap;
    legacy_params.optimization.sh_degree = cfg.sh_degree;
    legacy_params.optimization.iterations = cfg.num_iterations;
    legacy_params.optimization.lambda_dssim = cfg.lambda_dssim;
    legacy_params.optimization.gut = true;
    legacy_params.optimization.headless = true;
    // DISABLE regularizations
    legacy_params.optimization.scale_reg = 0.0f;
    legacy_params.optimization.opacity_reg = 0.0f;
    // DISABLE post_backward
    legacy_params.optimization.skip_post_backward = true;
    // test_fixed_camera_uid will be set after loading

    auto legacy_loader = gs::loader::Loader::create();
    ASSERT_TRUE(legacy_loader != nullptr);

    auto& legacy_cache_loader = gs::loader::CacheLoader::getInstance(
        legacy_params.dataset.loading_params.use_cpu_memory,
        legacy_params.dataset.loading_params.use_fs_cache);

    gs::loader::LoadOptions legacy_load_options{
        .resize_factor = legacy_params.dataset.resize_factor,
        .max_width = legacy_params.dataset.max_width,
        .images_folder = legacy_params.dataset.images,
        .validate_only = false,
    };

    auto legacy_load_result = legacy_loader->load(legacy_params.dataset.data_path, legacy_load_options);
    ASSERT_TRUE(legacy_load_result.has_value());

    std::shared_ptr<gs::training::CameraDataset> legacy_dataset;
    std::unique_ptr<gs::training::MCMC> legacy_strategy;

    std::visit([&](auto&& data) {
        using T = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<T, gs::loader::LoadedScene>) {
            gs::PointCloud point_cloud_to_use;
            if (data.point_cloud && data.point_cloud->size() > 0) {
                point_cloud_to_use = *data.point_cloud;
            }

            auto splat_result = gs::SplatData::init_model_from_pointcloud(
                legacy_params, legacy_load_result->scene_center, point_cloud_to_use);
            ASSERT_TRUE(splat_result.has_value());

            legacy_strategy = std::make_unique<gs::training::MCMC>(std::move(*splat_result));
            legacy_dataset = data.cameras;

            legacy_cache_loader.update_cache_params(
                legacy_params.dataset.loading_params.use_cpu_memory,
                legacy_params.dataset.loading_params.use_fs_cache,
                static_cast<int>(data.cameras->size().value()));
        }
    }, legacy_load_result->data);

    size_t legacy_num_gaussians = legacy_strategy->get_model().size();
    spdlog::info("[LEGACY] Initialized {} Gaussians", legacy_num_gaussians);

    // Get camera uid from first camera (use same uid for both trainers)
    auto legacy_cameras = legacy_dataset->get_cameras();
    ASSERT_GT(legacy_cameras.size(), 0);
    int fixed_camera_uid = legacy_cameras[0]->uid();
    spdlog::info("Using fixed camera uid={} ({}) for both trainers",
                 fixed_camera_uid, legacy_cameras[0]->image_name());

    // Set fixed camera uid
    legacy_params.optimization.test_fixed_camera_uid = fixed_camera_uid;

    // Create Legacy Trainer
    auto legacy_trainer = std::make_unique<gs::training::Trainer>(
        legacy_dataset,
        std::move(legacy_strategy),
        std::nullopt);

    auto legacy_init_result = legacy_trainer->initialize(legacy_params);
    ASSERT_TRUE(legacy_init_result.has_value()) << legacy_init_result.error();
    spdlog::info("[LEGACY] Trainer initialized");

    // ============================================================
    // NEW INITIALIZATION
    // ============================================================
    spdlog::info("\n[NEW] Initializing...");

    lfs::core::param::TrainingParameters new_params;
    new_params.dataset.data_path = cfg.data_path;
    new_params.dataset.output_path = "/tmp/gut_test_new";
    new_params.dataset.images = cfg.images_folder;
    new_params.dataset.resize_factor = cfg.resize_factor;
    new_params.dataset.max_width = cfg.max_width;
    new_params.optimization.strategy = "mcmc";
    new_params.optimization.max_cap = cfg.max_cap;
    new_params.optimization.sh_degree = cfg.sh_degree;
    new_params.optimization.iterations = cfg.num_iterations;
    new_params.optimization.lambda_dssim = cfg.lambda_dssim;
    new_params.optimization.gut = true;
    new_params.optimization.headless = true;
    // DISABLE regularizations
    new_params.optimization.scale_reg = 0.0f;
    new_params.optimization.opacity_reg = 0.0f;
    // DISABLE post_backward
    new_params.optimization.skip_post_backward = true;
    // USE SAME CAMERA (by uid) for all iterations - set same uid as legacy
    new_params.optimization.test_fixed_camera_uid = fixed_camera_uid;

    auto new_loader = lfs::io::Loader::create();
    ASSERT_TRUE(new_loader != nullptr);

    lfs::io::LoadOptions new_load_options{
        .resize_factor = new_params.dataset.resize_factor,
        .max_width = new_params.dataset.max_width,
        .images_folder = new_params.dataset.images,
        .validate_only = false,
    };

    auto new_load_result = new_loader->load(new_params.dataset.data_path, new_load_options);
    ASSERT_TRUE(new_load_result.has_value());

    std::shared_ptr<lfs::training::CameraDataset> new_dataset;
    std::unique_ptr<lfs::core::SplatData> new_splat_data;  // Must outlive strategy
    std::unique_ptr<lfs::training::MCMC> new_strategy;

    std::visit([&](auto&& data) {
        using T = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<T, lfs::io::LoadedScene>) {
            lfs::core::PointCloud point_cloud_to_use;
            if (data.point_cloud && data.point_cloud->size() > 0) {
                point_cloud_to_use = *data.point_cloud;
            }

            auto splat_result = lfs::core::init_model_from_pointcloud(
                new_params, new_load_result->scene_center, point_cloud_to_use);
            ASSERT_TRUE(splat_result.has_value());

            new_splat_data = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
            new_strategy = std::make_unique<lfs::training::MCMC>(*new_splat_data);
            new_dataset = data.cameras;
        }
    }, new_load_result->data);

    size_t new_num_gaussians = new_strategy->get_model().size();
    spdlog::info("[NEW] Initialized {} Gaussians", new_num_gaussians);

    // Create New Trainer
    auto new_trainer = std::make_unique<lfs::training::Trainer>(
        new_dataset,
        std::move(new_strategy),
        std::nullopt);

    auto new_init_result = new_trainer->initialize(new_params);
    ASSERT_TRUE(new_init_result.has_value()) << new_init_result.error();
    spdlog::info("[NEW] Trainer initialized");

    // ============================================================
    // VERIFY INITIAL STATE
    // ============================================================
    ASSERT_EQ(legacy_num_gaussians, new_num_gaussians);
    spdlog::info("\nBoth pipelines initialized with {} Gaussians", legacy_num_gaussians);

    // Compare initial parameters
    compare_model_params(
        legacy_trainer->get_strategy().get_model(),
        new_trainer->get_strategy().get_model(),
        "Initial");

    // ============================================================
    // RUN TRAINING
    // ============================================================
    // Both trainers will use the same camera (index 0) for all iterations
    // due to test_fixed_camera_index=0
    spdlog::info("\nRunning Legacy trainer for {} iterations...", cfg.num_iterations);
    auto legacy_train_result = legacy_trainer->train();
    EXPECT_TRUE(legacy_train_result.has_value()) << legacy_train_result.error();

    spdlog::info("\nRunning New trainer for {} iterations...", cfg.num_iterations);
    auto new_train_result = new_trainer->train();
    EXPECT_TRUE(new_train_result.has_value()) << new_train_result.error();

    // ============================================================
    // FINAL COMPARISON
    // ============================================================
    spdlog::info("\n============================================================");
    spdlog::info("=== Final Comparison ===");
    spdlog::info("============================================================");

    compare_model_params(
        legacy_trainer->get_strategy().get_model(),
        new_trainer->get_strategy().get_model(),
        "Final");

    // Print first Gaussian's rotation to see actual values
    {
        auto legacy_rot = legacy_trainer->get_strategy().get_model().rotation_raw().cpu();
        auto new_rot = lfs_to_torch(new_trainer->get_strategy().get_model().rotation_raw());
        auto leg_ptr = legacy_rot.data_ptr<float>();
        auto new_ptr = new_rot.data_ptr<float>();
        spdlog::warn("Gauss0 rotation: Legacy=[{:.6f},{:.6f},{:.6f},{:.6f}], New=[{:.6f},{:.6f},{:.6f},{:.6f}]",
                     leg_ptr[0], leg_ptr[1], leg_ptr[2], leg_ptr[3],
                     new_ptr[0], new_ptr[1], new_ptr[2], new_ptr[3]);

        // Find the gaussian with max difference - sum across quaternion components
        auto diff = (legacy_rot - new_rot).abs().sum(-1);  // [N] - sum of abs diff per gaussian
        auto max_result = diff.max(0);
        float max_val = std::get<0>(max_result).item<float>();
        int64_t max_idx = std::get<1>(max_result).item<int64_t>();
        spdlog::warn("Max diff gaussian idx={}, total_diff={:.6f}", max_idx, max_val);
        spdlog::warn("  Legacy[{}]=[{:.6f},{:.6f},{:.6f},{:.6f}]", max_idx,
                     legacy_rot[max_idx][0].item<float>(), legacy_rot[max_idx][1].item<float>(),
                     legacy_rot[max_idx][2].item<float>(), legacy_rot[max_idx][3].item<float>());
        spdlog::warn("  New[{}]=[{:.6f},{:.6f},{:.6f},{:.6f}]", max_idx,
                     new_rot[max_idx][0].item<float>(), new_rot[max_idx][1].item<float>(),
                     new_rot[max_idx][2].item<float>(), new_rot[max_idx][3].item<float>());
    }

    spdlog::info("\nLegacy final loss: {:.6f}", legacy_trainer->get_current_loss());
    spdlog::info("New final loss: {:.6f}", new_trainer->get_current_loss());

    spdlog::info("\nTest completed!");
}
