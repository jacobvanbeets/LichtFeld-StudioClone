/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <filesystem>
#include <iomanip>

// Legacy implementation
#include "training/strategies/mcmc.hpp"
#include "training/rasterization/rasterizer.hpp"
#include "loader/loader.hpp"
#include "core/splat_data.hpp"

// New implementation
#include "training_new/strategies/mcmc.hpp"
#include "training_new/optimizer/render_output.hpp"
#include "loader_new/loader.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/logger.hpp"

/**
 * Real Dataset Optimizer Comparison Test
 *
 * This test runs both LEGACY and NEW MCMC implementations using the REAL bicycle point cloud
 * for 4000 iterations. Uses synthetic gradients (like toy test) but with realistic data sizes.
 *
 * It reports parameter divergence every 100 iterations to identify WHERE divergence starts.
 */

namespace {
    constexpr int TEST_SEED = 42;
    constexpr int TEST_ITERATIONS = 4000;
    constexpr float TOLERANCE = 1e-3f;  // Relaxed tolerance for real data

    void setup_deterministic_environment(int seed = TEST_SEED) {
        // CPU random
        srand(seed);
        std::srand(seed);

        // Torch random (critical for multinomial sampling!)
        torch::manual_seed(seed);
        torch::cuda::manual_seed(seed);
        torch::cuda::manual_seed_all(seed);

        // CUDA random
        cudaSetDevice(0);

        // Deterministic algorithms (may impact performance but ensures reproducibility)
        at::globalContext().setDeterministicAlgorithms(true, true);  // warn_only = true
        at::globalContext().setBenchmarkCuDNN(false);
        at::globalContext().setAllowTF32CuDNN(false);

        std::cout << "Deterministic environment setup with seed=" << seed << std::endl;
    }

    // Helper to convert torch::Tensor to lfs::core::Tensor
    lfs::core::Tensor torch_to_lfs(const torch::Tensor& t) {
        auto shape_vec = t.sizes().vec();
        std::vector<size_t> shape(shape_vec.begin(), shape_vec.end());

        lfs::core::Tensor result = lfs::core::Tensor::empty(
            lfs::core::TensorShape(shape),
            lfs::core::Device::CUDA,
            lfs::core::DataType::Float32
        );

        cudaMemcpy(
            result.ptr<float>(),
            t.data_ptr<float>(),
            t.numel() * sizeof(float),
            cudaMemcpyDeviceToDevice
        );

        return result;
    }
}

TEST(OptimizerComparison, RealBicycleDataset) {
    std::cout << "========================================" << std::endl;
    std::cout << "Real Dataset Optimizer Comparison" << std::endl;
    std::cout << "Dataset: data/bicycle" << std::endl;
    std::cout << "========================================" << std::endl;

    // Setup deterministic environment
    setup_deterministic_environment(TEST_SEED);

    std::filesystem::path dataset_path = "data/bicycle";

    // Check if dataset exists
    if (!std::filesystem::exists(dataset_path)) {
        GTEST_SKIP() << "Bicycle dataset not found at: " << dataset_path;
    }

    // ====================
    // Load dataset with LEGACY loader
    // ====================
    std::cout << "Loading point cloud with LEGACY loader..." << std::endl;
    auto legacy_loader = gs::loader::Loader::create();

    gs::loader::LoadOptions legacy_load_options;
    legacy_load_options.images_folder = "images_4";
    legacy_load_options.validate_only = false;

    auto legacy_load_result = legacy_loader->load(dataset_path, legacy_load_options);
    ASSERT_TRUE(legacy_load_result.has_value()) << "Failed to load dataset with LEGACY loader";

    auto legacy_scene = std::get<gs::loader::LoadedScene>(legacy_load_result->data);
    auto legacy_scene_center = legacy_load_result->scene_center;
    std::cout << "LEGACY: Point cloud size: " << legacy_scene.point_cloud->size() << std::endl;

    // ====================
    // Load dataset with NEW loader
    // ====================
    std::cout << "Loading point cloud with NEW loader..." << std::endl;
    auto new_loader = lfs::loader::Loader::create();

    lfs::loader::LoadOptions new_load_options;
    new_load_options.images_folder = "images_4";
    new_load_options.validate_only = false;

    auto new_load_result = new_loader->load(dataset_path, new_load_options);
    ASSERT_TRUE(new_load_result.has_value()) << "Failed to load dataset with NEW loader";

    auto new_scene = std::get<lfs::loader::LoadedScene>(new_load_result->data);
    auto new_scene_center = new_load_result->scene_center;
    std::cout << "NEW: Point cloud size: " << new_scene.point_cloud->size() << std::endl;

    // Verify both loaders produced same point cloud size
    ASSERT_EQ(legacy_scene.point_cloud->size(), new_scene.point_cloud->size())
        << "Point cloud sizes differ between loaders!";

    // ====================
    // Initialize LEGACY model from point cloud
    // ====================
    std::cout << "\nInitializing LEGACY model from point cloud..." << std::endl;

    gs::param::TrainingParameters legacy_params;
    legacy_params.dataset.data_path = dataset_path;
    legacy_params.dataset.images = "images_4";
    legacy_params.optimization.iterations = TEST_ITERATIONS;
    legacy_params.optimization.start_refine = 500;
    legacy_params.optimization.stop_refine = 15000;
    legacy_params.optimization.refine_every = 100;
    legacy_params.optimization.max_cap = 1000000;
    legacy_params.optimization.means_lr = 0.00016f;
    legacy_params.optimization.shs_lr = 0.0025f;
    legacy_params.optimization.scaling_lr = 0.005f;
    legacy_params.optimization.rotation_lr = 0.001f;
    legacy_params.optimization.opacity_lr = 0.05f;

    torch::manual_seed(TEST_SEED);  // Ensure deterministic initialization
    auto legacy_splat_result = gs::SplatData::init_model_from_pointcloud(
        legacy_params,
        legacy_scene_center,
        *legacy_scene.point_cloud
    );
    ASSERT_TRUE(legacy_splat_result.has_value()) << "Failed to initialize LEGACY SplatData";
    auto legacy_splat_data = std::move(legacy_splat_result.value());

    std::cout << "LEGACY: Initialized with " << legacy_splat_data.size() << " Gaussians" << std::endl;

    // ====================
    // Initialize NEW model from point cloud
    // ====================
    std::cout << "Initializing NEW model from point cloud..." << std::endl;

    lfs::core::param::TrainingParameters new_params;
    new_params.dataset.data_path = dataset_path;
    new_params.dataset.images = "images_4";
    new_params.optimization.iterations = TEST_ITERATIONS;
    new_params.optimization.start_refine = 500;
    new_params.optimization.stop_refine = 15000;
    new_params.optimization.refine_every = 100;
    new_params.optimization.max_cap = 1000000;
    new_params.optimization.means_lr = 0.00016f;
    new_params.optimization.shs_lr = 0.0025f;
    new_params.optimization.scaling_lr = 0.005f;
    new_params.optimization.rotation_lr = 0.001f;
    new_params.optimization.opacity_lr = 0.05f;

    torch::manual_seed(TEST_SEED);  // Use SAME seed for deterministic initialization
    auto new_splat_result = lfs::core::init_model_from_pointcloud(
        new_params,
        new_scene_center,
        *new_scene.point_cloud
    );
    ASSERT_TRUE(new_splat_result.has_value()) << "Failed to initialize NEW SplatData";
    auto new_splat_data = std::move(new_splat_result.value());

    std::cout << "NEW: Initialized with " << new_splat_data.size() << " Gaussians" << std::endl;

    // Verify both start with same number of Gaussians
    ASSERT_EQ(legacy_splat_data.size(), new_splat_data.size())
        << "Initial Gaussian counts differ!";

    // ====================
    // Initialize MCMC strategies
    // ====================
    std::cout << "\nInitializing MCMC strategies..." << std::endl;

    gs::training::MCMC legacy_strategy(std::move(legacy_splat_data));
    legacy_strategy.initialize(legacy_params.optimization);

    lfs::training::MCMC new_strategy(new_splat_data);
    new_strategy.initialize(new_params.optimization);

    // ====================
    // Training loop with SYNTHETIC gradients
    // ====================
    std::cout << "\nRunning " << TEST_ITERATIONS << " iterations with synthetic gradients..." << std::endl;

    for (int iter = 0; iter < TEST_ITERATIONS; iter++) {
        // Reset seeds before each iteration for determinism
        torch::manual_seed(TEST_SEED + iter);
        torch::cuda::manual_seed(TEST_SEED + iter);

        auto& legacy_model = legacy_strategy.get_model();
        auto& new_model = new_strategy.get_model();

        // Generate identical synthetic gradients for both
        torch::manual_seed(TEST_SEED + iter + 1000);  // Different seed for gradients
        auto grad_means = torch::randn({legacy_model.size(), 3}, torch::kCUDA) * 0.01f;
        auto grad_sh0 = torch::randn({legacy_model.size(), 3}, torch::kCUDA) * 0.001f;
        auto grad_shN = torch::randn({legacy_model.size(), 45}, torch::kCUDA) * 0.0001f;
        auto grad_scaling = torch::randn({legacy_model.size(), 3}, torch::kCUDA) * 0.001f;
        auto grad_rotation = torch::randn({legacy_model.size(), 4}, torch::kCUDA) * 0.001f;
        auto grad_opacity = torch::randn({legacy_model.size(), 1}, torch::kCUDA) * 0.001f;

        // Apply gradients to legacy
        legacy_model.means().mutable_grad() = grad_means.clone();
        legacy_model.sh0().mutable_grad() = grad_sh0.clone();
        legacy_model.shN().mutable_grad() = grad_shN.clone();
        legacy_model.scaling_raw().mutable_grad() = grad_scaling.clone();
        legacy_model.rotation_raw().mutable_grad() = grad_rotation.clone();
        legacy_model.opacity_raw().mutable_grad() = grad_opacity.clone();

        // Apply gradients to new (convert from torch)
        auto& new_opt = new_strategy.get_optimizer();
        cudaMemcpy(new_opt.get_grad(lfs::training::ParamType::Means).ptr<float>(), grad_means.data_ptr<float>(),
                   grad_means.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_opt.get_grad(lfs::training::ParamType::Sh0).ptr<float>(), grad_sh0.data_ptr<float>(),
                   grad_sh0.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_opt.get_grad(lfs::training::ParamType::ShN).ptr<float>(), grad_shN.data_ptr<float>(),
                   grad_shN.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_opt.get_grad(lfs::training::ParamType::Scaling).ptr<float>(), grad_scaling.data_ptr<float>(),
                   grad_scaling.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_opt.get_grad(lfs::training::ParamType::Rotation).ptr<float>(), grad_rotation.data_ptr<float>(),
                   grad_rotation.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_opt.get_grad(lfs::training::ParamType::Opacity).ptr<float>(), grad_opacity.data_ptr<float>(),
                   grad_opacity.numel() * sizeof(float), cudaMemcpyDeviceToDevice);

        // Create minimal render outputs for post_backward
        gs::training::RenderOutput legacy_render_out;
        legacy_render_out.radii = torch::ones({legacy_model.size()}, torch::kCUDA).to(torch::kInt32);
        legacy_render_out.visibility = torch::ones({legacy_model.size()}, torch::kCUDA);

        lfs::training::RenderOutput new_render_out;
        new_render_out.radii = lfs::core::Tensor::ones({static_cast<size_t>(new_model.size())}, lfs::core::Device::CUDA, lfs::core::DataType::Int32);
        new_render_out.visibility = lfs::core::Tensor::ones({static_cast<size_t>(new_model.size())}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        // MCMC operations (reset RNG for determinism)
        torch::manual_seed(TEST_SEED + iter);
        torch::cuda::manual_seed(TEST_SEED + iter);
        legacy_strategy.post_backward(iter, legacy_render_out);

        torch::manual_seed(TEST_SEED + iter);
        torch::cuda::manual_seed(TEST_SEED + iter);
        new_strategy.post_backward(iter, new_render_out);

        // Verify gradients match before optimizer step
        if ((iter + 1) % 100 == 0) {
            auto legacy_grad_means = legacy_model.means().grad();
            auto new_grad_means_torch = torch::from_blob(
                const_cast<float*>(new_opt.get_grad(lfs::training::ParamType::Means).ptr<float>()),
                {static_cast<long>(new_model.size()), 3},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
            );
            float grad_diff = (legacy_grad_means - new_grad_means_torch).abs().max().item().toFloat();
            if (grad_diff > 1e-6f) {
                std::cout << "  WARNING: Gradients differ at iter " << (iter + 1)
                          << " by " << grad_diff << std::endl;
            }
        }

        // Optimizer step
        legacy_strategy.step(iter);
        new_strategy.step(iter);

        // Report progress every 100 iterations
        if ((iter + 1) % 100 == 0) {
            // Compare ALL parameters
            auto legacy_means = legacy_model.means();
            auto new_means_torch = torch::from_blob(
                const_cast<float*>(new_model.means().ptr<float>()),
                {static_cast<long>(new_model.size()), 3},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
            ).clone();
            float means_diff = (legacy_means - new_means_torch).abs().max().item<float>();

            auto legacy_sh0 = legacy_model.sh0();
            auto new_sh0_torch = torch::from_blob(
                const_cast<float*>(new_model.sh0().ptr<float>()),
                {static_cast<long>(new_model.size()), 1, 3},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
            ).clone();
            float sh0_diff = (legacy_sh0 - new_sh0_torch).abs().max().item<float>();

            auto legacy_shN = legacy_model.shN();
            auto new_shN_torch = torch::from_blob(
                const_cast<float*>(new_model.shN().ptr<float>()),
                {static_cast<long>(new_model.size()), 15, 3},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
            ).clone();
            float shN_diff = (legacy_shN - new_shN_torch).abs().max().item<float>();

            auto legacy_scaling = legacy_model.scaling_raw();
            auto new_scaling_torch = torch::from_blob(
                const_cast<float*>(new_model.scaling_raw().ptr<float>()),
                {static_cast<long>(new_model.size()), 3},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
            ).clone();
            float scaling_diff = (legacy_scaling - new_scaling_torch).abs().max().item<float>();

            auto legacy_rotation = legacy_model.rotation_raw();
            auto new_rotation_torch = torch::from_blob(
                const_cast<float*>(new_model.rotation_raw().ptr<float>()),
                {static_cast<long>(new_model.size()), 4},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
            ).clone();
            float rotation_diff = (legacy_rotation - new_rotation_torch).abs().max().item<float>();

            auto legacy_opacity = legacy_model.opacity_raw();
            auto new_opacity_torch = torch::from_blob(
                const_cast<float*>(new_model.opacity_raw().ptr<float>()),
                {static_cast<long>(new_model.size()), 1},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
            ).clone();
            float opacity_diff = (legacy_opacity - new_opacity_torch).abs().max().item<float>();

            std::cout << "  Iter " << (iter + 1) << "/" << TEST_ITERATIONS
                      << std::scientific
                      << " | m=" << means_diff
                      << " sh0=" << sh0_diff
                      << " shN=" << shN_diff
                      << " scl=" << scaling_diff
                      << " rot=" << rotation_diff
                      << " opa=" << opacity_diff << std::endl;
        }
    }

    std::cout << "\nTraining complete!" << std::endl;

    // Final comparison
    auto& final_legacy = legacy_strategy.get_model();
    auto& final_new = new_strategy.get_model();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Final Comparison" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Gaussians: LEGACY=" << final_legacy.size()
              << ", NEW=" << final_new.size() << std::endl;

    auto legacy_means_final = final_legacy.means();
    auto new_means_final = torch::from_blob(
        const_cast<float*>(final_new.means().ptr<float>()),
        {static_cast<long>(final_new.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
    ).clone();
    float final_means_diff = (legacy_means_final - new_means_final).abs().max().item<float>();

    std::cout << "Max means difference: " << std::scientific << final_means_diff << std::endl;
    std::cout << "Tolerance: " << TOLERANCE << std::endl;

    EXPECT_LE(final_means_diff, TOLERANCE) << "Final means differ by more than tolerance";
    std::cout << "========================================" << std::endl;
}
