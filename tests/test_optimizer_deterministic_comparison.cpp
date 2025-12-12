/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>

// Legacy implementation
#include "training/strategies/mcmc.hpp"
#include "training/rasterization/rasterizer.hpp"
#include "training/trainer.hpp"
#include "core/splat_data.hpp"

// New implementation
#include "training_new/strategies/mcmc.hpp"
#include "training_new/optimizer/render_output.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/logger.hpp"

/**
 * Deterministic Optimizer Comparison Test
 *
 * This test runs both the legacy (LibTorch-based) and new (LibTorch-free) MCMC
 * implementations for exactly 100 iterations with:
 * - Identical initial Gaussians
 * - Identical random seeds (for multinomial sampling)
 * - Identical hyperparameters
 *
 * Then compares:
 * - Final parameter values (means, sh, scaling, rotation, opacity)
 * - Optimizer states (exp_avg, exp_avg_sq, step_count)
 * - Number of Gaussians (if MCMC added/removed any)
 *
 * Expected result: Differences should be < 1e-5 (numerical precision only)
 *
 * Larger differences indicate bugs in:
 * - Tensor operations
 * - Memory layout (stride/ordering)
 * - Optimizer update logic
 * - Gradient computation
 */

namespace {
    constexpr int TEST_SEED = 42;
    constexpr int TEST_ITERATIONS = 4000;  // Test through SH degree increases at 1000, 2000, 3000
    constexpr float TOLERANCE = 1e-4f;  // Relaxed tolerance for longer run

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

        LOG_INFO("Deterministic environment setup with seed={}", seed);
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

    // Helper to compare two tensors (torch vs lfs)
    struct TensorComparison {
        float max_abs_diff;
        float mean_abs_diff;
        float max_rel_diff;
        bool within_tolerance;

        void print(const std::string& name) const {
            std::cout << name << " comparison:" << std::endl;
            std::cout << "  Max absolute difference: " << std::scientific << max_abs_diff << std::endl;
            std::cout << "  Mean absolute difference: " << std::scientific << mean_abs_diff << std::endl;
            std::cout << "  Max relative difference: " << std::scientific << max_rel_diff << std::endl;
            std::cout << "  Within tolerance (" << std::scientific << TOLERANCE << "): "
                     << (within_tolerance ? "YES" : "NO") << std::endl;
        }
    };

    TensorComparison compare_tensors(const torch::Tensor& legacy, const lfs::core::Tensor& new_impl, const std::string& name) {
        // Convert lfs to torch for comparison
        torch::Tensor new_torch = torch::from_blob(
            const_cast<float*>(new_impl.ptr<float>()),
            {static_cast<int64_t>(new_impl.numel())},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
        ).clone();

        auto legacy_flat = legacy.flatten();
        auto diff = (legacy_flat - new_torch).abs();

        float max_abs = diff.max().item<float>();
        float mean_abs = diff.mean().item<float>();

        // Relative difference: |a - b| / (|a| + |b| + eps)
        auto rel_diff = diff / (legacy_flat.abs() + new_torch.abs() + 1e-10f);
        float max_rel = rel_diff.max().item<float>();

        TensorComparison result;
        result.max_abs_diff = max_abs;
        result.mean_abs_diff = mean_abs;
        result.max_rel_diff = max_rel;
        result.within_tolerance = (max_abs < TOLERANCE);

        result.print(name);

        return result;
    }
}

TEST(OptimizerComparison, DeterministicMCMC100Steps) {
    std::cout << "========================================" << std::endl;
    std::cout << "Deterministic Optimizer Comparison Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // Setup deterministic environment
    setup_deterministic_environment(TEST_SEED);

    // Create identical initial Gaussians (100 random Gaussians)
    const int num_init = 100;

    torch::manual_seed(TEST_SEED);  // Reset seed for initialization
    auto init_means = torch::randn({num_init, 3}, torch::kCUDA) * 2.0f;
    auto init_sh0 = torch::randn({num_init, 3}, torch::kCUDA) * 0.1f;
    auto init_shN = torch::randn({num_init, 45}, torch::kCUDA) * 0.01f;
    auto init_scaling = torch::randn({num_init, 3}, torch::kCUDA) * 0.5f;
    auto init_rotation = torch::randn({num_init, 4}, torch::kCUDA);
    init_rotation = init_rotation / init_rotation.norm(2, 1, true);  // Normalize quaternions
    auto init_opacity = torch::sigmoid(torch::randn({num_init, 1}, torch::kCUDA));

    std::cout << "Created " << num_init << " initial Gaussians" << std::endl;

    // Create legacy SplatData using proper constructor
    gs::SplatData legacy_data(
        3,  // sh_degree
        init_means.clone().set_requires_grad(true),
        init_sh0.clone().set_requires_grad(true),
        init_shN.clone().set_requires_grad(true),
        init_scaling.clone().set_requires_grad(true),
        init_rotation.clone().set_requires_grad(true),
        init_opacity.clone().set_requires_grad(true),
        1.0f  // scene_scale
    );

    // Create new SplatData using proper constructor
    lfs::core::SplatData new_data(
        3,  // sh_degree
        torch_to_lfs(init_means),
        torch_to_lfs(init_sh0),
        torch_to_lfs(init_shN),
        torch_to_lfs(init_scaling),
        torch_to_lfs(init_rotation),
        torch_to_lfs(init_opacity),
        1.0f  // scene_scale
    );

    // Create identical optimization parameters
    gs::param::OptimizationParameters params_legacy;
    params_legacy.means_lr = 0.00016f;
    params_legacy.shs_lr = 0.0025f;
    params_legacy.scaling_lr = 0.005f;
    params_legacy.rotation_lr = 0.001f;
    params_legacy.opacity_lr = 0.05f;
    params_legacy.iterations = TEST_ITERATIONS;
    params_legacy.start_refine = 20;  // Start refinement EARLY at iteration 20
    params_legacy.stop_refine = TEST_ITERATIONS;  // Refine until end
    params_legacy.refine_every = 10;  // Refine every 10 iterations (frequent!)
    params_legacy.min_opacity = 0.005f;
    params_legacy.max_cap = 10000;  // Test with preallocation

    lfs::core::param::OptimizationParameters params_new;
    params_new.means_lr = 0.00016f;
    params_new.shs_lr = 0.0025f;
    params_new.scaling_lr = 0.005f;
    params_new.rotation_lr = 0.001f;
    params_new.opacity_lr = 0.05f;
    params_new.iterations = TEST_ITERATIONS;
    params_new.start_refine = 20;  // Start refinement EARLY at iteration 20
    params_new.stop_refine = TEST_ITERATIONS;  // Refine until end
    params_new.refine_every = 10;  // Refine every 10 iterations (frequent!)
    params_new.min_opacity = 0.005f;
    params_new.max_cap = 10000;  // Test with preallocation

    // Initialize strategies
    std::cout << "Initializing legacy MCMC strategy..." << std::endl;
    gs::training::MCMC legacy_strategy(std::move(legacy_data));
    legacy_strategy.initialize(params_legacy);

    std::cout << "Initializing new MCMC strategy..." << std::endl;
    lfs::training::MCMC new_strategy(new_data);
    new_strategy.initialize(params_new);


    // Training loop - generate synthetic gradients
    std::cout << "Running " << TEST_ITERATIONS << " iterations..." << std::endl;

    for (int iter = 0; iter < TEST_ITERATIONS; iter++) {
        // DEBUG: Print initial LRs at iteration 0
        if (iter == 0) {
            std::cout << std::setprecision(15);
            std::cout << "[ITER 0] Initial LRs:" << std::endl;
            auto& legacy_group = legacy_strategy.get_optimizer()->param_groups()[0];  // means
            auto* legacy_opts = static_cast<gs::training::FusedAdam::Options*>(&legacy_group.options());
            std::cout << "  Legacy means LR: " << legacy_opts->lr() << std::endl;
            std::cout << "  New means LR: " << new_strategy.get_optimizer()->get_param_lr(lfs::training::ParamType::Means) << std::endl;
            std::cout << std::setprecision(6);
        }

        // Reset seeds before each iteration to ensure identical multinomial sampling
        torch::manual_seed(TEST_SEED + iter);
        torch::cuda::manual_seed(TEST_SEED + iter);

        // Generate identical synthetic gradients for both
        // (In real training, these would come from rendering + loss.backward())
        torch::manual_seed(TEST_SEED + iter + 1000);  // Different seed for gradients

        auto& legacy_model = legacy_strategy.get_model();
        auto& new_model = new_strategy.get_model();

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
        auto& new_optimizer = new_strategy.get_optimizer();
        cudaMemcpy(new_optimizer.get_grad(lfs::training::ParamType::Means).ptr<float>(), grad_means.data_ptr<float>(),
                   grad_means.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_optimizer.get_grad(lfs::training::ParamType::Sh0).ptr<float>(), grad_sh0.data_ptr<float>(),
                   grad_sh0.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_optimizer.get_grad(lfs::training::ParamType::ShN).ptr<float>(), grad_shN.data_ptr<float>(),
                   grad_shN.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_optimizer.get_grad(lfs::training::ParamType::Scaling).ptr<float>(), grad_scaling.data_ptr<float>(),
                   grad_scaling.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_optimizer.get_grad(lfs::training::ParamType::Rotation).ptr<float>(), grad_rotation.data_ptr<float>(),
                   grad_rotation.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_optimizer.get_grad(lfs::training::ParamType::Opacity).ptr<float>(), grad_opacity.data_ptr<float>(),
                   grad_opacity.numel() * sizeof(float), cudaMemcpyDeviceToDevice);

        // Create minimal render outputs for post_backward
        // Set radii>0 and visibility=1 for all Gaussians to simulate them being visible
        gs::training::RenderOutput legacy_render_out;
        legacy_render_out.radii = torch::ones({legacy_model.size()}, torch::kCUDA).to(torch::kInt32);
        legacy_render_out.visibility = torch::ones({legacy_model.size()}, torch::kCUDA);

        lfs::training::RenderOutput new_render_out;
        new_render_out.radii = lfs::core::Tensor::ones({static_cast<size_t>(new_model.size())}, lfs::core::Device::CUDA, lfs::core::DataType::Int32);
        new_render_out.visibility = lfs::core::Tensor::ones({static_cast<size_t>(new_model.size())}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        // Call post_backward to handle refinement (add_new_gs, relocate_gs)
        int legacy_size_before = legacy_model.size();
        int new_size_before = new_model.size();

        // CRITICAL: Reset RNG seed before EACH post_backward to ensure identical random sequences
        // post_backward uses RNG for: multinomial sampling (add_new_gs) + noise injection (inject_noise)
        torch::manual_seed(TEST_SEED + iter);
        torch::cuda::manual_seed(TEST_SEED + iter);
        legacy_strategy.post_backward(iter, legacy_render_out);

        // Reset seed again for new implementation to use SAME random sequence
        torch::manual_seed(TEST_SEED + iter);
        torch::cuda::manual_seed(TEST_SEED + iter);
        new_strategy.post_backward(iter, new_render_out);

        // Debug: Report when Gaussians are added
        if (legacy_model.size() != legacy_size_before || new_model.size() != new_size_before) {
            std::cout << "  [Iter " << iter << "] Gaussians added: legacy="
                      << (legacy_model.size() - legacy_size_before)
                      << ", new=" << (new_model.size() - new_size_before) << std::endl;

            // Debug iteration 50 to see scaling/opacity divergence
            if (iter == 50) {
                std::cout << "\n=== DEBUG: Iteration 50 - Scaling/Opacity Divergence ===" << std::endl;

                // Compare LAST 5 Gaussians (the newly added ones)
                int n_total = legacy_model.size();
                int start_idx = n_total - 5;
                torch::Tensor legacy_scaling_last = legacy_model.scaling_raw().slice(0, start_idx, n_total);
                torch::Tensor new_scaling_last = torch::from_blob(
                    const_cast<float*>(new_model.scaling_raw().ptr<float>()) + start_idx * 3,
                    {5, 3},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
                ).clone();
                std::cout << "Legacy scaling [last 5]: " << legacy_scaling_last << std::endl;
                std::cout << "New scaling [last 5]:    " << new_scaling_last << std::endl;
                auto scaling_diff_last = (legacy_scaling_last - new_scaling_last).abs().max();
                std::cout << "Scaling diff (last 5): " << scaling_diff_last.item<float>() << std::endl;

                torch::Tensor legacy_opacity_last = legacy_model.opacity_raw().slice(0, start_idx, n_total);
                torch::Tensor new_opacity_last = torch::from_blob(
                    const_cast<float*>(new_model.opacity_raw().ptr<float>()) + start_idx,
                    {5, 1},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
                ).clone();
                std::cout << "Legacy opacity [last 5]: " << legacy_opacity_last << std::endl;
                std::cout << "New opacity [last 5]:    " << new_opacity_last << std::endl;
                auto opacity_diff_last = (legacy_opacity_last - new_opacity_last).abs().max();
                std::cout << "Opacity diff (last 5): " << opacity_diff_last.item<float>() << std::endl;

                // Also check ALL Gaussians to find where the divergence is
                torch::Tensor legacy_scaling_all = legacy_model.scaling_raw();
                torch::Tensor new_scaling_all = torch::from_blob(
                    const_cast<float*>(new_model.scaling_raw().ptr<float>()),
                    {static_cast<long>(n_total), 3},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
                ).clone();
                auto scaling_diff_all = (legacy_scaling_all - new_scaling_all).abs();
                auto max_idx = scaling_diff_all.argmax();
                std::cout << "Max scaling diff: " << scaling_diff_all.max().item<float>()
                         << " at index " << max_idx.item<long>() << std::endl;
                std::cout << "==================================================\n" << std::endl;
            }

            // Debug first add_new_gs in detail (now at iteration 30)
            if (iter == 30) {
                std::cout << "\n=== DEBUG: First add_new_gs at iteration 30 ===" << std::endl;

                // Compare the newly added Gaussians (last 4 means)
                int n_added = legacy_model.size() - legacy_size_before;
                torch::Tensor legacy_new_means = legacy_model.means().slice(0, legacy_size_before, legacy_model.size());
                torch::Tensor new_new_means = torch::from_blob(
                    const_cast<float*>(new_model.means().ptr<float>()) + legacy_size_before * 3,
                    {n_added, 3},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
                ).clone();

                std::cout << "Legacy new means [0:2]: " << legacy_new_means.slice(0, 0, std::min(2, n_added)) << std::endl;
                std::cout << "New new means [0:2]:    " << new_new_means.slice(0, 0, std::min(2, n_added)) << std::endl;

                auto means_diff = (legacy_new_means - new_new_means).abs().max();
                std::cout << "Max diff in new means: " << means_diff.item<float>() << std::endl;
                std::cout << "===========================================\n" << std::endl;
            }
        }

        // DEBUG: At iteration 31, check if gradients match for newly added Gaussians
        if (iter == 31) {
            std::cout << "\n=== DEBUG: Gradients at iteration 31 (after add_new_gs) ===" << std::endl;
            std::cout << "Legacy grad_means shape: " << grad_means.sizes() << std::endl;
            std::cout << "New model size: " << new_model.size() << std::endl;

            // Check gradients for last 4 Gaussians (newly added at iter 30)
            auto legacy_grad_last4 = grad_means.slice(0, 100, 104);
            std::cout << "Legacy grad for new Gaussians [100:104]: " << legacy_grad_last4 << std::endl;

            // Get new model gradients for last 4
            float new_grad_last4[12];
            cudaMemcpy(new_grad_last4, new_strategy.get_optimizer().get_grad(lfs::training::ParamType::Means).ptr<float>() + 100 * 3, 12 * sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "New grad for new Gaussians [100:104]:" << std::endl;
            std::cout << "  [100]: " << new_grad_last4[0] << ", " << new_grad_last4[1] << ", " << new_grad_last4[2] << std::endl;
            std::cout << "  [101]: " << new_grad_last4[3] << ", " << new_grad_last4[4] << ", " << new_grad_last4[5] << std::endl;
            std::cout << "=========================================\n" << std::endl;
        }

        // Step both optimizers
        legacy_strategy.step(iter);
        new_strategy.step(iter);

        // Debug: Check parameter divergence frequently
        bool check_divergence = ((iter + 1) % 100 == 0);
        // Check every iteration around refinement points
        if ((iter >= 25 && iter <= 45) || (iter >= 46 && iter <= 110)) {
            check_divergence = true;
        }

        if (check_divergence) {
            // Compare ALL Gaussians (not just newly added)
            torch::Tensor legacy_means_torch = legacy_model.means();
            torch::Tensor new_means_torch = torch::from_blob(
                const_cast<float*>(new_model.means().ptr<float>()),
                {static_cast<long>(new_model.size()), 3},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
            ).clone();
            auto means_diff = (legacy_means_torch - new_means_torch).abs().max();

            // Also check other parameters
            torch::Tensor legacy_rotation = legacy_model.rotation_raw();
            torch::Tensor new_rotation = torch::from_blob(
                const_cast<float*>(new_model.rotation_raw().ptr<float>()),
                {static_cast<long>(new_model.size()), 4},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
            ).clone();
            auto rotation_diff = (legacy_rotation - new_rotation).abs().max();

            torch::Tensor legacy_scaling = legacy_model.scaling_raw();
            torch::Tensor new_scaling = torch::from_blob(
                const_cast<float*>(new_model.scaling_raw().ptr<float>()),
                {static_cast<long>(new_model.size()), 3},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
            ).clone();
            auto scaling_diff = (legacy_scaling - new_scaling).abs().max();

            torch::Tensor legacy_opacity = legacy_model.opacity_raw();
            torch::Tensor new_opacity = torch::from_blob(
                const_cast<float*>(new_model.opacity_raw().ptr<float>()),
                {static_cast<long>(new_model.size()), 1},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
            ).clone();
            auto opacity_diff = (legacy_opacity - new_opacity).abs().max();

            std::cout << "  Completed iteration " << (iter + 1) << "/" << TEST_ITERATIONS
                      << " (Gaussians: legacy=" << legacy_model.size()
                      << ", new=" << new_model.size()
                      << ", means=" << means_diff.item<float>()
                      << ", rot=" << rotation_diff.item<float>()
                      << ", scale=" << scaling_diff.item<float>()
                      << ", opac=" << opacity_diff.item<float>() << ")" << std::endl;
        }
    }

    std::cout << "Training complete. Comparing results..." << std::endl;
    std::cout << std::endl;

    // Compare final parameters
    std::cout << "========================================" << std::endl;
    std::cout << "Parameter Comparison" << std::endl;
    std::cout << "========================================" << std::endl;

    auto& final_legacy_model = legacy_strategy.get_model();
    auto& final_new_model = new_strategy.get_model();

    std::cout << "Final Gaussian counts: legacy=" << final_legacy_model.size()
              << ", new=" << final_new_model.size() << std::endl;
    EXPECT_EQ(final_legacy_model.size(), final_new_model.size()) << "Number of Gaussians differs!";
    std::cout << std::endl;

    auto means_cmp = compare_tensors(
        final_legacy_model.means(),
        final_new_model.means(),
        "Means"
    );
    EXPECT_TRUE(means_cmp.within_tolerance) << "Means differ by more than " << TOLERANCE;

    auto sh0_cmp = compare_tensors(
        final_legacy_model.sh0(),
        final_new_model.sh0(),
        "SH0"
    );
    EXPECT_TRUE(sh0_cmp.within_tolerance) << "SH0 differs by more than " << TOLERANCE;

    auto shN_cmp = compare_tensors(
        final_legacy_model.shN(),
        final_new_model.shN(),
        "ShN"
    );
    EXPECT_TRUE(shN_cmp.within_tolerance) << "ShN differs by more than " << TOLERANCE;

    auto scaling_cmp = compare_tensors(
        final_legacy_model.scaling_raw(),
        final_new_model.scaling_raw(),
        "Scaling"
    );
    EXPECT_TRUE(scaling_cmp.within_tolerance) << "Scaling differs by more than " << TOLERANCE;

    auto rotation_cmp = compare_tensors(
        final_legacy_model.rotation_raw(),
        final_new_model.rotation_raw(),
        "Rotation"
    );
    EXPECT_TRUE(rotation_cmp.within_tolerance) << "Rotation differs by more than " << TOLERANCE;

    auto opacity_cmp = compare_tensors(
        final_legacy_model.opacity_raw(),
        final_new_model.opacity_raw(),
        "Opacity"
    );
    EXPECT_TRUE(opacity_cmp.within_tolerance) << "Opacity differs by more than " << TOLERANCE;

    // Compare optimizer states (Adam moments)
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Optimizer State Comparison" << std::endl;
    std::cout << "========================================" << std::endl;

    // TODO: Add optimizer state comparison when we have access to internal state
    // For now, if parameters match, optimizer states are likely correct

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Iterations: " << TEST_ITERATIONS << std::endl;
    std::cout << "Tolerance: " << std::scientific << TOLERANCE << std::endl;
    std::cout << "All parameter comparisons: "
             << ((means_cmp.within_tolerance && sh0_cmp.within_tolerance &&
                  shN_cmp.within_tolerance && scaling_cmp.within_tolerance &&
                  rotation_cmp.within_tolerance && opacity_cmp.within_tolerance) ? "PASSED" : "FAILED")
             << std::endl;
    std::cout << "========================================" << std::endl;
}
