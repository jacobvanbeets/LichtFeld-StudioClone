/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Comprehensive comparison test between legacy (LibTorch) and new (LFS) gsplat rasterizers
// to identify differences in forward/backward passes

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

// Legacy LibTorch-based gsplat
#include "training/rasterization/rasterizer.hpp"
#include "core/camera.hpp"
#include "core/splat_data.hpp"

// New LFS gsplat
#include "training_new/rasterization/gsplat_rasterizer.hpp"
#include "training_new/optimizer/adam_optimizer.hpp"
#include "core_new/tensor.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/camera.hpp"
#include "core_new/logger.hpp"

// Use correct namespaces for legacy types
using LegacySplatData = gs::SplatData;
using LegacyCamera = gs::Camera;

class GsplatLegacyVsLfsTest : public ::testing::Test {
protected:
    void SetUp() override {
        const int64_t N = 1000;  // Number of Gaussians
        const int sh_degree = 0;

        // Create deterministic Gaussian parameters using LibTorch
        torch::manual_seed(42);

        // Means [N, 3]
        means_torch_ = torch::randn({N, 3}, torch::kCUDA) * 2.0f;

        // sh0 is [N, 1, 3] for optimizer compatibility
        sh0_torch_ = torch::randn({N, 1, 3}, torch::kCUDA) * 0.5f + 0.5f;

        // shN is empty for sh_degree=0
        shN_torch_ = torch::zeros({N, 0, 3}, torch::kCUDA);

        // Raw scales (will be exp'd) [N, 3]
        scaling_torch_ = torch::randn({N, 3}, torch::kCUDA) * 0.5f - 3.0f;

        // Raw rotations (will be normalized) [N, 4]
        rotation_torch_ = torch::randn({N, 4}, torch::kCUDA);

        // Raw opacity (will be sigmoid'd) [N, 1]
        opacity_torch_ = torch::randn({N, 1}, torch::kCUDA);

        // Create legacy SplatData
        legacy_splat_data_ = std::make_unique<LegacySplatData>(
            sh_degree,
            means_torch_.clone(),
            sh0_torch_.clone(),
            shN_torch_.clone(),
            scaling_torch_.clone(),
            rotation_torch_.clone(),
            opacity_torch_.clone(),
            1.0f
        );

        // Create LFS tensors from LibTorch tensors
        auto to_lfs = [](const torch::Tensor& t) {
            auto cpu = t.cpu().contiguous();
            std::vector<size_t> shape_vec;
            for (int i = 0; i < cpu.dim(); ++i) {
                shape_vec.push_back(cpu.size(i));
            }
            lfs::core::TensorShape shape(shape_vec);
            auto lfs_tensor = lfs::core::Tensor::from_blob(
                cpu.data_ptr<float>(), shape,
                lfs::core::Device::CPU, lfs::core::DataType::Float32
            ).clone().to(lfs::core::Device::CUDA);
            return lfs_tensor;
        };

        means_lfs_ = to_lfs(means_torch_);
        sh0_lfs_ = to_lfs(sh0_torch_);
        shN_lfs_ = lfs::core::Tensor::zeros({static_cast<size_t>(N), 0, 3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        scaling_lfs_ = to_lfs(scaling_torch_);
        rotation_lfs_ = to_lfs(rotation_torch_);
        opacity_lfs_ = to_lfs(opacity_torch_);

        // Create LFS SplatData
        lfs_splat_data_ = std::make_unique<lfs::core::SplatData>(
            sh_degree,
            means_lfs_.clone(),
            sh0_lfs_.clone(),
            shN_lfs_,
            scaling_lfs_.clone(),
            rotation_lfs_.clone(),
            opacity_lfs_.clone(),
            1.0f
        );

        // Create legacy camera
        auto R_torch = torch::eye(3, torch::kCUDA);
        auto T_torch = torch::tensor({0.0f, 0.0f, 5.0f}, torch::kCUDA);

        legacy_camera_ = std::make_unique<LegacyCamera>(
            R_torch, T_torch,
            500.0f, 500.0f,
            320.0f, 240.0f,
            torch::Tensor(), torch::Tensor(),
            gsplat::CameraModelType::PINHOLE,
            "test_image", "",
            std::filesystem::path{},  // No mask path
            640, 480, 0
        );

        // Create LFS camera
        auto R_lfs = lfs::core::Tensor::eye(3, lfs::core::Device::CUDA);
        std::vector<float> T_data = {0.0f, 0.0f, 5.0f};
        auto T_lfs = lfs::core::Tensor::from_blob(T_data.data(), {3}, lfs::core::Device::CPU, lfs::core::DataType::Float32).to(lfs::core::Device::CUDA);

        lfs_camera_ = std::make_unique<lfs::core::Camera>(
            R_lfs, T_lfs,
            500.0f, 500.0f,
            320.0f, 240.0f,
            lfs::core::Tensor(), lfs::core::Tensor(),
            gsplat::CameraModelType::PINHOLE,
            "test_image", "",
            std::filesystem::path{},  // No mask path
            640, 480, 0
        );

        // Background colors
        bg_color_torch_ = torch::ones({3}, torch::kCUDA) * 0.5f;
        bg_color_lfs_ = lfs::core::Tensor::ones({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32).mul(0.5f);
    }

    // Legacy (LibTorch)
    std::unique_ptr<LegacySplatData> legacy_splat_data_;
    std::unique_ptr<LegacyCamera> legacy_camera_;
    torch::Tensor means_torch_, sh0_torch_, shN_torch_, scaling_torch_, rotation_torch_, opacity_torch_;
    torch::Tensor bg_color_torch_;

    // LFS
    std::unique_ptr<lfs::core::SplatData> lfs_splat_data_;
    std::unique_ptr<lfs::core::Camera> lfs_camera_;
    lfs::core::Tensor means_lfs_, sh0_lfs_, shN_lfs_, scaling_lfs_, rotation_lfs_, opacity_lfs_;
    lfs::core::Tensor bg_color_lfs_;
};

// Helper to convert LFS tensor to torch tensor
torch::Tensor lfs_to_torch(const lfs::core::Tensor& t) {
    auto cpu = t.cpu();
    std::vector<int64_t> shape;
    for (size_t i = 0; i < cpu.ndim(); ++i) {
        shape.push_back(cpu.shape()[i]);
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto torch_tensor = torch::from_blob(
        const_cast<float*>(cpu.ptr<float>()),
        shape, options
    ).clone().to(torch::kCUDA);
    return torch_tensor;
}

// Compare forward pass outputs
TEST_F(GsplatLegacyVsLfsTest, ForwardOutputComparison) {
    std::cout << "\n=== Forward Output Comparison (Legacy vs LFS gsplat) ===" << std::endl;

    // First verify input tensors match between legacy and LFS
    std::cout << "Input verification:" << std::endl;
    auto verify_input = [](const torch::Tensor& torch_t, const lfs::core::Tensor& lfs_t, const char* name) {
        auto lfs_cpu = lfs_t.cpu();
        std::vector<int64_t> shape;
        for (size_t i = 0; i < lfs_cpu.ndim(); ++i) {
            shape.push_back(lfs_cpu.shape()[i]);
        }
        auto torch_cpu = torch_t.cpu().contiguous();
        auto lfs_as_torch = torch::from_blob(
            const_cast<float*>(lfs_cpu.ptr<float>()),
            shape, torch::kFloat32
        ).clone();

        float max_diff = (torch_cpu - lfs_as_torch).abs().max().item<float>();
        std::cout << "  " << name << " input diff: " << max_diff << std::endl;
        return max_diff < 1e-6f;
    };

    verify_input(legacy_splat_data_->means(), lfs_splat_data_->means(), "Means");
    verify_input(legacy_splat_data_->sh0(), lfs_splat_data_->sh0(), "SH0");
    verify_input(legacy_splat_data_->scaling_raw(), lfs_splat_data_->scaling_raw(), "Scaling");
    verify_input(legacy_splat_data_->rotation_raw(), lfs_splat_data_->rotation_raw(), "Rotation");
    verify_input(legacy_splat_data_->opacity_raw(), lfs_splat_data_->opacity_raw(), "Opacity");

    // Compare camera matrices
    std::cout << "\nCamera comparison:" << std::endl;
    auto legacy_viewmat = legacy_camera_->world_view_transform().cpu();
    auto lfs_viewmat = lfs_to_torch(lfs_camera_->world_view_transform()).cpu();  // Also move to CPU
    float viewmat_diff = (legacy_viewmat - lfs_viewmat).abs().max().item<float>();
    std::cout << "  ViewMatrix diff: " << viewmat_diff << std::endl;
    // Compare K matrices
    auto legacy_K = legacy_camera_->K().cpu();
    auto lfs_K = lfs_to_torch(lfs_camera_->K()).cpu();  // Also move to CPU
    float K_diff = (legacy_K - lfs_K).abs().max().item<float>();
    std::cout << "  K matrix diff: " << K_diff << std::endl;

    // Run legacy forward
    auto legacy_output = gs::training::rasterize(
        *legacy_camera_,
        *legacy_splat_data_,
        bg_color_torch_,
        1.0f, false, false,
        gs::training::RenderMode::RGB
    );

    // Run LFS forward
    auto lfs_result = lfs::training::gsplat_rasterize_forward(
        *lfs_camera_,
        *lfs_splat_data_,
        bg_color_lfs_,
        1.0f, false,
        lfs::training::GsplatRenderMode::RGB,
        false  // use_gut
    );
    ASSERT_TRUE(lfs_result.has_value()) << "LFS forward failed: " << lfs_result.error();
    auto& [lfs_output, lfs_ctx] = lfs_result.value();

    // Convert LFS output to torch for comparison
    auto lfs_img_torch = lfs_to_torch(lfs_output.image);
    auto lfs_alpha_torch = lfs_to_torch(lfs_output.alpha);

    // Compare dimensions
    std::cout << "Legacy image shape: " << legacy_output.image.sizes() << std::endl;
    std::cout << "LFS image shape: " << lfs_img_torch.sizes() << std::endl;

    EXPECT_EQ(legacy_output.image.size(0), lfs_img_torch.size(0));
    EXPECT_EQ(legacy_output.image.size(1), lfs_img_torch.size(1));
    EXPECT_EQ(legacy_output.image.size(2), lfs_img_torch.size(2));

    // Compare rendered images
    auto img_diff = (legacy_output.image - lfs_img_torch).abs();
    float max_diff = img_diff.max().item<float>();
    float mean_diff = img_diff.mean().item<float>();

    std::cout << "Image max_diff: " << max_diff << std::endl;
    std::cout << "Image mean_diff: " << mean_diff << std::endl;

    // Compare alpha
    auto alpha_diff = (legacy_output.alpha - lfs_alpha_torch).abs();
    float alpha_max_diff = alpha_diff.max().item<float>();
    float alpha_mean_diff = alpha_diff.mean().item<float>();

    std::cout << "Alpha max_diff: " << alpha_max_diff << std::endl;
    std::cout << "Alpha mean_diff: " << alpha_mean_diff << std::endl;

    // Print ranges
    std::cout << "Legacy image range: [" << legacy_output.image.min().item<float>()
              << ", " << legacy_output.image.max().item<float>() << "]" << std::endl;
    std::cout << "LFS image range: [" << lfs_img_torch.min().item<float>()
              << ", " << lfs_img_torch.max().item<float>() << "]" << std::endl;

    // Print means for each channel to see where the difference is
    std::cout << "\nPer-channel analysis:" << std::endl;
    for (int c = 0; c < 3; ++c) {
        auto leg_ch = legacy_output.image[c];
        auto lfs_ch = lfs_img_torch[c];
        std::cout << "Channel " << c << ": legacy_mean=" << leg_ch.mean().item<float>()
                  << ", lfs_mean=" << lfs_ch.mean().item<float>()
                  << ", diff_mean=" << (leg_ch - lfs_ch).abs().mean().item<float>() << std::endl;
    }

    // Check visibility/radii comparison
    std::cout << "\nVisibility comparison:" << std::endl;
    std::cout << "Legacy visible: " << legacy_output.visibility.sum().item<int64_t>() << " / 1000" << std::endl;

    // Check render context values (context stores raw pointers, not tensors)
    std::cout << "\nLFS context: N=" << lfs_ctx.N << ", n_isects=" << lfs_ctx.n_isects << std::endl;

    // Clean up arena
    auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
    arena.end_frame(lfs_ctx.frame_id);

    // They should be IDENTICAL (same underlying CUDA kernels, just different tensor wrappers)
    // Tolerance should be near floating point precision
    EXPECT_LT(max_diff, 1e-5f) << "Forward outputs must match exactly! max_diff=" << max_diff;
    EXPECT_LT(mean_diff, 1e-6f) << "Forward outputs must match exactly! mean_diff=" << mean_diff;
}

// Test campos (camera position) calculation - this is computed differently in Legacy vs LFS
TEST_F(GsplatLegacyVsLfsTest, CamposCalculationComparison) {
    std::cout << "\n=== Campos Calculation Comparison ===" << std::endl;

    // Get viewmat from both cameras
    auto legacy_viewmat = legacy_camera_->world_view_transform().cuda();  // [1, 4, 4]
    auto lfs_viewmat = lfs_camera_->world_view_transform();  // [1, 4, 4]

    // Legacy method: full matrix inverse
    auto legacy_viewmat_inv = at::inverse(legacy_viewmat);  // [1, 4, 4]
    auto legacy_campos = legacy_viewmat_inv.index({torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 3), 3}); // [1, 3]

    // LFS method: -R^T * t
    auto R = lfs_viewmat.slice(1, 0, 3).slice(2, 0, 3);  // [1, 3, 3]
    auto t = lfs_viewmat.slice(1, 0, 3).slice(2, 3, 4);  // [1, 3, 1]
    auto R_t = R.transpose(-1, -2);  // [1, 3, 3]
    auto lfs_campos = R_t.bmm(t).mul(-1.0f).squeeze(-1);  // [1, 3]

    // Convert LFS to torch for comparison
    auto lfs_campos_torch = lfs_to_torch(lfs_campos).cpu();
    auto legacy_campos_cpu = legacy_campos.cpu();

    std::cout << "Legacy campos: " << legacy_campos_cpu << std::endl;
    std::cout << "LFS campos: " << lfs_campos_torch << std::endl;

    float campos_diff = (legacy_campos_cpu - lfs_campos_torch).abs().max().item<float>();
    std::cout << "Campos diff: " << campos_diff << std::endl;

    EXPECT_LT(campos_diff, 1e-5f) << "Campos calculation differs!";
}

// Compare intermediate values - dirs and SH coefficients
TEST_F(GsplatLegacyVsLfsTest, IntermediateValuesComparison) {
    std::cout << "\n=== Intermediate Values Comparison ===" << std::endl;

    // Get activated params from both
    auto means_leg = legacy_splat_data_->get_means();
    auto sh_coeffs_leg = legacy_splat_data_->get_shs();

    auto means_lfs = lfs_splat_data_->get_means();
    auto sh_coeffs_lfs = lfs_splat_data_->get_shs();

    // Get viewmat
    auto viewmat_leg = legacy_camera_->world_view_transform().cuda();
    auto viewmat_lfs = lfs_camera_->world_view_transform();

    // ============= LEGACY: Compute campos and dirs =============
    auto viewmat_inv_leg = at::inverse(viewmat_leg);
    auto campos_leg = viewmat_inv_leg.index({torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 3), 3});
    auto dirs_leg = means_leg.unsqueeze(0) - campos_leg.unsqueeze(1);  // [1, N, 3]

    // ============= LFS: Compute campos and dirs =============
    auto R = viewmat_lfs.slice(1, 0, 3).slice(2, 0, 3);
    auto t = viewmat_lfs.slice(1, 0, 3).slice(2, 3, 4);
    auto R_t = R.transpose(-1, -2);
    auto campos_lfs = R_t.bmm(t).mul(-1.0f).squeeze(-1);
    auto means_exp = means_lfs.unsqueeze(0);
    auto campos_exp = campos_lfs.unsqueeze(1);
    auto dirs_lfs = means_exp.sub(campos_exp);  // [1, N, 3]

    // Compare dirs
    torch::Tensor dirs_lfs_torch = lfs_to_torch(dirs_lfs).cpu();
    torch::Tensor dirs_leg_cpu = dirs_leg.cpu();
    torch::Tensor dirs_diff_t = (dirs_leg_cpu - dirs_lfs_torch).abs();
    float dirs_diff = dirs_diff_t.max().item<float>();
    std::cout << "Dirs max diff: " << dirs_diff << std::endl;

    // Compare first few dirs
    std::cout << "First 3 dirs (legacy): " << dirs_leg_cpu[0].slice(0, 0, 3) << std::endl;
    std::cout << "First 3 dirs (LFS): " << dirs_lfs_torch[0].slice(0, 0, 3) << std::endl;

    // Check SH coefficients directly
    std::cout << "\nSH coefficients (first gaussian):" << std::endl;
    std::cout << "Legacy sh_coeffs[0]: " << sh_coeffs_leg[0] << std::endl;
    auto sh_lfs_torch = lfs_to_torch(sh_coeffs_lfs).cpu();
    std::cout << "LFS sh_coeffs[0]: " << sh_lfs_torch[0] << std::endl;

    // Manual SH0 calculation check:
    // For SH degree 0: color = 0.28209479... * sh0 + 0.5
    float sh0_coeff = 0.2820947917738781f;
    std::cout << "\nManual SH0 check for first gaussian:" << std::endl;
    std::cout << "  sh0_rgb = " << sh_coeffs_leg[0] << std::endl;
    auto sh0_vals = sh_coeffs_leg[0][0].cpu();  // [3] RGB
    std::cout << "  Expected colors (SH0 only): ";
    for (int c = 0; c < 3; ++c) {
        float sh0_val = sh0_vals[c].item<float>();
        float expected = std::max(0.0f, sh0_coeff * sh0_val + 0.5f);
        std::cout << expected << " ";
    }
    std::cout << std::endl;

    EXPECT_LT(dirs_diff, 1e-5f) << "Dirs calculation differs!";
}

// Compare backward pass gradients
TEST_F(GsplatLegacyVsLfsTest, BackwardGradientComparison) {
    std::cout << "\n=== Backward Gradient Comparison (Legacy vs LFS gsplat) ===" << std::endl;

    // ========== LEGACY BACKWARD ==========
    // Create fresh tensors, apply operations, then set requires_grad
    const int64_t N = 1000;
    torch::manual_seed(42);

    // Create raw tensors first, then enable grad tracking
    auto means_grad = (torch::randn({N, 3}, torch::kCUDA) * 2.0f).clone().detach().set_requires_grad(true);
    auto sh0_grad = (torch::randn({N, 1, 3}, torch::kCUDA) * 0.5f + 0.5f).clone().detach().set_requires_grad(true);
    auto shN_grad = torch::zeros({N, 0, 3}, torch::kCUDA);
    auto scaling_grad = (torch::randn({N, 3}, torch::kCUDA) * 0.5f - 3.0f).clone().detach().set_requires_grad(true);
    auto rotation_grad = torch::randn({N, 4}, torch::kCUDA).clone().detach().set_requires_grad(true);
    auto opacity_grad = torch::randn({N, 1}, torch::kCUDA).clone().detach().set_requires_grad(true);

    // Create legacy splat data with requires_grad tensors
    auto legacy_splat_grad = std::make_unique<LegacySplatData>(
        0,  // sh_degree
        means_grad,
        sh0_grad,
        shN_grad,
        scaling_grad,
        rotation_grad,
        opacity_grad,
        1.0f
    );

    // Run legacy forward
    auto legacy_output = gs::training::rasterize(
        *legacy_camera_,
        *legacy_splat_grad,
        bg_color_torch_,
        1.0f, false, false,
        gs::training::RenderMode::RGB
    );

    // Backward with sum of all pixels
    auto loss_legacy = legacy_output.image.sum();
    loss_legacy.backward();

    // Get legacy gradients (these are gradients w.r.t. raw parameters)
    auto legacy_grad_means = means_grad.grad().clone();
    auto legacy_grad_scaling = scaling_grad.grad().clone();
    auto legacy_grad_rotation = rotation_grad.grad().clone();
    auto legacy_grad_opacity = opacity_grad.grad().clone();
    auto legacy_grad_sh0 = sh0_grad.grad().clone();

    std::cout << "Legacy grad_means: mean=" << legacy_grad_means.mean().item<float>()
              << ", max=" << legacy_grad_means.abs().max().item<float>() << std::endl;
    std::cout << "Legacy grad_scaling: mean=" << legacy_grad_scaling.mean().item<float>()
              << ", max=" << legacy_grad_scaling.abs().max().item<float>() << std::endl;
    std::cout << "Legacy grad_opacity: mean=" << legacy_grad_opacity.mean().item<float>()
              << ", max=" << legacy_grad_opacity.abs().max().item<float>() << std::endl;

    // ========== LFS BACKWARD ==========
    // Create LFS optimizer
    lfs::training::AdamConfig config;
    config.initial_capacity = lfs_splat_data_->size();
    lfs::training::AdamOptimizer lfs_optimizer(*lfs_splat_data_, config);
    lfs_optimizer.allocate_gradients();

    // Run LFS forward
    auto lfs_result = lfs::training::gsplat_rasterize_forward(
        *lfs_camera_,
        *lfs_splat_data_,
        bg_color_lfs_,
        1.0f, false,
        lfs::training::GsplatRenderMode::RGB,
        false  // use_gut
    );
    ASSERT_TRUE(lfs_result.has_value()) << "LFS forward failed: " << lfs_result.error();
    auto& [lfs_output, lfs_ctx] = lfs_result.value();

    // Create gradient (all ones, same as sum().backward())
    auto grad_image = lfs::core::Tensor::ones(lfs_output.image.shape(), lfs::core::Device::CUDA, lfs::core::DataType::Float32);
    auto grad_alpha = lfs::core::Tensor::zeros(lfs_output.alpha.shape(), lfs::core::Device::CUDA, lfs::core::DataType::Float32);

    // Run LFS backward
    lfs_optimizer.zero_grad(0);
    lfs::training::gsplat_rasterize_backward(lfs_ctx, grad_image, grad_alpha, *lfs_splat_data_, lfs_optimizer);

    // Get LFS gradients
    auto lfs_grad_means = lfs_to_torch(lfs_optimizer.get_grad(lfs::training::ParamType::Means));
    auto lfs_grad_scaling = lfs_to_torch(lfs_optimizer.get_grad(lfs::training::ParamType::Scaling));
    auto lfs_grad_rotation = lfs_to_torch(lfs_optimizer.get_grad(lfs::training::ParamType::Rotation));
    auto lfs_grad_opacity = lfs_to_torch(lfs_optimizer.get_grad(lfs::training::ParamType::Opacity));
    auto lfs_grad_sh0 = lfs_to_torch(lfs_optimizer.get_grad(lfs::training::ParamType::Sh0));

    std::cout << "LFS grad_means: mean=" << lfs_grad_means.mean().item<float>()
              << ", max=" << lfs_grad_means.abs().max().item<float>() << std::endl;
    std::cout << "LFS grad_scaling: mean=" << lfs_grad_scaling.mean().item<float>()
              << ", max=" << lfs_grad_scaling.abs().max().item<float>() << std::endl;
    std::cout << "LFS grad_opacity: mean=" << lfs_grad_opacity.mean().item<float>()
              << ", max=" << lfs_grad_opacity.abs().max().item<float>() << std::endl;

    // ========== COMPARE GRADIENTS ==========
    auto compare_grads = [](const torch::Tensor& grad1, const torch::Tensor& grad2, const char* name) {
        // Ensure same shape
        auto g1 = grad1.contiguous();
        auto g2 = grad2.contiguous();

        if (g1.sizes() != g2.sizes()) {
            // Try to reshape
            if (g1.numel() == g2.numel()) {
                g2 = g2.view(g1.sizes());
            } else {
                std::cout << name << ": SHAPE MISMATCH - legacy " << g1.sizes()
                          << " vs lfs " << g2.sizes() << std::endl;
                return 1.0f;
            }
        }

        auto diff = (g1 - g2).abs();
        float max_diff = diff.max().item<float>();
        float mean_diff = diff.mean().item<float>();

        float g1_max = g1.abs().max().item<float>();
        float g2_max = g2.abs().max().item<float>();
        float scale = std::max(g1_max, g2_max);
        float rel_diff = max_diff / (scale + 1e-8f);

        std::cout << name << ": max_diff=" << max_diff
                  << ", mean_diff=" << mean_diff
                  << ", rel_diff=" << rel_diff
                  << " (legacy_max=" << g1_max << ", lfs_max=" << g2_max << ")" << std::endl;

        return rel_diff;
    };

    float means_rel = compare_grads(legacy_grad_means, lfs_grad_means, "Means grad");
    float scaling_rel = compare_grads(legacy_grad_scaling, lfs_grad_scaling, "Scaling grad");
    float rotation_rel = compare_grads(legacy_grad_rotation, lfs_grad_rotation, "Rotation grad");
    float opacity_rel = compare_grads(legacy_grad_opacity, lfs_grad_opacity, "Opacity grad");
    float sh0_rel = compare_grads(legacy_grad_sh0, lfs_grad_sh0, "Sh0 grad");

    // Clean up arena
    auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
    arena.end_frame(lfs_ctx.frame_id);

    // Gradients should be very close since it's the same implementation
    EXPECT_LT(means_rel, 0.1f) << "Means gradients differ significantly";
    EXPECT_LT(scaling_rel, 0.1f) << "Scaling gradients differ significantly";
    EXPECT_LT(rotation_rel, 0.1f) << "Rotation gradients differ significantly";
    EXPECT_LT(opacity_rel, 0.1f) << "Opacity gradients differ significantly";
}

// Test activated parameters match between legacy and LFS
TEST_F(GsplatLegacyVsLfsTest, ActivatedParameterComparison) {
    std::cout << "\n=== Activated Parameter Comparison ===" << std::endl;

    // Legacy activated parameters (use the member variables from SetUp)
    auto legacy_scales = legacy_splat_data_->get_scaling();  // exp(raw)
    auto legacy_opacity = legacy_splat_data_->get_opacity(); // sigmoid(raw)
    auto legacy_quats = legacy_splat_data_->get_rotation();  // normalize(raw)

    // LFS activated parameters
    auto lfs_scales_t = lfs_to_torch(lfs_splat_data_->get_scaling());
    auto lfs_opacity_t = lfs_to_torch(lfs_splat_data_->get_opacity());
    auto lfs_quats_t = lfs_to_torch(lfs_splat_data_->get_rotation());

    // Compare scales
    float scale_diff = (legacy_scales.cpu() - lfs_scales_t.cpu()).abs().max().item<float>();
    std::cout << "Scale activation diff: " << scale_diff << std::endl;
    EXPECT_LT(scale_diff, 1e-5f);

    // Opacity might have different shapes
    auto lfs_op = lfs_opacity_t.cpu();
    auto leg_op = legacy_opacity.cpu();
    if (lfs_op.dim() != leg_op.dim()) {
        if (lfs_op.dim() == 1) lfs_op = lfs_op.unsqueeze(-1);
        if (leg_op.dim() == 1) leg_op = leg_op.unsqueeze(-1);
    }
    float opacity_diff = (leg_op - lfs_op).abs().max().item<float>();
    std::cout << "Opacity activation diff: " << opacity_diff << std::endl;
    EXPECT_LT(opacity_diff, 1e-5f);

    float quat_diff = (legacy_quats.cpu() - lfs_quats_t.cpu()).abs().max().item<float>();
    std::cout << "Quaternion activation diff: " << quat_diff << std::endl;
    EXPECT_LT(quat_diff, 1e-5f);
}
