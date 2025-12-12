// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * @file test_gsplat_training_debug.cpp
 * @brief Compare Legacy (LibTorch) gsplat vs LFS (LibTorch-free) gsplat rasterizers
 *
 * This test loads the same dataset with both loaders, renders using gsplat
 * rasterizers from both implementations, and compares:
 * - Forward pass outputs (rendered images)
 * - Backward pass gradients
 * - Loss values
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
#include "optimizers/fused_adam.hpp"

// New modules (LibTorch-free)
#include "loader_new/loader.hpp"
#include "loader_new/cache_image_loader.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/point_cloud.hpp"
#include "core_new/parameters.hpp"
#include "core_new/camera.hpp"

// Include dataset headers
#include "../src/training/dataset.hpp"
#include "../src/training_new/dataset.hpp"

// Include strategy headers
#include "../src/training/strategies/mcmc.hpp"
#include "../src/training_new/strategies/mcmc.hpp"

// Include render output structs
#include "../src/training/rasterization/rasterizer.hpp"
#include "../src/training_new/optimizer/render_output.hpp"

// Include Legacy gsplat Ops header FIRST to define CameraModelType, ShutterType, etc
// at global scope before LFS Common.h is included
#include "Ops.h"  // Legacy gsplat:: namespace (via gsplat/ in include path)

// Include LFS gsplat Ops (guards prevent type redefinition)
#include "rasterization/gsplat/Ops.h"  // lfs::gsplat:: namespace

// Include gsplat rasterizers - BOTH Legacy and LFS
#include "../src/training/rasterization/rasterizer.hpp"       // gs::training::rasterize (uses gsplat)
#include "../src/training_new/rasterization/gsplat_rasterizer.hpp"  // lfs::training::gsplat_rasterize_forward

// Include losses
#include "../src/training_new/losses/photometric_loss.hpp"
#include "kernels/fused_ssim.cuh"

class GsplatTrainingDebugTest : public ::testing::Test {
protected:
    void SetUp() override {
        spdlog::set_level(spdlog::level::info);
    }
};

// Shared initialization result structs
struct LegacyInit {
    std::shared_ptr<gs::training::CameraDataset> dataset;
    std::shared_ptr<gs::training::MCMC> strategy;
    gs::param::TrainingParameters params;
    torch::Tensor background;
    std::unordered_map<size_t, std::shared_ptr<gs::Camera>> cam_id_to_cam;
    size_t num_gaussians = 0;
};

struct NewInit {
    std::shared_ptr<lfs::training::CameraDataset> dataset;
    std::unique_ptr<lfs::core::SplatData> splat_data;  // Must outlive strategy
    std::shared_ptr<lfs::training::MCMC> strategy;
    lfs::core::param::TrainingParameters params;
    lfs::core::Tensor background;
    std::unordered_map<size_t, std::shared_ptr<lfs::core::Camera>> cam_id_to_cam;
    size_t num_gaussians = 0;
};

// Helper to initialize both pipelines
std::pair<LegacyInit, NewInit> initialize_both_gsplat() {
    spdlog::info("=== Initializing BOTH pipelines for gsplat comparison ===");

    LegacyInit legacy;
    NewInit new_impl;

    // Setup parameters - USE BICYCLE DATASET
    legacy.params.dataset.data_path = "data/bicycle";
    legacy.params.dataset.images = "images_4";
    legacy.params.dataset.resize_factor = -1;
    legacy.params.dataset.max_width = 3840;
    legacy.params.optimization.strategy = "mcmc";
    legacy.params.optimization.max_cap = 1000000;
    legacy.params.optimization.sh_degree = 0;  // SH0 only for simplicity

    new_impl.params.dataset.data_path = "data/bicycle";
    new_impl.params.dataset.images = "images_4";
    new_impl.params.dataset.resize_factor = -1;
    new_impl.params.dataset.max_width = 3840;
    new_impl.params.optimization.strategy = "mcmc";
    new_impl.params.optimization.max_cap = 1000000;
    new_impl.params.optimization.sh_degree = 0;

    // === LEGACY INITIALIZATION ===
    spdlog::info("[LEGACY] Creating loader...");
    auto legacy_loader = gs::loader::Loader::create();
    EXPECT_TRUE(legacy_loader != nullptr);

    auto& legacy_cache_loader = gs::loader::CacheLoader::getInstance(
        legacy.params.dataset.loading_params.use_cpu_memory,
        legacy.params.dataset.loading_params.use_fs_cache
    );

    gs::loader::LoadOptions legacy_load_options{
        .resize_factor = legacy.params.dataset.resize_factor,
        .max_width = legacy.params.dataset.max_width,
        .images_folder = legacy.params.dataset.images,
        .validate_only = false,
    };

    spdlog::info("[LEGACY] Loading dataset...");
    auto legacy_load_result = legacy_loader->load(legacy.params.dataset.data_path, legacy_load_options);
    EXPECT_TRUE(legacy_load_result.has_value());

    // Handle legacy loaded data
    std::visit([&legacy, &legacy_load_result, &legacy_cache_loader](auto&& data) {
        using T = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<T, gs::loader::LoadedScene>) {
            gs::PointCloud point_cloud_to_use;
            if (data.point_cloud && data.point_cloud->size() > 0) {
                point_cloud_to_use = *data.point_cloud;
            }

            auto splat_result = gs::SplatData::init_model_from_pointcloud(
                legacy.params, legacy_load_result->scene_center, point_cloud_to_use);
            EXPECT_TRUE(splat_result.has_value());

            legacy.num_gaussians = splat_result->size();
            legacy.strategy = std::make_shared<gs::training::MCMC>(std::move(*splat_result));
            legacy.strategy->get_model().set_active_sh_degree(legacy.params.optimization.sh_degree);
            legacy.strategy->initialize(legacy.params.optimization);
            legacy.background = torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            legacy.dataset = data.cameras;

            for (const auto& cam : data.cameras->get_cameras()) {
                legacy.cam_id_to_cam[cam->uid()] = cam;
            }

            legacy_cache_loader.update_cache_params(
                legacy.params.dataset.loading_params.use_cpu_memory,
                legacy.params.dataset.loading_params.use_fs_cache,
                static_cast<int>(data.cameras->size().value())
            );
        }
    }, legacy_load_result->data);

    spdlog::info("[LEGACY] Initialized {} Gaussians", legacy.num_gaussians);

    // === NEW INITIALIZATION ===
    spdlog::info("[NEW] Creating loader...");
    auto new_loader = lfs::loader::Loader::create();
    EXPECT_TRUE(new_loader != nullptr);

    auto& new_cache_loader = lfs::loader::CacheLoader::getInstance(
        new_impl.params.dataset.loading_params.use_cpu_memory,
        new_impl.params.dataset.loading_params.use_fs_cache
    );

    lfs::loader::LoadOptions new_load_options{
        .resize_factor = new_impl.params.dataset.resize_factor,
        .max_width = new_impl.params.dataset.max_width,
        .images_folder = new_impl.params.dataset.images,
        .validate_only = false,
    };

    spdlog::info("[NEW] Loading dataset...");
    auto new_load_result = new_loader->load(new_impl.params.dataset.data_path, new_load_options);
    EXPECT_TRUE(new_load_result.has_value());

    // Handle new loaded data
    std::visit([&new_impl, &new_load_result](auto&& data) {
        using T = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<T, lfs::loader::LoadedScene>) {
            lfs::core::PointCloud point_cloud_to_use;
            if (data.point_cloud && data.point_cloud->size() > 0) {
                point_cloud_to_use = *data.point_cloud;
            }

            auto splat_result = lfs::core::init_model_from_pointcloud(
                new_impl.params, new_load_result->scene_center, point_cloud_to_use);
            EXPECT_TRUE(splat_result.has_value());

            new_impl.num_gaussians = splat_result->size();
            new_impl.splat_data = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
            new_impl.strategy = std::make_shared<lfs::training::MCMC>(*new_impl.splat_data);
            new_impl.strategy->get_model().set_active_sh_degree(new_impl.params.optimization.sh_degree);
            new_impl.strategy->initialize(new_impl.params.optimization);
            new_impl.background = lfs::core::Tensor::zeros({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            new_impl.dataset = data.cameras;

            for (const auto& cam : data.cameras->get_cameras()) {
                new_impl.cam_id_to_cam[cam->uid()] = cam;
            }
        }
    }, new_load_result->data);

    spdlog::info("[NEW] Initialized {} Gaussians", new_impl.num_gaussians);

    EXPECT_EQ(legacy.num_gaussians, new_impl.num_gaussians);

    return {std::move(legacy), std::move(new_impl)};
}

/**
 * @brief Compare gsplat forward pass between Legacy and LFS
 */
TEST_F(GsplatTrainingDebugTest, GsplatForwardComparison) {
    spdlog::info("=== Starting GsplatForwardComparison test ===");

    auto [legacy, new_impl] = initialize_both_gsplat();

    // Get cameras
    auto legacy_cameras = legacy.dataset->get_cameras();
    auto new_cameras = new_impl.dataset->get_cameras();

    ASSERT_GT(legacy_cameras.size(), 0);
    ASSERT_GT(new_cameras.size(), 0);

    auto legacy_cam = legacy_cameras[0];
    auto new_cam = new_cameras[0];

    // DEBUG: Compare camera and model parameters
    {
        auto legacy_viewmat = legacy_cam->world_view_transform().cpu();
        auto new_viewmat = new_cam->world_view_transform().to(lfs::core::Device::CPU);

        // Convert new_viewmat to torch for comparison
        std::vector<long> vshape;
        for (size_t i = 0; i < new_viewmat.ndim(); ++i) {
            vshape.push_back(static_cast<long>(new_viewmat.shape()[i]));
        }
        torch::Tensor new_viewmat_torch = torch::from_blob(
            const_cast<float*>(new_viewmat.ptr<float>()),
            torch::IntArrayRef(vshape),
            torch::kFloat32).clone();

        auto viewmat_diff = (legacy_viewmat - new_viewmat_torch).abs().max().item<float>();
        spdlog::info("[DEBUG] Viewmat max diff: {:.6e}", viewmat_diff);
        spdlog::info("[DEBUG] Legacy viewmat[0,0,:]: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                     legacy_viewmat[0][0][0].item<float>(), legacy_viewmat[0][0][1].item<float>(),
                     legacy_viewmat[0][0][2].item<float>(), legacy_viewmat[0][0][3].item<float>());
        spdlog::info("[DEBUG] New viewmat[0,0,:]: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                     new_viewmat_torch[0][0][0].item<float>(), new_viewmat_torch[0][0][1].item<float>(),
                     new_viewmat_torch[0][0][2].item<float>(), new_viewmat_torch[0][0][3].item<float>());

        // Compare model means
        auto legacy_means = legacy.strategy->get_model().get_means().cpu();
        auto new_means_lfs = new_impl.strategy->get_model().get_means().to(lfs::core::Device::CPU);
        std::vector<long> mshape = {static_cast<long>(new_means_lfs.shape()[0]),
                                    static_cast<long>(new_means_lfs.shape()[1])};
        torch::Tensor new_means_torch = torch::from_blob(
            const_cast<float*>(new_means_lfs.ptr<float>()),
            torch::IntArrayRef(mshape),
            torch::kFloat32).clone();

        auto means_diff = (legacy_means - new_means_torch).abs().max().item<float>();
        spdlog::info("[DEBUG] Model means max diff: {:.6e}", means_diff);
        spdlog::info("[DEBUG] Legacy means[0,:]: [{:.6f}, {:.6f}, {:.6f}]",
                     legacy_means[0][0].item<float>(), legacy_means[0][1].item<float>(),
                     legacy_means[0][2].item<float>());
        spdlog::info("[DEBUG] New means[0,:]: [{:.6f}, {:.6f}, {:.6f}]",
                     new_means_torch[0][0].item<float>(), new_means_torch[0][1].item<float>(),
                     new_means_torch[0][2].item<float>());

        // Compare SH coefficients
        auto legacy_shs = legacy.strategy->get_model().get_shs().cpu();
        auto new_shs_lfs = new_impl.strategy->get_model().get_shs().to(lfs::core::Device::CPU);
        std::vector<long> sshape = {static_cast<long>(new_shs_lfs.shape()[0]),
                                    static_cast<long>(new_shs_lfs.shape()[1]),
                                    static_cast<long>(new_shs_lfs.shape()[2])};
        torch::Tensor new_shs_torch = torch::from_blob(
            const_cast<float*>(new_shs_lfs.ptr<float>()),
            torch::IntArrayRef(sshape),
            torch::kFloat32).clone();

        auto shs_diff = (legacy_shs - new_shs_torch).abs().max().item<float>();
        spdlog::info("[DEBUG] Model SH max diff: {:.6e}", shs_diff);
        spdlog::info("[DEBUG] Legacy SH[0,0,:]: [{:.6f}, {:.6f}, {:.6f}]",
                     legacy_shs[0][0][0].item<float>(), legacy_shs[0][0][1].item<float>(),
                     legacy_shs[0][0][2].item<float>());
        spdlog::info("[DEBUG] New SH[0,0,:]: [{:.6f}, {:.6f}, {:.6f}]",
                     new_shs_torch[0][0][0].item<float>(), new_shs_torch[0][0][1].item<float>(),
                     new_shs_torch[0][0][2].item<float>());

        // Also compare other model parameters
        auto legacy_opacities = legacy.strategy->get_model().get_opacity().cpu();
        auto new_opac_lfs = new_impl.strategy->get_model().get_opacity().to(lfs::core::Device::CPU);
        spdlog::info("[DEBUG] Legacy opacity[0]: {:.6f}", legacy_opacities[0].item<float>());
        spdlog::info("[DEBUG] New opacity[0]: {:.6f}", new_opac_lfs.ptr<float>()[0]);

        auto legacy_scales = legacy.strategy->get_model().get_scaling().cpu();
        auto new_scales_lfs = new_impl.strategy->get_model().get_scaling().to(lfs::core::Device::CPU);
        spdlog::info("[DEBUG] Legacy scales[0]: [{:.6f}, {:.6f}, {:.6f}]",
                     legacy_scales[0][0].item<float>(), legacy_scales[0][1].item<float>(),
                     legacy_scales[0][2].item<float>());
        spdlog::info("[DEBUG] New scales[0]: [{:.6f}, {:.6f}, {:.6f}]",
                     new_scales_lfs.ptr<float>()[0], new_scales_lfs.ptr<float>()[1],
                     new_scales_lfs.ptr<float>()[2]);

        auto legacy_quats = legacy.strategy->get_model().get_rotation().cpu();
        auto new_quats_lfs = new_impl.strategy->get_model().get_rotation().to(lfs::core::Device::CPU);
        spdlog::info("[DEBUG] Legacy quats[0]: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                     legacy_quats[0][0].item<float>(), legacy_quats[0][1].item<float>(),
                     legacy_quats[0][2].item<float>(), legacy_quats[0][3].item<float>());
        spdlog::info("[DEBUG] New quats[0]: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                     new_quats_lfs.ptr<float>()[0], new_quats_lfs.ptr<float>()[1],
                     new_quats_lfs.ptr<float>()[2], new_quats_lfs.ptr<float>()[3]);
    }

    spdlog::info("Rendering camera 0 using GSPLAT rasterizers...");

    // === LEGACY GSPLAT FORWARD ===
    spdlog::info("[LEGACY] Calling gs::training::rasterize (gsplat)...");
    auto legacy_output = gs::training::rasterize(
        *legacy_cam,
        legacy.strategy->get_model(),
        legacy.background,
        1.0f,           // scaling_modifier
        false,          // packed
        false,          // antialiased
        gs::training::RenderMode::RGB
    );
    spdlog::info("[LEGACY] Rendered {}x{}", legacy_output.width, legacy_output.height);
    spdlog::info("[LEGACY] Image defined: {}, numel: {}",
                 legacy_output.image.defined(),
                 legacy_output.image.defined() ? legacy_output.image.numel() : 0);
    if (legacy_output.image.defined() && legacy_output.image.numel() > 0) {
        spdlog::info("[LEGACY] Image shape: [{}, {}, {}]",
                     legacy_output.image.size(0), legacy_output.image.size(1), legacy_output.image.size(2));
        spdlog::info("[LEGACY] Image min: {}, max: {}",
                     legacy_output.image.min().item<float>(), legacy_output.image.max().item<float>());
    }

    // === NEW GSPLAT FORWARD ===
    spdlog::info("[NEW] Calling lfs::training::gsplat_rasterize_forward...");
    auto new_result = lfs::training::gsplat_rasterize_forward(
        *new_cam,
        new_impl.strategy->get_model(),
        new_impl.background,
        1.0f,           // scaling_modifier
        false,          // antialiased
        lfs::training::GsplatRenderMode::RGB,
        false           // use_gut
    );
    ASSERT_TRUE(new_result.has_value()) << "LFS gsplat forward failed: " << new_result.error();
    auto [new_output, new_ctx] = new_result.value();
    spdlog::info("[NEW] Rendered {}x{}", new_output.width, new_output.height);

    // === COMPARE RENDERED IMAGES ===
    spdlog::info("=== Comparing Rendered Images ===");

    auto legacy_img = legacy_output.image.cpu();
    auto new_img_cpu = new_output.image.to(lfs::core::Device::CPU);

    // Convert new image to torch for comparison
    std::vector<long> img_shape;
    for (size_t i = 0; i < new_img_cpu.ndim(); ++i) {
        img_shape.push_back(static_cast<long>(new_img_cpu.shape()[i]));
    }
    torch::Tensor new_img_torch = torch::from_blob(
        const_cast<float*>(new_img_cpu.ptr<float>()),
        torch::IntArrayRef(img_shape),
        torch::kFloat32).clone();

    spdlog::info("Legacy image shape: [{}, {}, {}]",
                 legacy_img.size(0), legacy_img.size(1), legacy_img.size(2));
    spdlog::info("New image shape: [{}, {}, {}]",
                 new_img_torch.size(0), new_img_torch.size(1), new_img_torch.size(2));
    spdlog::info("[NEW] Image min: {}, max: {}",
                 new_img_torch.min().item<float>(), new_img_torch.max().item<float>());

    // Compare center pixel
    int mid_y = legacy_img.size(1) / 2;
    int mid_x = legacy_img.size(2) / 2;
    int H = legacy_img.size(1);
    int W = legacy_img.size(2);

    spdlog::info("Image dims: H={}, W={}", H, W);

    // Check corners and center
    struct { int x, y; const char* name; } test_points[] = {
        {mid_x, mid_y, "Center"},
        {10, 10, "Top-Left"},
        {W-10, 10, "Top-Right"},
        {10, H-10, "Bottom-Left"},
        {W-10, H-10, "Bottom-Right"},
        {W/4, H/4, "Quarter"},
    };

    for (auto& pt : test_points) {
        float legacy_r = legacy_img[0][pt.y][pt.x].item<float>();
        float legacy_g = legacy_img[1][pt.y][pt.x].item<float>();
        float legacy_b = legacy_img[2][pt.y][pt.x].item<float>();
        float new_r = new_img_torch[0][pt.y][pt.x].item<float>();
        float new_g = new_img_torch[1][pt.y][pt.x].item<float>();
        float new_b = new_img_torch[2][pt.y][pt.x].item<float>();

        // Also check flipped location
        float new_flip_r = new_img_torch[0][H-1-pt.y][pt.x].item<float>();
        float new_flip_g = new_img_torch[1][H-1-pt.y][pt.x].item<float>();
        float new_flip_b = new_img_torch[2][H-1-pt.y][pt.x].item<float>();

        spdlog::info("{} ({},{}): Legacy=[{:.4f},{:.4f},{:.4f}] New=[{:.4f},{:.4f},{:.4f}] NewFlipY=[{:.4f},{:.4f},{:.4f}]",
                     pt.name, pt.x, pt.y,
                     legacy_r, legacy_g, legacy_b,
                     new_r, new_g, new_b,
                     new_flip_r, new_flip_g, new_flip_b);
    }

    // Compute differences
    auto img_diff = (legacy_img - new_img_torch).abs();
    float img_max_diff = img_diff.max().item<float>();
    float img_mean_diff = img_diff.mean().item<float>();

    spdlog::info("Rendered Image - Max diff: {:.6e}, Mean diff: {:.6e}", img_max_diff, img_mean_diff);

    // Per-channel analysis
    if (legacy_img.size(0) == 3) {
        for (int c = 0; c < 3; ++c) {
            float ch_max = img_diff[c].max().item<float>();
            float ch_mean = img_diff[c].mean().item<float>();
            const char* ch_name = (c == 0) ? "R" : (c == 1) ? "G" : "B";
            spdlog::info("  Channel {} - Max diff: {:.6e}, Mean diff: {:.6e}", ch_name, ch_max, ch_mean);
        }
    }

    // Show sample pixels
    spdlog::info("First 5 pixels comparison:");
    for (int i = 0; i < 5; ++i) {
        if (legacy_img.size(0) == 3) {
            // [3, H, W] format
            spdlog::info("  Pixel (0,{}): Legacy=[{:.6f}, {:.6f}, {:.6f}], New=[{:.6f}, {:.6f}, {:.6f}]",
                        i,
                        legacy_img[0][0][i].item<float>(),
                        legacy_img[1][0][i].item<float>(),
                        legacy_img[2][0][i].item<float>(),
                        new_img_torch[0][0][i].item<float>(),
                        new_img_torch[1][0][i].item<float>(),
                        new_img_torch[2][0][i].item<float>());
        }
    }

    // Tolerance check
    float tolerance = 0.01f;  // 1% tolerance
    if (img_max_diff > tolerance) {
        spdlog::error("GSPLAT FORWARD MISMATCH! Max diff {} > tolerance {}", img_max_diff, tolerance);

        // Find where the max difference is
        auto max_idx = img_diff.argmax();
        spdlog::error("Max diff location (flat index): {}", max_idx.item<int>());
    }

    EXPECT_LT(img_max_diff, tolerance) << "Gsplat forward pass differs too much!";

    // Release arena frame
    auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
    arena.end_frame(new_ctx.frame_id);
}

/**
 * @brief Compare gsplat training loop (forward + backward + optimizer step)
 */
TEST_F(GsplatTrainingDebugTest, GsplatTrainingLoopComparison) {
    spdlog::info("=== Starting GsplatTrainingLoopComparison test ===");

    auto [legacy, new_impl] = initialize_both_gsplat();

    // Get cameras and GT
    auto legacy_cameras = legacy.dataset->get_cameras();
    auto new_cameras = new_impl.dataset->get_cameras();
    auto legacy_cam = legacy_cameras[0];
    auto new_cam = new_cameras[0];

    auto legacy_gt = legacy_cam->load_and_get_image(
        legacy.params.dataset.resize_factor,
        legacy.params.dataset.max_width);
    auto new_gt = new_cam->load_and_get_image(
        new_impl.params.dataset.resize_factor,
        new_impl.params.dataset.max_width);

    // CRITICAL: Compare GT images to ensure loss inputs are identical
    {
        spdlog::info("=== Comparing GT Images (Loss Inputs) ===");
        auto legacy_gt_cpu = legacy_gt.cpu();
        auto new_gt_cpu = new_gt.to(lfs::core::Device::CPU);

        spdlog::info("  Legacy GT shape: [{}, {}, {}]", legacy_gt_cpu.size(0), legacy_gt_cpu.size(1), legacy_gt_cpu.size(2));
        spdlog::info("  New GT shape: [{}, {}, {}]", new_gt_cpu.shape()[0], new_gt_cpu.shape()[1], new_gt_cpu.shape()[2]);

        ASSERT_EQ(legacy_gt_cpu.size(0), static_cast<int64_t>(new_gt_cpu.shape()[0])) << "GT channel mismatch";
        ASSERT_EQ(legacy_gt_cpu.size(1), static_cast<int64_t>(new_gt_cpu.shape()[1])) << "GT height mismatch";
        ASSERT_EQ(legacy_gt_cpu.size(2), static_cast<int64_t>(new_gt_cpu.shape()[2])) << "GT width mismatch";

        // Convert new_gt to torch for comparison
        std::vector<long> gt_shape;
        for (size_t i = 0; i < new_gt_cpu.ndim(); ++i) {
            gt_shape.push_back(static_cast<long>(new_gt_cpu.shape()[i]));
        }
        torch::Tensor new_gt_torch = torch::from_blob(
            const_cast<float*>(new_gt_cpu.ptr<float>()),
            torch::IntArrayRef(gt_shape), torch::kFloat32).clone();

        auto gt_diff = (legacy_gt_cpu - new_gt_torch).abs();
        float gt_max_diff = gt_diff.max().item<float>();
        float gt_mean_diff = gt_diff.mean().item<float>();

        spdlog::info("  GT image max diff: {:.6e}", gt_max_diff);
        spdlog::info("  GT image mean diff: {:.6e}", gt_mean_diff);
        spdlog::info("  Legacy GT range: [{:.4f}, {:.4f}]", legacy_gt_cpu.min().item<float>(), legacy_gt_cpu.max().item<float>());
        spdlog::info("  New GT range: [{:.4f}, {:.4f}]", new_gt_torch.min().item<float>(), new_gt_torch.max().item<float>());

        // GT images should be nearly identical (small differences from float precision)
        ASSERT_LT(gt_max_diff, 1e-3f) << "GT images differ significantly! This will cause gradient divergence.";
    }

    const int max_iterations = 3;

    for (int iter = 1; iter <= max_iterations; ++iter) {
        spdlog::info("");
        spdlog::info("=== Iteration {} / {} ===", iter, max_iterations);

        // === FORWARD PASS ===
        spdlog::info("[{}] Forward pass (gsplat)...", iter);

        // Legacy forward
        auto legacy_output = gs::training::rasterize(
            *legacy_cam,
            legacy.strategy->get_model(),
            legacy.background,
            1.0f, false, false, gs::training::RenderMode::RGB);

        // New forward
        auto new_result = lfs::training::gsplat_rasterize_forward(
            *new_cam,
            new_impl.strategy->get_model(),
            new_impl.background,
            1.0f, false, lfs::training::GsplatRenderMode::RGB, false);
        ASSERT_TRUE(new_result.has_value()) << "LFS gsplat forward failed: " << new_result.error();
        auto [new_output, new_ctx] = new_result.value();

        // Compare rendered images
        auto legacy_img = legacy_output.image.cpu();
        auto new_img_cpu = new_output.image.to(lfs::core::Device::CPU);
        std::vector<long> img_shape;
        for (size_t i = 0; i < new_img_cpu.ndim(); ++i) {
            img_shape.push_back(static_cast<long>(new_img_cpu.shape()[i]));
        }
        torch::Tensor new_img_torch = torch::from_blob(
            const_cast<float*>(new_img_cpu.ptr<float>()),
            torch::IntArrayRef(img_shape), torch::kFloat32).clone();

        auto img_diff = (legacy_img - new_img_torch).abs();
        float img_max_diff = img_diff.max().item<float>();
        float img_mean_diff = img_diff.mean().item<float>();
        spdlog::info("[{}] Rendered Image - Max diff: {:.6e}, Mean diff: {:.6e}",
                     iter, img_max_diff, img_mean_diff);

        // === COMPUTE LOSS ===
        spdlog::info("[{}] Computing loss...", iter);

        // Legacy loss
        torch::Tensor legacy_rendered = legacy_output.image;
        legacy_rendered = legacy_rendered.dim() == 3 ? legacy_rendered.unsqueeze(0) : legacy_rendered;
        torch::Tensor legacy_gt_4d = legacy_gt.dim() == 3 ? legacy_gt.unsqueeze(0) : legacy_gt;

        auto legacy_l1_loss = torch::l1_loss(legacy_rendered, legacy_gt_4d);
        auto legacy_ssim_loss = 1.f - fused_ssim(legacy_rendered, legacy_gt_4d, "valid", true);
        torch::Tensor legacy_loss_tensor = (1.f - legacy.params.optimization.lambda_dssim) * legacy_l1_loss +
                                            legacy.params.optimization.lambda_dssim * legacy_ssim_loss;
        float legacy_loss_value = legacy_loss_tensor.item<float>();

        // New loss
        static lfs::training::losses::PhotometricLoss new_loss;
        lfs::training::losses::PhotometricLoss::Params loss_params{
            .lambda_dssim = new_impl.params.optimization.lambda_dssim
        };
        auto new_loss_result = new_loss.forward(new_output.image, new_gt, loss_params);
        ASSERT_TRUE(new_loss_result.has_value());
        auto [new_loss_tensor, new_loss_ctx] = *new_loss_result;
        float new_loss_value = new_loss_tensor.item();

        spdlog::info("[{}] Loss - Legacy: {:.6f}, New: {:.6f}, Diff: {:.6f}",
                     iter, legacy_loss_value, new_loss_value,
                     std::abs(legacy_loss_value - new_loss_value));

        // === BACKWARD PASS ===
        spdlog::info("[{}] Backward pass...", iter);

        // Allocate gradients if needed
        if (!new_impl.strategy->get_optimizer().has_gradients()) {
            new_impl.strategy->get_optimizer().allocate_gradients();
        } else {
            new_impl.strategy->get_optimizer().zero_grad(iter);
        }

        // Legacy backward
        legacy_loss_tensor.backward();

        // New backward
        // grad_alpha should be in CHW format [C=1, H, W] to match grad_image format
        lfs::core::Tensor grad_alpha = lfs::core::Tensor::zeros(
            {1, static_cast<int>(new_output.height), static_cast<int>(new_output.width)},
            lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        lfs::training::gsplat_rasterize_backward(
            new_ctx, new_loss_ctx.grad_image, grad_alpha,
            new_impl.strategy->get_model(), new_impl.strategy->get_optimizer());

        // Compare ALL gradients - means, scales, rotations, opacities, sh
        auto& legacy_model = legacy.strategy->get_model();
        auto& new_opt = new_impl.strategy->get_optimizer();

        // Helper to convert lfs tensor to torch for comparison
        auto to_torch_flat = [](const lfs::core::Tensor& t) -> torch::Tensor {
            auto t_cpu = t.to(lfs::core::Device::CPU);
            return torch::from_blob(
                t_cpu.ptr<float>(),
                {static_cast<long>(t_cpu.numel())},
                torch::kFloat32).clone();
        };

        // 1. Means gradients
        {
            auto legacy_grad = legacy_model.means().grad();
            auto new_grad = new_opt.get_grad(lfs::training::ParamType::Means);
            if (legacy_grad.defined() && legacy_grad.numel() > 0) {
                auto new_grad_torch = to_torch_flat(new_grad);
                auto grad_diff = (legacy_grad.cpu().flatten() - new_grad_torch).abs();
                float max_diff = grad_diff.max().item().toFloat();
                spdlog::info("[{}] Means Grad - Max diff: {:.2e}", iter, max_diff);
                EXPECT_LT(max_diff, 1e-4f) << "Means gradient mismatch at iter " << iter;
            }
        }

        // 2. Scales gradients (raw, before activation)
        {
            auto legacy_grad = legacy_model.scaling_raw().grad();
            auto new_grad = new_opt.get_grad(lfs::training::ParamType::Scaling);
            if (legacy_grad.defined() && legacy_grad.numel() > 0) {
                auto new_grad_torch = to_torch_flat(new_grad);
                auto grad_diff = (legacy_grad.cpu().flatten() - new_grad_torch).abs();
                float max_diff = grad_diff.max().item().toFloat();
                spdlog::info("[{}] Scaling Grad - Max diff: {:.2e}", iter, max_diff);
                EXPECT_LT(max_diff, 1e-4f) << "Scaling gradient mismatch at iter " << iter;
            }
        }

        // 3. Rotation gradients (raw quaternions)
        {
            auto legacy_grad = legacy_model.rotation_raw().grad();
            auto new_grad = new_opt.get_grad(lfs::training::ParamType::Rotation);
            if (legacy_grad.defined() && legacy_grad.numel() > 0) {
                auto new_grad_torch = to_torch_flat(new_grad);
                auto grad_diff = (legacy_grad.cpu().flatten() - new_grad_torch).abs();
                float max_diff = grad_diff.max().item().toFloat();
                spdlog::info("[{}] Rotation Grad - Max diff: {:.2e}", iter, max_diff);
                EXPECT_LT(max_diff, 1e-4f) << "Rotation gradient mismatch at iter " << iter;
            }
        }

        // 4. Opacity gradients (raw, before activation)
        {
            auto legacy_grad = legacy_model.opacity_raw().grad();
            auto new_grad = new_opt.get_grad(lfs::training::ParamType::Opacity);
            if (legacy_grad.defined() && legacy_grad.numel() > 0) {
                auto new_grad_torch = to_torch_flat(new_grad);
                auto grad_diff = (legacy_grad.cpu().flatten() - new_grad_torch).abs();
                float max_diff = grad_diff.max().item().toFloat();
                spdlog::info("[{}] Opacity Grad - Max diff: {:.2e}", iter, max_diff);
                EXPECT_LT(max_diff, 1e-4f) << "Opacity gradient mismatch at iter " << iter;
            }
        }

        // 5. SH0 gradients (slightly larger tolerance due to SH evaluation precision)
        {
            auto legacy_grad = legacy_model.sh0().grad();
            auto new_grad = new_opt.get_grad(lfs::training::ParamType::Sh0);
            if (legacy_grad.defined() && legacy_grad.numel() > 0) {
                auto new_grad_torch = to_torch_flat(new_grad);
                auto legacy_flat = legacy_grad.cpu().flatten();
                auto grad_diff = (legacy_flat - new_grad_torch).abs();
                float max_diff = grad_diff.max().item().toFloat();
                auto max_idx = grad_diff.argmax().item().toLong();
                spdlog::info("[{}] SH0 Grad - Max diff: {:.2e} at idx {}", iter, max_diff, max_idx);

                // Show actual gradient values at max diff location
                int gauss_idx = max_idx / 3;  // SH0 is [N, 1, 3], flattened is N*3
                int channel = max_idx % 3;
                spdlog::info("[{}] SH0 Grad Detail: gauss={}, ch={}, Legacy={:.6e}, New={:.6e}",
                    iter, gauss_idx, channel,
                    legacy_flat[max_idx].item<float>(),
                    new_grad_torch[max_idx].item<float>());

                // Show first few gradient values
                for (int i = 0; i < std::min(5, (int)legacy_flat.size(0)); i += 3) {
                    spdlog::info("[{}] SH0[{}] Legacy=[{:.6e},{:.6e},{:.6e}] New=[{:.6e},{:.6e},{:.6e}]",
                        iter, i/3,
                        legacy_flat[i].item<float>(), legacy_flat[i+1].item<float>(), legacy_flat[i+2].item<float>(),
                        new_grad_torch[i].item<float>(), new_grad_torch[i+1].item<float>(), new_grad_torch[i+2].item<float>());
                }

                EXPECT_LT(max_diff, 1e-3f) << "SH0 gradient mismatch at iter " << iter;
            }
        }

        // === OPTIMIZER STEP ===
        spdlog::info("[{}] Optimizer step...", iter);

        // Convert new output to legacy format for post_backward
        gs::training::RenderOutput legacy_compat_output;
        legacy_compat_output.width = new_output.width;
        legacy_compat_output.height = new_output.height;
        // Note: visibility comes from radii > 0

        legacy.strategy->post_backward(iter, legacy_output);
        new_impl.strategy->post_backward(iter, new_output);

        legacy.strategy->step(iter);
        new_impl.strategy->step(iter);

        // Compare updated parameters (reuse legacy_model from above)
        auto& new_model = new_impl.strategy->get_model();

        // Compare means
        auto legacy_means = legacy_model.means().cpu();
        auto new_means_cpu = new_model.means().to(lfs::core::Device::CPU);
        torch::Tensor new_means_torch = torch::from_blob(
            new_means_cpu.ptr<float>(),
            {static_cast<long>(new_means_cpu.shape()[0]),
             static_cast<long>(new_means_cpu.shape()[1])},
            torch::kFloat32).clone();

        auto means_diff = (legacy_means - new_means_torch).abs();
        spdlog::info("[{}] Means - Max diff: {:.2e}, Mean diff: {:.2e}",
                     iter, means_diff.max().item<float>(), means_diff.mean().item<float>());

        spdlog::info("[{}] Iteration complete", iter);
    }

    spdlog::info("=== GsplatTrainingLoopComparison test passed ===");
}

/**
 * @brief Extended 1000-iteration comparison with no regularization
 * - Same camera for all iterations
 * - No post_backward (no densification/pruning)
 * - No scale/opacity regularization
 */
TEST_F(GsplatTrainingDebugTest, GsplatExtended1000Iterations) {
    spdlog::info("=== Starting GsplatExtended1000Iterations test ===");

    // Initialize both pipelines
    auto [legacy, new_impl] = initialize_both_gsplat();

    // Get cameras and GT - use same camera throughout
    auto legacy_cameras = legacy.dataset->get_cameras();
    auto new_cameras = new_impl.dataset->get_cameras();
    auto legacy_cam = legacy_cameras[0];
    auto new_cam = new_cameras[0];

    auto legacy_gt = legacy_cam->load_and_get_image(
        legacy.params.dataset.resize_factor,
        legacy.params.dataset.max_width);
    auto new_gt = new_cam->load_and_get_image(
        new_impl.params.dataset.resize_factor,
        new_impl.params.dataset.max_width);

    // CRITICAL: Compare GT images to ensure loss inputs are identical
    {
        auto legacy_gt_cpu = legacy_gt.cpu();
        auto new_gt_cpu = new_gt.to(lfs::core::Device::CPU);

        // Convert new_gt to torch for comparison
        std::vector<long> gt_shape;
        for (size_t i = 0; i < new_gt_cpu.ndim(); ++i) {
            gt_shape.push_back(static_cast<long>(new_gt_cpu.shape()[i]));
        }
        torch::Tensor new_gt_torch = torch::from_blob(
            const_cast<float*>(new_gt_cpu.ptr<float>()),
            torch::IntArrayRef(gt_shape), torch::kFloat32).clone();

        auto gt_diff = (legacy_gt_cpu - new_gt_torch).abs();
        float gt_max_diff = gt_diff.max().item<float>();
        spdlog::info("  GT image max diff: {:.6e}", gt_max_diff);

        // GT images MUST be identical for valid comparison
        ASSERT_LT(gt_max_diff, 1e-3f) << "GT images differ! This invalidates the comparison.";
    }

    const int max_iterations = 1000;
    const int log_interval = 100;  // Log every 100 iterations

    // Track divergence over time
    float last_loss_legacy = 0.0f;
    float last_loss_new = 0.0f;

    // Helper to convert lfs tensor to torch
    auto to_torch_flat = [](const lfs::core::Tensor& t) -> torch::Tensor {
        auto t_cpu = t.to(lfs::core::Device::CPU);
        return torch::from_blob(
            t_cpu.ptr<float>(),
            {static_cast<long>(t_cpu.numel())},
            torch::kFloat32).clone();
    };

    // === COMPARE INITIAL PARAMETERS (before any optimization) ===
    {
        spdlog::info("=== Initial Model Comparison (iteration 0) ===");
        auto& legacy_model = legacy.strategy->get_model();
        auto& new_model = new_impl.strategy->get_model();

        // Raw means
        auto legacy_means = legacy_model.means().cpu().flatten();
        auto new_means = to_torch_flat(new_model.means());
        spdlog::info("  Means: max diff = {:.6e}", (legacy_means - new_means).abs().max().item<float>());

        // Raw scales (before exp)
        auto legacy_scales = legacy_model.scaling_raw().cpu().flatten();
        auto new_scales = to_torch_flat(new_model.scaling_raw());
        spdlog::info("  Scales (raw): max diff = {:.6e}", (legacy_scales - new_scales).abs().max().item<float>());

        // Raw quats (before normalize)
        auto legacy_quats = legacy_model.rotation_raw().cpu().flatten();
        auto new_quats = to_torch_flat(new_model.rotation_raw());
        spdlog::info("  Quats (raw): max diff = {:.6e}", (legacy_quats - new_quats).abs().max().item<float>());

        // Raw opacity (before sigmoid)
        auto legacy_opacity = legacy_model.opacity_raw().cpu().flatten();
        auto new_opacity = to_torch_flat(new_model.opacity_raw());
        spdlog::info("  Opacity (raw): max diff = {:.6e}", (legacy_opacity - new_opacity).abs().max().item<float>());

        // SH0
        auto legacy_sh0 = legacy_model.sh0().cpu().flatten();
        auto new_sh0 = to_torch_flat(new_model.sh0());
        spdlog::info("  SH0: max diff = {:.6e}", (legacy_sh0 - new_sh0).abs().max().item<float>());

        // Show first few values
        spdlog::info("  First Gaussian:");
        spdlog::info("    Legacy means[0]: [{:.6f}, {:.6f}, {:.6f}]",
                     legacy_means[0].item<float>(), legacy_means[1].item<float>(), legacy_means[2].item<float>());
        spdlog::info("    New means[0]:    [{:.6f}, {:.6f}, {:.6f}]",
                     new_means[0].item<float>(), new_means[1].item<float>(), new_means[2].item<float>());
        spdlog::info("    Legacy scales_raw[0]: [{:.6f}, {:.6f}, {:.6f}]",
                     legacy_scales[0].item<float>(), legacy_scales[1].item<float>(), legacy_scales[2].item<float>());
        spdlog::info("    New scales_raw[0]:    [{:.6f}, {:.6f}, {:.6f}]",
                     new_scales[0].item<float>(), new_scales[1].item<float>(), new_scales[2].item<float>());
        spdlog::info("    Legacy quats_raw[0]: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                     legacy_quats[0].item<float>(), legacy_quats[1].item<float>(),
                     legacy_quats[2].item<float>(), legacy_quats[3].item<float>());
        spdlog::info("    New quats_raw[0]:    [{:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                     new_quats[0].item<float>(), new_quats[1].item<float>(),
                     new_quats[2].item<float>(), new_quats[3].item<float>());
    }

    for (int iter = 1; iter <= max_iterations; ++iter) {
        bool should_log = (iter == 1) || (iter % log_interval == 0) || (iter == max_iterations);

        // === FORWARD PASS ===
        auto legacy_output = gs::training::rasterize(
            *legacy_cam,
            legacy.strategy->get_model(),
            legacy.background,
            1.0f, false, false, gs::training::RenderMode::RGB);

        auto new_result = lfs::training::gsplat_rasterize_forward(
            *new_cam,
            new_impl.strategy->get_model(),
            new_impl.background,
            1.0f, false, lfs::training::GsplatRenderMode::RGB, false);
        ASSERT_TRUE(new_result.has_value()) << "LFS gsplat forward failed at iter " << iter;
        auto [new_output, new_ctx] = new_result.value();

        // === COMPUTE LOSS (no regularization) ===
        torch::Tensor legacy_rendered = legacy_output.image;
        legacy_rendered = legacy_rendered.dim() == 3 ? legacy_rendered.unsqueeze(0) : legacy_rendered;
        torch::Tensor legacy_gt_4d = legacy_gt.dim() == 3 ? legacy_gt.unsqueeze(0) : legacy_gt;

        auto legacy_l1_loss = torch::l1_loss(legacy_rendered, legacy_gt_4d);
        auto legacy_ssim_loss = 1.f - fused_ssim(legacy_rendered, legacy_gt_4d, "valid", true);
        // Pure photometric loss - NO scale/opacity regularization
        torch::Tensor legacy_loss_tensor = (1.f - legacy.params.optimization.lambda_dssim) * legacy_l1_loss +
                                            legacy.params.optimization.lambda_dssim * legacy_ssim_loss;
        float legacy_loss_value = legacy_loss_tensor.item<float>();

        static lfs::training::losses::PhotometricLoss new_loss;
        lfs::training::losses::PhotometricLoss::Params loss_params{
            .lambda_dssim = new_impl.params.optimization.lambda_dssim
        };
        auto new_loss_result = new_loss.forward(new_output.image, new_gt, loss_params);
        ASSERT_TRUE(new_loss_result.has_value());
        auto [new_loss_tensor, new_loss_ctx] = *new_loss_result;
        float new_loss_value = new_loss_tensor.item();

        // === BACKWARD PASS ===
        if (!new_impl.strategy->get_optimizer().has_gradients()) {
            new_impl.strategy->get_optimizer().allocate_gradients();
        } else {
            new_impl.strategy->get_optimizer().zero_grad(iter);
        }

        legacy_loss_tensor.backward();

        lfs::core::Tensor grad_alpha = lfs::core::Tensor::zeros(
            {1, static_cast<int>(new_output.height), static_cast<int>(new_output.width)},
            lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        lfs::training::gsplat_rasterize_backward(
            new_ctx, new_loss_ctx.grad_image, grad_alpha,
            new_impl.strategy->get_model(), new_impl.strategy->get_optimizer());

        // === OPTIMIZER STEP (NO post_backward - no densification/pruning) ===
        legacy.strategy->step(iter);
        new_impl.strategy->step(iter);

        // Release arena frame
        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
        arena.end_frame(new_ctx.frame_id);

        // === LOG PROGRESS ===
        if (should_log) {
            auto legacy_img = legacy_output.image.cpu();
            auto new_img_cpu = new_output.image.to(lfs::core::Device::CPU);
            std::vector<long> img_shape;
            for (size_t i = 0; i < new_img_cpu.ndim(); ++i) {
                img_shape.push_back(static_cast<long>(new_img_cpu.shape()[i]));
            }
            torch::Tensor new_img_torch = torch::from_blob(
                const_cast<float*>(new_img_cpu.ptr<float>()),
                torch::IntArrayRef(img_shape), torch::kFloat32).clone();

            auto img_diff = (legacy_img - new_img_torch).abs();
            float img_max_diff = img_diff.max().item<float>();

            // Compare ALL parameters
            auto& legacy_model = legacy.strategy->get_model();
            auto& new_model = new_impl.strategy->get_model();

            // Means
            auto legacy_means = legacy_model.means().cpu().flatten();
            auto new_means = to_torch_flat(new_model.means());
            float means_diff = (legacy_means - new_means).abs().max().item<float>();

            // Scales (raw)
            auto legacy_scales = legacy_model.scaling_raw().cpu().flatten();
            auto new_scales = to_torch_flat(new_model.scaling_raw());
            float scales_diff = (legacy_scales - new_scales).abs().max().item<float>();

            // Rotations (raw quats)
            auto legacy_quats = legacy_model.rotation_raw().cpu().flatten();
            auto new_quats = to_torch_flat(new_model.rotation_raw());
            float quats_diff = (legacy_quats - new_quats).abs().max().item<float>();

            // Opacity (raw)
            auto legacy_opacity = legacy_model.opacity_raw().cpu().flatten();
            auto new_opacity = to_torch_flat(new_model.opacity_raw());
            float opacity_diff = (legacy_opacity - new_opacity).abs().max().item<float>();

            // SH0
            auto legacy_sh0 = legacy_model.sh0().cpu().flatten();
            auto new_sh0 = to_torch_flat(new_model.sh0());
            float sh0_diff = (legacy_sh0 - new_sh0).abs().max().item<float>();

            spdlog::info("[Iter {:4d}] Loss: L={:.6f} N={:.6f} diff={:.2e} | Image: {:.2e}",
                         iter, legacy_loss_value, new_loss_value,
                         std::abs(legacy_loss_value - new_loss_value), img_max_diff);
            spdlog::info("           Means: {:.2e} | Scales: {:.2e} | Quats: {:.2e} | Opacity: {:.2e} | SH0: {:.2e}",
                         means_diff, scales_diff, quats_diff, opacity_diff, sh0_diff);
        }

        last_loss_legacy = legacy_loss_value;
        last_loss_new = new_loss_value;
    }

    spdlog::info("=== Final Results after {} iterations ===", max_iterations);
    spdlog::info("  Legacy final loss: {:.6f}", last_loss_legacy);
    spdlog::info("  New final loss: {:.6f}", last_loss_new);
    spdlog::info("  Loss difference: {:.6f} ({:.2f}%)",
                 std::abs(last_loss_legacy - last_loss_new),
                 100.0f * std::abs(last_loss_legacy - last_loss_new) / last_loss_legacy);

    // Final comparison should show both converging similarly
    // Allow 5% difference in final loss
    float loss_ratio = last_loss_new / last_loss_legacy;
    EXPECT_GT(loss_ratio, 0.95f) << "New loss diverged too much below legacy";
    EXPECT_LT(loss_ratio, 1.05f) << "New loss diverged too much above legacy";

    spdlog::info("=== GsplatExtended1000Iterations test passed ===");
}

/**
 * @brief Detailed input comparison - verify inputs to gsplat kernels match
 */
TEST_F(GsplatTrainingDebugTest, GsplatInputComparison) {
    spdlog::info("=== Starting GsplatInputComparison test ===");

    auto [legacy, new_impl] = initialize_both_gsplat();

    auto& legacy_model = legacy.strategy->get_model();
    auto& new_model = new_impl.strategy->get_model();

    auto legacy_cameras = legacy.dataset->get_cameras();
    auto new_cameras = new_impl.dataset->get_cameras();
    auto legacy_cam = legacy_cameras[0];
    auto new_cam = new_cameras[0];

    spdlog::info("=== Comparing Model Parameters (Inputs to Rasterizer) ===");

    // Compare means
    auto legacy_means = legacy_model.get_means().cpu();
    auto new_means_cpu = new_model.means().to(lfs::core::Device::CPU);
    torch::Tensor new_means_torch = torch::from_blob(
        new_means_cpu.ptr<float>(),
        {static_cast<long>(new_means_cpu.shape()[0]),
         static_cast<long>(new_means_cpu.shape()[1])},
        torch::kFloat32).clone();

    auto means_diff = (legacy_means - new_means_torch).abs();
    spdlog::info("Means - Max diff: {:.6e}, Mean diff: {:.6e}",
                 means_diff.max().item<float>(), means_diff.mean().item<float>());

    // Compare opacities (activated)
    auto legacy_opacities = legacy_model.get_opacity().cpu();
    auto new_opacities_cpu = new_model.get_opacity().to(lfs::core::Device::CPU);

    // Build shape for new tensor
    std::vector<long> opacity_shape;
    for (size_t i = 0; i < new_opacities_cpu.ndim(); ++i) {
        opacity_shape.push_back(static_cast<long>(new_opacities_cpu.shape()[i]));
    }
    torch::Tensor new_opacities_torch = torch::from_blob(
        new_opacities_cpu.ptr<float>(),
        torch::IntArrayRef(opacity_shape),
        torch::kFloat32).clone();

    // Squeeze to 1D for comparison
    legacy_opacities = legacy_opacities.flatten();
    new_opacities_torch = new_opacities_torch.flatten();

    auto opacity_diff = (legacy_opacities - new_opacities_torch).abs();
    spdlog::info("Opacities (activated) - Max diff: {:.6e}, Mean diff: {:.6e}",
                 opacity_diff.max().item<float>(), opacity_diff.mean().item<float>());

    // Compare scaling (activated)
    auto legacy_scales = legacy_model.get_scaling().cpu();
    auto new_scales_cpu = new_model.get_scaling().to(lfs::core::Device::CPU);

    std::vector<long> scales_shape;
    for (size_t i = 0; i < new_scales_cpu.ndim(); ++i) {
        scales_shape.push_back(static_cast<long>(new_scales_cpu.shape()[i]));
    }
    torch::Tensor new_scales_torch = torch::from_blob(
        new_scales_cpu.ptr<float>(),
        torch::IntArrayRef(scales_shape),
        torch::kFloat32).clone();

    auto scales_diff = (legacy_scales - new_scales_torch).abs();
    spdlog::info("Scaling (activated) - Max diff: {:.6e}, Mean diff: {:.6e}",
                 scales_diff.max().item<float>(), scales_diff.mean().item<float>());

    // Compare rotation (activated)
    auto legacy_rotations = legacy_model.get_rotation().cpu();
    auto new_rotations_cpu = new_model.get_rotation().to(lfs::core::Device::CPU);

    std::vector<long> rotations_shape;
    for (size_t i = 0; i < new_rotations_cpu.ndim(); ++i) {
        rotations_shape.push_back(static_cast<long>(new_rotations_cpu.shape()[i]));
    }
    torch::Tensor new_rotations_torch = torch::from_blob(
        new_rotations_cpu.ptr<float>(),
        torch::IntArrayRef(rotations_shape),
        torch::kFloat32).clone();

    auto rotation_diff = (legacy_rotations - new_rotations_torch).abs();
    spdlog::info("Rotation (activated) - Max diff: {:.6e}, Mean diff: {:.6e}",
                 rotation_diff.max().item<float>(), rotation_diff.mean().item<float>());

    // Compare SH coefficients
    auto legacy_shs = legacy_model.get_shs().cpu();
    auto new_shs = new_model.get_shs();
    auto new_shs_cpu = new_shs.to(lfs::core::Device::CPU);

    spdlog::info("Legacy SH shape: [{}, {}, {}]",
                 legacy_shs.size(0), legacy_shs.size(1), legacy_shs.size(2));
    spdlog::info("New SH shape: [{}, {}, {}]",
                 new_shs_cpu.shape()[0], new_shs_cpu.shape()[1], new_shs_cpu.shape()[2]);

    // Convert new SH to torch for comparison
    std::vector<long> sh_shape;
    for (size_t i = 0; i < new_shs_cpu.ndim(); ++i) {
        sh_shape.push_back(static_cast<long>(new_shs_cpu.shape()[i]));
    }
    torch::Tensor new_shs_torch = torch::from_blob(
        new_shs_cpu.ptr<float>(),
        torch::IntArrayRef(sh_shape),
        torch::kFloat32).clone();

    auto sh_diff = (legacy_shs - new_shs_torch).abs();
    spdlog::info("SH coefficients - Max diff: {:.6e}, Mean diff: {:.6e}",
                 sh_diff.max().item<float>(), sh_diff.mean().item<float>());

    // Show sample SH values
    spdlog::info("First Gaussian SH0: Legacy=[{:.6f}, {:.6f}, {:.6f}], New=[{:.6f}, {:.6f}, {:.6f}]",
                 legacy_shs[0][0][0].item<float>(),
                 legacy_shs[0][0][1].item<float>(),
                 legacy_shs[0][0][2].item<float>(),
                 new_shs_torch[0][0][0].item<float>(),
                 new_shs_torch[0][0][1].item<float>(),
                 new_shs_torch[0][0][2].item<float>());

    // Compare camera matrices
    spdlog::info("=== Comparing Camera Parameters ===");

    auto legacy_viewmat = legacy_cam->world_view_transform().cpu();
    auto new_viewmat = new_cam->world_view_transform();
    auto new_viewmat_cpu = new_viewmat.to(lfs::core::Device::CPU);

    spdlog::info("Legacy viewmat shape: [{}, {}, {}]",
                 legacy_viewmat.size(0), legacy_viewmat.size(1), legacy_viewmat.size(2));
    spdlog::info("New viewmat shape: [{}, {}, {}]",
                 new_viewmat_cpu.shape()[0], new_viewmat_cpu.shape()[1], new_viewmat_cpu.shape()[2]);

    // Convert new viewmat to torch
    torch::Tensor new_viewmat_torch = torch::from_blob(
        new_viewmat_cpu.ptr<float>(),
        {static_cast<long>(new_viewmat_cpu.shape()[0]),
         static_cast<long>(new_viewmat_cpu.shape()[1]),
         static_cast<long>(new_viewmat_cpu.shape()[2])},
        torch::kFloat32).clone();

    auto viewmat_diff = (legacy_viewmat - new_viewmat_torch).abs();
    spdlog::info("Viewmat - Max diff: {:.6e}, Mean diff: {:.6e}",
                 viewmat_diff.max().item<float>(), viewmat_diff.mean().item<float>());

    // Compare K matrix
    auto legacy_K = legacy_cam->K().cpu();
    auto new_K = new_cam->K();
    auto new_K_cpu = new_K.to(lfs::core::Device::CPU);

    torch::Tensor new_K_torch = torch::from_blob(
        new_K_cpu.ptr<float>(),
        {static_cast<long>(new_K_cpu.shape()[0]),
         static_cast<long>(new_K_cpu.shape()[1]),
         static_cast<long>(new_K_cpu.shape()[2])},
        torch::kFloat32).clone();

    auto K_diff = (legacy_K - new_K_torch).abs();
    spdlog::info("K matrix - Max diff: {:.6e}, Mean diff: {:.6e}",
                 K_diff.max().item<float>(), K_diff.mean().item<float>());

    // Print full K matrices for comparison
    spdlog::info("Legacy K matrix:");
    for (int i = 0; i < 3; ++i) {
        spdlog::info("  [{:.6f}, {:.6f}, {:.6f}]",
                     legacy_K[0][i][0].item<float>(),
                     legacy_K[0][i][1].item<float>(),
                     legacy_K[0][i][2].item<float>());
    }
    spdlog::info("New K matrix:");
    for (int i = 0; i < 3; ++i) {
        spdlog::info("  [{:.6f}, {:.6f}, {:.6f}]",
                     new_K_torch[0][i][0].item<float>(),
                     new_K_torch[0][i][1].item<float>(),
                     new_K_torch[0][i][2].item<float>());
    }

    // Also print image dimensions
    spdlog::info("Legacy camera dims: {}x{}", legacy_cam->image_width(), legacy_cam->image_height());
    spdlog::info("New camera dims: {}x{}", new_cam->image_width(), new_cam->image_height());

    // Compare campos computation (critical for SH evaluation!)
    spdlog::info("=== Comparing Campos Computation ===");

    // Legacy method: use matrix inverse
    auto legacy_viewmat_cpu = legacy_cam->world_view_transform().cpu();
    auto legacy_viewmat_inv = torch::inverse(legacy_viewmat_cpu);  // [1, 4, 4]
    auto legacy_campos = legacy_viewmat_inv.index({torch::indexing::Slice(),
                                                    torch::indexing::Slice(torch::indexing::None, 3),
                                                    3}); // [1, 3]
    spdlog::info("Legacy campos (via inverse): [{:.6f}, {:.6f}, {:.6f}]",
                 legacy_campos[0][0].item<float>(),
                 legacy_campos[0][1].item<float>(),
                 legacy_campos[0][2].item<float>());

    // LFS method: -R^T * t (use torch tensor converted from LFS tensor)
    auto R = legacy_viewmat_cpu.index({torch::indexing::Slice(),
                                        torch::indexing::Slice(torch::indexing::None, 3),
                                        torch::indexing::Slice(torch::indexing::None, 3)}); // [1, 3, 3]
    auto t = legacy_viewmat_cpu.index({torch::indexing::Slice(),
                                        torch::indexing::Slice(torch::indexing::None, 3),
                                        3}); // [1, 3]
    auto R_t = R.transpose(-1, -2);  // [1, 3, 3]
    auto new_campos = torch::bmm(R_t, t.unsqueeze(-1)).mul(-1.0f).squeeze(-1);  // [1, 3]
    spdlog::info("New campos (via -R^T*t): [{:.6f}, {:.6f}, {:.6f}]",
                 new_campos[0][0].item<float>(),
                 new_campos[0][1].item<float>(),
                 new_campos[0][2].item<float>());

    auto campos_diff = (legacy_campos - new_campos).abs();
    spdlog::info("Campos - Max diff: {:.6e}", campos_diff.max().item<float>());

    spdlog::info("=== GsplatInputComparison test complete ===");
}

/**
 * @brief Compare campos computation methods between Legacy (inverse) and LFS (-R^T*t)
 * This is a simplified test that doesn't require calling gsplat functions directly.
 */
TEST_F(GsplatTrainingDebugTest, GsplatCamposComparison) {
    spdlog::info("=== Starting GsplatCamposComparison test ===");

    auto [legacy, new_impl] = initialize_both_gsplat();

    auto legacy_cameras = legacy.dataset->get_cameras();
    auto legacy_cam = legacy_cameras[0];

    // Get viewmat from Legacy camera
    auto legacy_viewmat = legacy_cam->world_view_transform();  // [1, 4, 4]
    spdlog::info("Legacy viewmat shape: [{}, {}, {}]",
                 legacy_viewmat.size(0), legacy_viewmat.size(1), legacy_viewmat.size(2));

    spdlog::info("=== Comparing Campos Computation ===");

    // Method 1: Legacy campos via matrix inverse (what Legacy gsplat uses)
    auto legacy_viewmat_inv = torch::inverse(legacy_viewmat);
    auto legacy_campos = legacy_viewmat_inv.index({torch::indexing::Slice(),
                                                    torch::indexing::Slice(torch::indexing::None, 3),
                                                    3}); // [C, 3]

    // Method 2: LFS-style campos: -R^T * t
    auto R = legacy_viewmat.index({torch::indexing::Slice(),
                                    torch::indexing::Slice(torch::indexing::None, 3),
                                    torch::indexing::Slice(torch::indexing::None, 3)});
    auto t = legacy_viewmat.index({torch::indexing::Slice(),
                                    torch::indexing::Slice(torch::indexing::None, 3),
                                    3});
    auto R_t = R.transpose(-1, -2);
    auto lfs_style_campos = torch::bmm(R_t, t.unsqueeze(-1)).mul(-1.0f).squeeze(-1);

    spdlog::info("Method 1 - Legacy campos (via inverse): [{:.6f}, {:.6f}, {:.6f}]",
                 legacy_campos[0][0].item().toFloat(),
                 legacy_campos[0][1].item().toFloat(),
                 legacy_campos[0][2].item().toFloat());
    spdlog::info("Method 2 - LFS-style campos (via -R^T*t): [{:.6f}, {:.6f}, {:.6f}]",
                 lfs_style_campos[0][0].item().toFloat(),
                 lfs_style_campos[0][1].item().toFloat(),
                 lfs_style_campos[0][2].item().toFloat());

    auto campos_diff = (legacy_campos - lfs_style_campos).abs();
    float max_diff = campos_diff.max().item().toFloat();
    spdlog::info("Campos diff - Max: {:.6e}", max_diff);

    // Both methods should give the same result for a valid rotation matrix
    // since inverse(V) = [R^T, -R^T*t; 0 0 0 1] when V = [R, t; 0 0 0 1]
    EXPECT_LT(max_diff, 1e-5f) << "Campos computation methods should match!";

    spdlog::info("=== GsplatCamposComparison test complete ===");
}

/**
 * @brief Deep comparison of gsplat intermediate values (radii, means2d, depths, colors, tile_offsets)
 * This test directly calls rasterize_from_world_with_sh_fwd from both implementations
 * to compare all intermediate results before the final rasterization.
 */
TEST_F(GsplatTrainingDebugTest, GsplatIntermediateComparison) {
    spdlog::info("=== Starting GsplatIntermediateComparison test ===");

    auto [legacy, new_impl] = initialize_both_gsplat();

    auto& legacy_model = legacy.strategy->get_model();
    auto& new_model = new_impl.strategy->get_model();

    auto legacy_cameras = legacy.dataset->get_cameras();
    auto new_cameras = new_impl.dataset->get_cameras();
    auto legacy_cam = legacy_cameras[0];
    auto new_cam = new_cameras[0];

    // === Prepare Legacy inputs ===
    auto legacy_means = legacy_model.get_means().contiguous();
    auto legacy_quats = legacy_model.get_rotation().contiguous();
    auto legacy_scales = legacy_model.get_scaling().contiguous();
    auto legacy_opacities = legacy_model.get_opacity().contiguous();
    auto legacy_sh_coeffs = legacy_model.get_shs().contiguous();
    const int legacy_sh_degree = legacy_model.get_active_sh_degree();
    auto legacy_viewmat = legacy_cam->world_view_transform().contiguous();
    auto legacy_K = legacy_cam->K().contiguous();  // Already [1, 3, 3]
    auto legacy_bg = legacy.background.view({1, -1}).contiguous();

    const int image_width = legacy_cam->image_width();
    const int image_height = legacy_cam->image_height();
    constexpr int tile_size = 16;
    constexpr float eps2d = 0.3f;
    constexpr float near_plane = 0.01f;
    constexpr float far_plane = 10000.0f;
    constexpr float radius_clip = 0.0f;
    constexpr float scaling_modifier = 1.0f;
    constexpr bool calc_compensations = false;

    // Use global-scope types from gsplat/Cameras.h (legacy gsplat is not namespaced)
    ::UnscentedTransformParameters ut_params;

    spdlog::info("Calling Legacy gsplat::rasterize_from_world_with_sh_fwd...");

    // Call Legacy forward
    auto legacy_results = gsplat::rasterize_from_world_with_sh_fwd(
        legacy_means,
        legacy_quats,
        legacy_scales,
        legacy_opacities,
        legacy_sh_coeffs,
        static_cast<uint32_t>(legacy_sh_degree),
        legacy_bg,
        std::nullopt,  // masks
        image_width,
        image_height,
        tile_size,
        legacy_viewmat,
        std::nullopt,  // viewmats1
        legacy_K,
        gsplat::CameraModelType::PINHOLE,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        scaling_modifier,
        calc_compensations,
        0,  // render_mode RGB
        ut_params,
        ::ShutterType::GLOBAL,
        std::nullopt,  // radial_coeffs
        std::nullopt,  // tangential_coeffs
        std::nullopt); // thin_prism_coeffs

    auto legacy_render = std::get<0>(legacy_results).cpu();
    auto legacy_radii = std::get<2>(legacy_results).cpu();
    auto legacy_means2d = std::get<3>(legacy_results).cpu();
    auto legacy_depths = std::get<4>(legacy_results).cpu();
    auto legacy_colors = std::get<5>(legacy_results).cpu();
    auto legacy_tile_offsets = std::get<6>(legacy_results).cpu();
    auto legacy_flatten_ids = std::get<7>(legacy_results).cpu();

    spdlog::info("Legacy render shape: [{}, {}, {}, {}]",
                 legacy_render.size(0), legacy_render.size(1), legacy_render.size(2), legacy_render.size(3));
    spdlog::info("Legacy radii shape: [{}, {}, {}]",
                 legacy_radii.size(0), legacy_radii.size(1), legacy_radii.size(2));
    spdlog::info("Legacy means2d shape: [{}, {}, {}]",
                 legacy_means2d.size(0), legacy_means2d.size(1), legacy_means2d.size(2));
    spdlog::info("Legacy depths shape: [{}, {}]",
                 legacy_depths.size(0), legacy_depths.size(1));
    spdlog::info("Legacy colors shape: [{}, {}, {}]",
                 legacy_colors.size(0), legacy_colors.size(1), legacy_colors.size(2));
    spdlog::info("Legacy tile_offsets shape: [{}, {}, {}]",
                 legacy_tile_offsets.size(0), legacy_tile_offsets.size(1), legacy_tile_offsets.size(2));
    spdlog::info("Legacy flatten_ids size: {}", legacy_flatten_ids.size(0));

    // === Prepare LFS inputs ===
    auto new_means = new_model.get_means().contiguous();
    auto new_quats = new_model.get_rotation().contiguous();
    auto new_scales = new_model.get_scaling().contiguous();
    auto new_opacities = new_model.get_opacity().contiguous();
    auto new_sh_coeffs = new_model.get_shs().contiguous();
    const int new_sh_degree = new_model.get_active_sh_degree();
    auto new_viewmat = new_cam->world_view_transform().contiguous();
    auto new_K = new_cam->K().contiguous();  // Already [1, 3, 3]
    auto new_bg = new_impl.background.view({1, -1}).contiguous();

    spdlog::info("Calling LFS lfs::training::gsplat_rasterize_forward...");

    // Use the high-level LFS API
    auto new_result = lfs::training::gsplat_rasterize_forward(
        *new_cam,
        new_model,
        new_impl.background,
        scaling_modifier,
        false,  // antialiased
        lfs::training::GsplatRenderMode::RGB,
        false   // use_gut
    );
    ASSERT_TRUE(new_result.has_value()) << "LFS gsplat forward failed: " << new_result.error();
    auto [new_output, new_ctx] = new_result.value();

    // Wrap context raw pointers into tensors for comparison
    // Note: These are views into arena memory, valid until arena.end_frame()
    const uint32_t N = new_ctx.N;
    const uint32_t C = 1;  // Single camera

    // Create tensor views from raw pointers
    auto new_render = new_output.image.to(lfs::core::Device::CPU);

    spdlog::info("LFS render shape: [{}, {}, {}]",
                 new_render.shape()[0], new_render.shape()[1], new_render.shape()[2]);
    spdlog::info("LFS context: N={}, n_isects={}", new_ctx.N, new_ctx.n_isects);

    // === Compare intermediate values BEFORE arena cleanup ===
    // Create torch tensors from LFS raw pointers (copy to CPU before arena cleanup)
    auto lfs_radii = torch::from_blob(new_ctx.radii_ptr, {(long)C, (long)N, 2},
                                      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32)).cpu().clone();
    auto lfs_means2d = torch::from_blob(new_ctx.means2d_ptr, {(long)C, (long)N, 2},
                                        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).cpu().clone();
    auto lfs_depths = torch::from_blob(new_ctx.depths_ptr, {(long)C, (long)N},
                                       torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).cpu().clone();
    auto lfs_colors = torch::from_blob(new_ctx.colors_ptr, {(long)C, (long)N, 3},
                                       torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).cpu().clone();

    // Clean up arena
    auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
    arena.end_frame(new_ctx.frame_id);
    if (new_ctx.isect_ids_ptr) cudaFree(new_ctx.isect_ids_ptr);
    if (new_ctx.flatten_ids_ptr) cudaFree(new_ctx.flatten_ids_ptr);

    // Now compare intermediate values
    spdlog::info("=== Comparing Intermediate Values ===");

    auto get_float = [](const torch::Tensor& t) -> float {
        return t.item().toFloat();
    };

    // Compare radii (number of visible Gaussians)
    auto legacy_visible = (legacy_radii.sum(-1) > 0).sum().item().toInt();
    auto lfs_visible = (lfs_radii.sum(-1) > 0).sum().item().toInt();
    spdlog::info("Visible Gaussians: Legacy={}, LFS={}", legacy_visible, lfs_visible);

    // Compare radii values directly
    auto radii_match = (legacy_radii == lfs_radii).all().item().toBool();
    spdlog::info("Radii match exactly: {}", radii_match ? "YES" : "NO");
    if (!radii_match) {
        auto radii_diff = (legacy_radii.to(torch::kFloat32) - lfs_radii.to(torch::kFloat32)).abs();
        spdlog::info("Radii diff: max={}, mean={}", get_float(radii_diff.max()), get_float(radii_diff.mean()));
    }

    // Compare means2d
    auto means2d_diff = (legacy_means2d - lfs_means2d).abs();
    spdlog::info("Means2D: max_diff={:.6f}, mean_diff={:.6f}",
                 get_float(means2d_diff.max()), get_float(means2d_diff.mean()));

    // Compare depths
    auto depths_diff = (legacy_depths - lfs_depths).abs();
    spdlog::info("Depths: max_diff={:.6f}, mean_diff={:.6f}",
                 get_float(depths_diff.max()), get_float(depths_diff.mean()));

    // Compare colors (after SH evaluation)
    auto colors_diff = (legacy_colors - lfs_colors).abs();
    spdlog::info("Colors: max_diff={:.6f}, mean_diff={:.6f}",
                 get_float(colors_diff.max()), get_float(colors_diff.mean()));
    spdlog::info("Legacy colors: min={:.6f}, max={:.6f}, mean={:.6f}",
                 get_float(legacy_colors.min()), get_float(legacy_colors.max()), get_float(legacy_colors.mean()));
    spdlog::info("LFS colors: min={:.6f}, max={:.6f}, mean={:.6f}",
                 get_float(lfs_colors.min()), get_float(lfs_colors.max()), get_float(lfs_colors.mean()));

    // Sample a few colors from visible Gaussians
    for (int sample = 0; sample < 5; ++sample) {
        int idx = sample * (N / 5);
        spdlog::info("Color[{}]: Legacy=[{:.4f},{:.4f},{:.4f}] LFS=[{:.4f},{:.4f},{:.4f}]", idx,
                     legacy_colors[0][idx][0].item().toFloat(),
                     legacy_colors[0][idx][1].item().toFloat(),
                     legacy_colors[0][idx][2].item().toFloat(),
                     lfs_colors[0][idx][0].item().toFloat(),
                     lfs_colors[0][idx][1].item().toFloat(),
                     lfs_colors[0][idx][2].item().toFloat());
    }

    // === Compare rendered output ===
    spdlog::info("=== Comparing Rendered Output ===");

    // Helper to convert lfs::core::Tensor to torch::Tensor
    auto to_torch = [](const lfs::core::Tensor& t) -> torch::Tensor {
        std::vector<long> shape;
        for (size_t i = 0; i < t.ndim(); ++i) {
            shape.push_back(static_cast<long>(t.shape()[i]));
        }
        return torch::from_blob(
            const_cast<float*>(t.ptr<float>()),
            torch::IntArrayRef(shape),
            torch::kFloat32).clone();
    };

    // Compare final rendered image
    auto new_render_torch = to_torch(new_render);
    // Legacy render is [1, H, W, C], LFS render is [C, H, W] - need to match shapes
    spdlog::info("Legacy render shape: [{}, {}, {}, {}]",
                 legacy_render.size(0), legacy_render.size(1), legacy_render.size(2), legacy_render.size(3));
    spdlog::info("LFS render shape: [{}, {}, {}]",
                 new_render_torch.size(0), new_render_torch.size(1), new_render_torch.size(2));

    // LFS render is [C, H, W], legacy is [1, H, W, C] - permute LFS to match
    auto new_render_permuted = new_render_torch.permute({1, 2, 0}).unsqueeze(0);  // [1, H, W, C]

    // Print per-image statistics
    spdlog::info("Legacy render stats: min={:.6f}, max={:.6f}, mean={:.6f}",
                 get_float(legacy_render.min()), get_float(legacy_render.max()), get_float(legacy_render.mean()));
    spdlog::info("LFS render stats: min={:.6f}, max={:.6f}, mean={:.6f}",
                 get_float(new_render_permuted.min()), get_float(new_render_permuted.max()), get_float(new_render_permuted.mean()));

    // Count non-zero pixels
    auto legacy_nonzero = (legacy_render.sum(-1) > 0).sum().item().toInt();
    auto lfs_nonzero = (new_render_permuted.sum(-1) > 0).sum().item().toInt();
    spdlog::info("Non-zero pixels: Legacy={}, LFS={}", legacy_nonzero, lfs_nonzero);

    auto render_diff = (legacy_render - new_render_permuted).abs();
    spdlog::info("Rendered Image - Max diff: {:.6e}, Mean diff: {:.6e}",
                 get_float(render_diff.max()), get_float(render_diff.mean()));

    // Compare specific pixels - sample multiple locations
    auto sample_pixel = [&](int y, int x, const char* label) {
        if (y >= 0 && y < image_height && x >= 0 && x < image_width) {
            spdlog::info("{} ({},{}): Legacy=[{:.4f},{:.4f},{:.4f}] LFS=[{:.4f},{:.4f},{:.4f}]",
                         label, x, y,
                         get_float(legacy_render[0][y][x][0]),
                         get_float(legacy_render[0][y][x][1]),
                         get_float(legacy_render[0][y][x][2]),
                         get_float(new_render_permuted[0][y][x][0]),
                         get_float(new_render_permuted[0][y][x][1]),
                         get_float(new_render_permuted[0][y][x][2]));
        }
    };
    sample_pixel(image_height / 2, image_width / 2, "Center");
    sample_pixel(0, 0, "TopLeft");
    sample_pixel(0, image_width - 1, "TopRight");
    sample_pixel(image_height - 1, 0, "BotLeft");
    sample_pixel(image_height - 1, image_width - 1, "BotRight");
    sample_pixel(100, 100, "Pixel(100,100)");
    sample_pixel(200, 300, "Pixel(300,200)");
    sample_pixel(400, 600, "Pixel(600,400)");

    // Find max value location in legacy
    auto legacy_flat = legacy_render.view({-1, 3}).sum(-1);
    auto legacy_max_idx = legacy_flat.argmax().item().toLong();
    int legacy_max_y = (legacy_max_idx / image_width) % image_height;
    int legacy_max_x = legacy_max_idx % image_width;
    spdlog::info("Legacy max at ({},{}): [{:.4f},{:.4f},{:.4f}]",
                 legacy_max_x, legacy_max_y,
                 get_float(legacy_render[0][legacy_max_y][legacy_max_x][0]),
                 get_float(legacy_render[0][legacy_max_y][legacy_max_x][1]),
                 get_float(legacy_render[0][legacy_max_y][legacy_max_x][2]));
    spdlog::info("LFS at same location: [{:.4f},{:.4f},{:.4f}]",
                 get_float(new_render_permuted[0][legacy_max_y][legacy_max_x][0]),
                 get_float(new_render_permuted[0][legacy_max_y][legacy_max_x][1]),
                 get_float(new_render_permuted[0][legacy_max_y][legacy_max_x][2]));

    // Find max value location in LFS
    auto lfs_flat = new_render_permuted.view({-1, 3}).sum(-1);
    auto lfs_max_idx = lfs_flat.argmax().item().toLong();
    int lfs_max_y = (lfs_max_idx / image_width) % image_height;
    int lfs_max_x = lfs_max_idx % image_width;
    spdlog::info("LFS max at ({},{}): [{:.4f},{:.4f},{:.4f}]",
                 lfs_max_x, lfs_max_y,
                 get_float(new_render_permuted[0][lfs_max_y][lfs_max_x][0]),
                 get_float(new_render_permuted[0][lfs_max_y][lfs_max_x][1]),
                 get_float(new_render_permuted[0][lfs_max_y][lfs_max_x][2]));
    spdlog::info("Legacy at same location: [{:.4f},{:.4f},{:.4f}]",
                 get_float(legacy_render[0][lfs_max_y][lfs_max_x][0]),
                 get_float(legacy_render[0][lfs_max_y][lfs_max_x][1]),
                 get_float(legacy_render[0][lfs_max_y][lfs_max_x][2]));

    // Tolerance check
    float tolerance = 0.01f;  // 1% tolerance
    EXPECT_LT(get_float(render_diff.max()), tolerance)
        << "Rendered image comparison shows significant difference!";

    spdlog::info("=== GsplatIntermediateComparison test complete ===");
}
