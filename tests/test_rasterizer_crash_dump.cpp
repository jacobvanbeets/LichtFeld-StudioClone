/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Rasterizer crash reproduction test using crash dump data

#include "core/cuda/memory_arena.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "rasterization/fastgs/rasterization/include/rasterization_api.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

using namespace lfs::core;

namespace {

    constexpr const char* CRASH_DUMP_PATH = "crash_dump_extracted/crash_dump_20251211_143826_101";
    constexpr int TILE_SIZE = 16;
    constexpr int STRESS_TEST_ITERATIONS = 50;
    constexpr float DEGENERATE_QUAT_THRESHOLD = 1e-8f;

    struct CrashParams {
        int n_primitives = 0;
        int active_sh_bases = 0;
        int total_bases_sh_rest = 0;
        int width = 0;
        int height = 0;
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        float near_plane = 0.0f;
        float far_plane = 0.0f;
        std::string error;

        static CrashParams load(const std::string& path) {
            CrashParams params;
            std::ifstream file(path);
            if (!file)
                return params;

            std::string line;
            while (std::getline(file, line)) {
                auto parse_int = [&](const char* key, int& value) {
                    const std::string pattern = std::string("\"") + key + "\":";
                    if (const auto pos = line.find(pattern); pos != std::string::npos) {
                        value = std::stoi(line.substr(line.find(':', pos) + 1));
                    }
                };
                auto parse_float = [&](const char* key, float& value) {
                    const std::string pattern = std::string("\"") + key + "\":";
                    if (const auto pos = line.find(pattern); pos != std::string::npos) {
                        value = std::stof(line.substr(line.find(':', pos) + 1));
                    }
                };
                auto parse_string = [&](const char* key, std::string& value) {
                    const std::string pattern = std::string("\"") + key + "\": \"";
                    if (const auto pos = line.find(pattern); pos != std::string::npos) {
                        const auto start = line.find(": \"", pos) + 3;
                        const auto end = line.rfind('"');
                        if (end > start)
                            value = line.substr(start, end - start);
                    }
                };

                parse_int("n_primitives", params.n_primitives);
                parse_int("active_sh_bases", params.active_sh_bases);
                parse_int("total_bases_sh_rest", params.total_bases_sh_rest);
                parse_int("width", params.width);
                parse_int("height", params.height);
                parse_float("fx", params.fx);
                parse_float("fy", params.fy);
                parse_float("cx", params.cx);
                parse_float("cy", params.cy);
                parse_float("near_plane", params.near_plane);
                parse_float("far_plane", params.far_plane);
                parse_string("error", params.error);
            }
            return params;
        }
    };

    struct TensorStats {
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();
        int nan_count = 0;
        int inf_count = 0;

        void update(const float v) {
            if (std::isnan(v))
                ++nan_count;
            else if (std::isinf(v))
                ++inf_count;
            else {
                min = std::min(min, v);
                max = std::max(max, v);
            }
        }
    };

    inline int div_up(const int a, const int b) { return (a + b - 1) / b; }

} // namespace

class RasterizerCrashDumpTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(std::string(CRASH_DUMP_PATH) + "/params.json")) {
            GTEST_SKIP() << "Crash dump not found at: " << CRASH_DUMP_PATH;
        }

        params_ = CrashParams::load(std::string(CRASH_DUMP_PATH) + "/params.json");

        LOG_INFO("Crash dump: error='{}', n_primitives={}, {}x{}, sh_bases={}",
                 params_.error, params_.n_primitives, params_.width, params_.height, params_.active_sh_bases);

        const std::string base_path = CRASH_DUMP_PATH;
        means_ = load_tensor(base_path + "/means.tensor").to(Device::CUDA);
        raw_scales_ = load_tensor(base_path + "/raw_scales.tensor").to(Device::CUDA);
        raw_rotations_ = load_tensor(base_path + "/raw_rotations.tensor").to(Device::CUDA);
        raw_opacities_ = load_tensor(base_path + "/raw_opacities.tensor").to(Device::CUDA);
        sh0_ = load_tensor(base_path + "/sh0.tensor").to(Device::CUDA);
        shN_ = load_tensor(base_path + "/shN.tensor").to(Device::CUDA);
        w2c_ = load_tensor(base_path + "/w2c.tensor").to(Device::CUDA);
        cam_position_ = load_tensor(base_path + "/cam_position.tensor").to(Device::CUDA);

        w2c_squeezed_ = w2c_.ndim() == 3 ? w2c_.squeeze(0) : w2c_;
    }

    fast_lfs::rasterization::ForwardContext run_forward(
        const int width, const int height,
        const float fx, const float fy, const float cx, const float cy,
        Tensor& image, Tensor& alpha) const {

        return fast_lfs::rasterization::forward_raw(
            means_.ptr<float>(), raw_scales_.ptr<float>(), raw_rotations_.ptr<float>(),
            raw_opacities_.ptr<float>(), sh0_.ptr<float>(), shN_.ptr<float>(),
            w2c_squeezed_.ptr<float>(), cam_position_.ptr<float>(),
            image.ptr<float>(), alpha.ptr<float>(),
            params_.n_primitives, params_.active_sh_bases, params_.total_bases_sh_rest,
            width, height, fx, fy, cx, cy, params_.near_plane, params_.far_plane);
    }

    CrashParams params_;
    Tensor means_, raw_scales_, raw_rotations_, raw_opacities_;
    Tensor sh0_, shN_, w2c_, cam_position_, w2c_squeezed_;
};

TEST_F(RasterizerCrashDumpTest, ReproduceCrash) {
    const int width = params_.width;
    const int height = params_.height;
    const int n_tiles = div_up(width, TILE_SIZE) * div_up(height, TILE_SIZE);

    LOG_INFO("Forward: {}x{} ({} tiles), {} primitives", width, height, n_tiles, params_.n_primitives);

    auto image = Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CUDA);
    auto alpha = Tensor::empty({1, static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CUDA);

    try {
        const auto ctx = run_forward(width, height, params_.fx, params_.fy, params_.cx, params_.cy, image, alpha);

        if (!ctx.success) {
            GlobalArenaManager::instance().get_arena().end_frame(ctx.frame_id);
            FAIL() << "Forward failed: " << ctx.error_message;
        }

        LOG_INFO("Forward succeeded: {} visible, {} instances, {} buckets",
                 ctx.n_visible_primitives, ctx.n_instances, ctx.n_buckets);

        GlobalArenaManager::instance().get_arena().end_frame(ctx.frame_id);
    } catch (const std::exception& e) {
        FAIL() << "std::exception: " << e.what();
    } catch (...) {
        FAIL() << "Unknown exception (crash reproduced!)";
    }
}

TEST_F(RasterizerCrashDumpTest, AnalyzeTensorValues) {
    auto analyze = [](const Tensor& t, const char* name) {
        const auto cpu = t.cpu();
        const float* data = cpu.ptr<float>();
        TensorStats stats;
        for (size_t i = 0; i < cpu.numel(); ++i)
            stats.update(data[i]);
        LOG_INFO("{}: min={:.4f}, max={:.4f}, NaN={}, Inf={}", name, stats.min, stats.max, stats.nan_count, stats.inf_count);
        return stats;
    };

    analyze(means_, "means");
    const auto scale_stats = analyze(raw_scales_, "raw_scales");
    LOG_INFO("  variance range: [{:.6f}, {:.4f}]", std::exp(2.0f * scale_stats.min), std::exp(2.0f * scale_stats.max));

    // Check for degenerate quaternions
    const auto rot_cpu = raw_rotations_.cpu();
    const float* rot_data = rot_cpu.ptr<float>();
    int degenerate_count = 0;
    for (int i = 0; i < params_.n_primitives; ++i) {
        const float norm_sq = rot_data[i * 4] * rot_data[i * 4] + rot_data[i * 4 + 1] * rot_data[i * 4 + 1] +
                              rot_data[i * 4 + 2] * rot_data[i * 4 + 2] + rot_data[i * 4 + 3] * rot_data[i * 4 + 3];
        if (norm_sq < DEGENERATE_QUAT_THRESHOLD)
            ++degenerate_count;
    }
    analyze(raw_rotations_, "raw_rotations");
    LOG_INFO("  degenerate quaternions: {}", degenerate_count);

    const auto opac_stats = analyze(raw_opacities_, "raw_opacities");
    const auto sigmoid = [](const float x) { return 1.0f / (1.0f + std::exp(-x)); };
    LOG_INFO("  opacity range: [{:.4f}, {:.4f}]", sigmoid(opac_stats.min), sigmoid(opac_stats.max));

    analyze(sh0_, "sh0");

    SUCCEED();
}

TEST_F(RasterizerCrashDumpTest, StressTest) {
    const int width = params_.width;
    const int height = params_.height;

    auto image = Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CUDA);
    auto alpha = Tensor::empty({1, static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CUDA);

    LOG_INFO("Running {} iterations...", STRESS_TEST_ITERATIONS);

    for (int iter = 0; iter < STRESS_TEST_ITERATIONS; ++iter) {
        try {
            const auto ctx = run_forward(width, height, params_.fx, params_.fy, params_.cx, params_.cy, image, alpha);

            GlobalArenaManager::instance().get_arena().end_frame(ctx.frame_id);

            if (!ctx.success) {
                FAIL() << "Iteration " << iter << " failed: " << ctx.error_message;
            }
        } catch (const std::exception& e) {
            FAIL() << "Iteration " << iter << " threw: " << e.what();
        } catch (...) {
            FAIL() << "Iteration " << iter << " threw unknown exception";
        }
    }

    LOG_INFO("All {} iterations completed", STRESS_TEST_ITERATIONS);
    SUCCEED();
}

TEST_F(RasterizerCrashDumpTest, HalfResolution) {
    const int width = params_.width / 2;
    const int height = params_.height / 2;
    const float fx = params_.fx / 2;
    const float fy = params_.fy / 2;
    const float cx = params_.cx / 2;
    const float cy = params_.cy / 2;

    LOG_INFO("Testing half resolution: {}x{}", width, height);

    auto image = Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CUDA);
    auto alpha = Tensor::empty({1, static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CUDA);

    try {
        const auto ctx = run_forward(width, height, fx, fy, cx, cy, image, alpha);

        if (!ctx.success) {
            GlobalArenaManager::instance().get_arena().end_frame(ctx.frame_id);
            FAIL() << "Half resolution failed: " << ctx.error_message;
        }

        LOG_INFO("Half resolution: {} visible, {} instances", ctx.n_visible_primitives, ctx.n_instances);
        GlobalArenaManager::instance().get_arena().end_frame(ctx.frame_id);
    } catch (const std::exception& e) {
        FAIL() << "Half resolution threw: " << e.what();
    } catch (...) {
        FAIL() << "Half resolution threw unknown exception";
    }
}
