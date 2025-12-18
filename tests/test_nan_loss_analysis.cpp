/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// NaN loss bug analysis tests using crash dump data
// These tests analyze tensor values to identify the source of numerical instability

#include "core/cuda/memory_arena.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "rasterization/fastgs/rasterization/include/rasterization_api.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;

namespace {

    // Crash dump paths - these are developer dumps, test skips if not found
    const std::vector<std::string> CRASH_DUMP_PATHS = {
        "crash_dump_20251211_181320_944", // NaN at iteration 50434
        "crash_dump_20251211_173951_611", // NaN at iteration 30863
    };

    constexpr int TILE_SIZE = 16;
    constexpr float DEGENERATE_QUAT_THRESHOLD = 1e-8f;
    constexpr float EXTREME_SCALE_THRESHOLD = 20.0f; // exp(2*20) = huge variance
    constexpr float TINY_SCALE_THRESHOLD = -20.0f;   // exp(2*-20) = tiny variance

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
        int iteration = 0;

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

                // Extract iteration from error message
                if (!params.error.empty()) {
                    auto pos = params.error.find("iteration ");
                    if (pos != std::string::npos) {
                        params.iteration = std::stoi(params.error.substr(pos + 10));
                    }
                }
            }
            return params;
        }
    };

    struct TensorStats {
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();
        double sum = 0.0;
        double sum_sq = 0.0;
        int nan_count = 0;
        int inf_count = 0;
        int neg_inf_count = 0;
        size_t total = 0;

        void update(const float v) {
            ++total;
            if (std::isnan(v)) {
                ++nan_count;
            } else if (std::isinf(v)) {
                if (v > 0)
                    ++inf_count;
                else
                    ++neg_inf_count;
            } else {
                min = std::min(min, v);
                max = std::max(max, v);
                sum += v;
                sum_sq += static_cast<double>(v) * v;
            }
        }

        float mean() const {
            const size_t valid = total - nan_count - inf_count - neg_inf_count;
            return valid > 0 ? static_cast<float>(sum / valid) : 0.0f;
        }

        float std_dev() const {
            const size_t valid = total - nan_count - inf_count - neg_inf_count;
            if (valid < 2)
                return 0.0f;
            const double var = (sum_sq - sum * sum / valid) / (valid - 1);
            return static_cast<float>(std::sqrt(std::max(0.0, var)));
        }

        bool has_invalid() const { return nan_count > 0 || inf_count > 0 || neg_inf_count > 0; }
    };

    inline int div_up(const int a, const int b) { return (a + b - 1) / b; }

} // namespace

class NaNLossAnalysisTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        dump_path_ = GetParam();
        if (!std::filesystem::exists(dump_path_ + "/params.json")) {
            GTEST_SKIP() << "Crash dump not found at: " << dump_path_;
        }

        params_ = CrashParams::load(dump_path_ + "/params.json");
        LOG_INFO("=== Analyzing crash dump: {} ===", dump_path_);
        LOG_INFO("Error: {}", params_.error);
        LOG_INFO("Primitives: {}, Resolution: {}x{}, SH bases: {}",
                 params_.n_primitives, params_.width, params_.height, params_.active_sh_bases);

        // Load all tensors
        means_ = load_tensor(dump_path_ + "/means.tensor").to(Device::CUDA);
        raw_scales_ = load_tensor(dump_path_ + "/raw_scales.tensor").to(Device::CUDA);
        raw_rotations_ = load_tensor(dump_path_ + "/raw_rotations.tensor").to(Device::CUDA);
        raw_opacities_ = load_tensor(dump_path_ + "/raw_opacities.tensor").to(Device::CUDA);
        sh0_ = load_tensor(dump_path_ + "/sh0.tensor").to(Device::CUDA);
        shN_ = load_tensor(dump_path_ + "/shN.tensor").to(Device::CUDA);
        w2c_ = load_tensor(dump_path_ + "/w2c.tensor").to(Device::CUDA);
        cam_position_ = load_tensor(dump_path_ + "/cam_position.tensor").to(Device::CUDA);

        w2c_squeezed_ = w2c_.ndim() == 3 ? w2c_.squeeze(0) : w2c_;
    }

    TensorStats analyze_tensor(const Tensor& t, const char* name) {
        const auto cpu = t.cpu();
        const float* data = cpu.ptr<float>();
        TensorStats stats;
        for (size_t i = 0; i < cpu.numel(); ++i) {
            stats.update(data[i]);
        }
        LOG_INFO("{}: min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}, NaN={}, +Inf={}, -Inf={}",
                 name, stats.min, stats.max, stats.mean(), stats.std_dev(),
                 stats.nan_count, stats.inf_count, stats.neg_inf_count);
        return stats;
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

    std::string dump_path_;
    CrashParams params_;
    Tensor means_, raw_scales_, raw_rotations_, raw_opacities_;
    Tensor sh0_, shN_, w2c_, cam_position_, w2c_squeezed_;
};

// Check all input tensors for NaN/Inf values
TEST_P(NaNLossAnalysisTest, CheckInputTensorsForInvalidValues) {
    LOG_INFO("--- Checking input tensors for NaN/Inf ---");

    bool any_invalid = false;

    auto check = [&](const Tensor& t, const char* name) {
        auto stats = analyze_tensor(t, name);
        if (stats.has_invalid()) {
            LOG_ERROR("FOUND INVALID VALUES in {}", name);
            any_invalid = true;
        }
        return stats;
    };

    check(means_, "means");
    check(raw_scales_, "raw_scales");
    check(raw_rotations_, "raw_rotations");
    check(raw_opacities_, "raw_opacities");
    check(sh0_, "sh0");
    check(shN_, "shN");
    check(w2c_, "w2c");
    check(cam_position_, "cam_position");

    if (any_invalid) {
        LOG_ERROR("Input tensors contain NaN/Inf - this is the bug source!");
    } else {
        LOG_INFO("All input tensors are valid (no NaN/Inf)");
    }
}

// Analyze scale values for numerical instability
TEST_P(NaNLossAnalysisTest, AnalyzeScaleDistribution) {
    LOG_INFO("--- Analyzing scale distribution ---");

    const auto cpu = raw_scales_.cpu();
    const float* data = cpu.ptr<float>();
    const size_t n = cpu.numel();

    int extreme_large = 0;
    int extreme_small = 0;
    float max_scale = std::numeric_limits<float>::lowest();
    float min_scale = std::numeric_limits<float>::max();

    for (size_t i = 0; i < n; ++i) {
        const float v = data[i];
        if (std::isfinite(v)) {
            max_scale = std::max(max_scale, v);
            min_scale = std::min(min_scale, v);
            if (v > EXTREME_SCALE_THRESHOLD)
                ++extreme_large;
            if (v < TINY_SCALE_THRESHOLD)
                ++extreme_small;
        }
    }

    // Compute actual variance range
    const float min_var = std::exp(2.0f * min_scale);
    const float max_var = std::exp(2.0f * max_scale);

    LOG_INFO("Raw scale range: [{:.4f}, {:.4f}]", min_scale, max_scale);
    LOG_INFO("Variance range: [{:.2e}, {:.2e}]", min_var, max_var);
    LOG_INFO("Extreme large scales (>{:.1f}): {}", EXTREME_SCALE_THRESHOLD, extreme_large);
    LOG_INFO("Extreme small scales (<{:.1f}): {}", TINY_SCALE_THRESHOLD, extreme_small);

    if (extreme_large > 0 || extreme_small > 0) {
        LOG_WARN("Found {} Gaussians with extreme scale values", extreme_large + extreme_small);
    }
}

// Analyze quaternion values for degenerate cases
TEST_P(NaNLossAnalysisTest, AnalyzeQuaternionDistribution) {
    LOG_INFO("--- Analyzing quaternion distribution ---");

    const auto cpu = raw_rotations_.cpu();
    const float* data = cpu.ptr<float>();
    const int n = params_.n_primitives;

    int degenerate_count = 0;
    int nan_quat_count = 0;
    float min_norm_sq = std::numeric_limits<float>::max();
    float max_norm_sq = 0.0f;
    std::vector<int> degenerate_indices;

    for (int i = 0; i < n; ++i) {
        const float* q = data + i * 4;
        bool has_nan = false;
        for (int j = 0; j < 4; ++j) {
            if (std::isnan(q[j]) || std::isinf(q[j])) {
                has_nan = true;
                break;
            }
        }

        if (has_nan) {
            ++nan_quat_count;
            continue;
        }

        const float norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        min_norm_sq = std::min(min_norm_sq, norm_sq);
        max_norm_sq = std::max(max_norm_sq, norm_sq);

        if (norm_sq < DEGENERATE_QUAT_THRESHOLD) {
            ++degenerate_count;
            if (degenerate_indices.size() < 10) {
                degenerate_indices.push_back(i);
            }
        }
    }

    LOG_INFO("Quaternion norm^2 range: [{:.2e}, {:.2e}]", min_norm_sq, max_norm_sq);
    LOG_INFO("Degenerate quaternions (norm^2 < {:.0e}): {}", DEGENERATE_QUAT_THRESHOLD, degenerate_count);
    LOG_INFO("NaN/Inf quaternions: {}", nan_quat_count);

    if (!degenerate_indices.empty()) {
        std::string indices_str;
        for (int idx : degenerate_indices) {
            indices_str += std::to_string(idx) + " ";
        }
        LOG_INFO("First degenerate indices: {}", indices_str);
    }

    if (degenerate_count > 0) {
        LOG_WARN("Degenerate quaternions can cause NaN in covariance computation!");
    }
}

// Analyze opacity distribution
TEST_P(NaNLossAnalysisTest, AnalyzeOpacityDistribution) {
    LOG_INFO("--- Analyzing opacity distribution ---");

    const auto cpu = raw_opacities_.cpu();
    const float* data = cpu.ptr<float>();
    const size_t n = cpu.numel();

    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };

    TensorStats stats;
    int saturated_high = 0; // opacity > 0.999
    int saturated_low = 0;  // opacity < 0.001

    for (size_t i = 0; i < n; ++i) {
        stats.update(data[i]);
        if (std::isfinite(data[i])) {
            float opacity = sigmoid(data[i]);
            if (opacity > 0.999f)
                ++saturated_high;
            if (opacity < 0.001f)
                ++saturated_low;
        }
    }

    LOG_INFO("Raw opacity range: [{:.4f}, {:.4f}]", stats.min, stats.max);
    LOG_INFO("Actual opacity range: [{:.6f}, {:.6f}]", sigmoid(stats.min), sigmoid(stats.max));
    LOG_INFO("Saturated high (>0.999): {}", saturated_high);
    LOG_INFO("Saturated low (<0.001): {}", saturated_low);
    LOG_INFO("NaN: {}, Inf: {}", stats.nan_count, stats.inf_count);
}

// Analyze SH coefficients
TEST_P(NaNLossAnalysisTest, AnalyzeSHCoefficients) {
    LOG_INFO("--- Analyzing SH coefficients ---");

    auto sh0_stats = analyze_tensor(sh0_, "sh0 (DC)");
    auto shN_stats = analyze_tensor(shN_, "shN (higher order)");

    // Check for extremely large SH values that could cause color overflow
    constexpr float SH_WARN_THRESHOLD = 10.0f;

    if (std::abs(sh0_stats.min) > SH_WARN_THRESHOLD || std::abs(sh0_stats.max) > SH_WARN_THRESHOLD) {
        LOG_WARN("SH0 has extreme values - could cause color overflow");
    }
    if (std::abs(shN_stats.min) > SH_WARN_THRESHOLD || std::abs(shN_stats.max) > SH_WARN_THRESHOLD) {
        LOG_WARN("SHN has extreme values - could cause color overflow");
    }
}

// Analyze positions (means)
TEST_P(NaNLossAnalysisTest, AnalyzePositionDistribution) {
    LOG_INFO("--- Analyzing position distribution ---");

    const auto cpu = means_.cpu();
    const float* data = cpu.ptr<float>();
    const int n = params_.n_primitives;

    float min_x = std::numeric_limits<float>::max(), max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max(), max_y = std::numeric_limits<float>::lowest();
    float min_z = std::numeric_limits<float>::max(), max_z = std::numeric_limits<float>::lowest();
    int behind_camera = 0;

    // Get camera position
    auto cam_cpu = cam_position_.cpu();
    const float* cam_pos = cam_cpu.ptr<float>();

    for (int i = 0; i < n; ++i) {
        const float x = data[i * 3 + 0];
        const float y = data[i * 3 + 1];
        const float z = data[i * 3 + 2];

        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
            min_z = std::min(min_z, z);
            max_z = std::max(max_z, z);
        }
    }

    LOG_INFO("Position X range: [{:.4f}, {:.4f}]", min_x, max_x);
    LOG_INFO("Position Y range: [{:.4f}, {:.4f}]", min_y, max_y);
    LOG_INFO("Position Z range: [{:.4f}, {:.4f}]", min_z, max_z);
    LOG_INFO("Camera position: ({:.4f}, {:.4f}, {:.4f})", cam_pos[0], cam_pos[1], cam_pos[2]);

    // Compute scene extent
    float extent = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
    LOG_INFO("Scene extent: {:.4f}", extent);
}

// Run forward pass and check output for NaN
TEST_P(NaNLossAnalysisTest, CheckForwardOutputForNaN) {
    LOG_INFO("--- Running forward pass and checking output ---");

    auto image = Tensor::empty({3, static_cast<size_t>(params_.height), static_cast<size_t>(params_.width)}, Device::CUDA);
    auto alpha = Tensor::empty({1, static_cast<size_t>(params_.height), static_cast<size_t>(params_.width)}, Device::CUDA);

    try {
        auto ctx = run_forward(params_.width, params_.height, params_.fx, params_.fy,
                               params_.cx, params_.cy, image, alpha);

        if (!ctx.success) {
            GlobalArenaManager::instance().get_arena().end_frame(ctx.frame_id);
            LOG_ERROR("Forward failed: {}", ctx.error_message);
            FAIL() << "Forward failed: " << ctx.error_message;
        }

        LOG_INFO("Forward succeeded: {} visible, {} instances", ctx.n_visible_primitives, ctx.n_instances);

        // Check output for NaN
        auto image_stats = analyze_tensor(image, "output_image");
        auto alpha_stats = analyze_tensor(alpha, "output_alpha");

        GlobalArenaManager::instance().get_arena().end_frame(ctx.frame_id);

        if (image_stats.has_invalid()) {
            LOG_ERROR("OUTPUT IMAGE CONTAINS NaN/Inf!");
        }
        if (alpha_stats.has_invalid()) {
            LOG_ERROR("OUTPUT ALPHA CONTAINS NaN/Inf!");
        }

        // This is expected to potentially have NaN based on the crash
        EXPECT_TRUE(true); // Don't fail - we're analyzing

    } catch (const std::exception& e) {
        LOG_ERROR("Forward threw exception: {}", e.what());
        FAIL() << "Forward threw: " << e.what();
    }
}

// Find specific Gaussians that might be causing issues
TEST_P(NaNLossAnalysisTest, FindProblematicGaussians) {
    LOG_INFO("--- Finding problematic Gaussians ---");

    const auto means_cpu = means_.cpu();
    const auto scales_cpu = raw_scales_.cpu();
    const auto rotations_cpu = raw_rotations_.cpu();
    const auto opacities_cpu = raw_opacities_.cpu();

    const float* means_data = means_cpu.ptr<float>();
    const float* scales_data = scales_cpu.ptr<float>();
    const float* rot_data = rotations_cpu.ptr<float>();
    const float* opac_data = opacities_cpu.ptr<float>();

    const int n = params_.n_primitives;
    int problematic_count = 0;

    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };

    for (int i = 0; i < n && problematic_count < 20; ++i) {
        bool problematic = false;
        std::string reason;

        // Check for NaN in any parameter
        for (int j = 0; j < 3; ++j) {
            if (!std::isfinite(means_data[i * 3 + j])) {
                problematic = true;
                reason += "NaN/Inf position; ";
                break;
            }
        }

        for (int j = 0; j < 3; ++j) {
            if (!std::isfinite(scales_data[i * 3 + j])) {
                problematic = true;
                reason += "NaN/Inf scale; ";
                break;
            } else if (scales_data[i * 3 + j] > EXTREME_SCALE_THRESHOLD) {
                problematic = true;
                reason += "extreme scale; ";
                break;
            }
        }

        // Check quaternion
        float norm_sq = 0;
        for (int j = 0; j < 4; ++j) {
            if (!std::isfinite(rot_data[i * 4 + j])) {
                problematic = true;
                reason += "NaN/Inf rotation; ";
                break;
            }
            norm_sq += rot_data[i * 4 + j] * rot_data[i * 4 + j];
        }
        if (norm_sq < DEGENERATE_QUAT_THRESHOLD) {
            problematic = true;
            reason += "degenerate quaternion; ";
        }

        if (!std::isfinite(opac_data[i])) {
            problematic = true;
            reason += "NaN/Inf opacity; ";
        }

        if (problematic) {
            LOG_WARN("Gaussian {}: {}", i, reason);
            LOG_INFO("  pos=({:.4f}, {:.4f}, {:.4f})",
                     means_data[i * 3], means_data[i * 3 + 1], means_data[i * 3 + 2]);
            LOG_INFO("  scale=({:.4f}, {:.4f}, {:.4f})",
                     scales_data[i * 3], scales_data[i * 3 + 1], scales_data[i * 3 + 2]);
            LOG_INFO("  rot=({:.4f}, {:.4f}, {:.4f}, {:.4f}), norm^2={:.2e}",
                     rot_data[i * 4], rot_data[i * 4 + 1], rot_data[i * 4 + 2], rot_data[i * 4 + 3], norm_sq);
            LOG_INFO("  raw_opacity={:.4f}, opacity={:.6f}",
                     opac_data[i], sigmoid(opac_data[i]));
            ++problematic_count;
        }
    }

    if (problematic_count == 0) {
        LOG_INFO("No obviously problematic Gaussians found in first scan");
    } else {
        LOG_WARN("Found {} problematic Gaussians (showing first 20)", problematic_count);
    }
}

// Statistical summary comparing both dumps
TEST_P(NaNLossAnalysisTest, PrintSummary) {
    LOG_INFO("=== SUMMARY for {} ===", dump_path_);
    LOG_INFO("Crash at iteration: {}", params_.iteration);
    LOG_INFO("Primitives: {}", params_.n_primitives);
    LOG_INFO("Resolution: {}x{}", params_.width, params_.height);
    LOG_INFO("SH degree: {} (bases={})", static_cast<int>(std::sqrt(params_.active_sh_bases)) - 1, params_.active_sh_bases);

    // Quick tensor validity check
    bool means_valid = !means_.has_nan() && !means_.has_inf();
    bool scales_valid = !raw_scales_.has_nan() && !raw_scales_.has_inf();
    bool rot_valid = !raw_rotations_.has_nan() && !raw_rotations_.has_inf();
    bool opac_valid = !raw_opacities_.has_nan() && !raw_opacities_.has_inf();
    bool sh0_valid = !sh0_.has_nan() && !sh0_.has_inf();
    bool shN_valid = !shN_.has_nan() && !shN_.has_inf();

    LOG_INFO("Tensor validity: means={}, scales={}, rot={}, opac={}, sh0={}, shN={}",
             means_valid ? "OK" : "INVALID",
             scales_valid ? "OK" : "INVALID",
             rot_valid ? "OK" : "INVALID",
             opac_valid ? "OK" : "INVALID",
             sh0_valid ? "OK" : "INVALID",
             shN_valid ? "OK" : "INVALID");

    if (!means_valid || !scales_valid || !rot_valid || !opac_valid || !sh0_valid || !shN_valid) {
        LOG_ERROR("*** INPUT TENSORS CONTAIN NaN/Inf - BUG SOURCE IDENTIFIED ***");
    }
}

INSTANTIATE_TEST_SUITE_P(
    CrashDumps,
    NaNLossAnalysisTest,
    ::testing::ValuesIn(CRASH_DUMP_PATHS),
    [](const testing::TestParamInfo<std::string>& info) {
        // Extract timestamp from path for test name
        std::string name = info.param;
        std::replace(name.begin(), name.end(), '/', '_');
        std::replace(name.begin(), name.end(), '.', '_');
        return name;
    });
