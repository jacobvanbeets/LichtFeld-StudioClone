/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_compressed_ply.cpp
 * @brief Tests for compressed PLY format support
 *
 * Verifies that our compressed PLY loader produces identical results to
 * the reference splat-transform implementation.
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <cmath>

#include "io/formats/compressed_ply.hpp"
#include "io/formats/ply.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"

namespace fs = std::filesystem;

class CompressedPlyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test files created by splat-transform
        test_dir = fs::path("/home/paja/projects/gaussian-splatting-cuda/test_formats");
        compressed_ply = test_dir / "test.compressed.ply";
        decompressed_ply = test_dir / "test_decompressed.ply";

        // Create test directory if it doesn't exist
        if (!fs::exists(test_dir)) {
            fs::create_directories(test_dir);
        }
    }

    static bool floatNear(float a, float b, float tol = 1e-4f) {
        return std::abs(a - b) <= tol;
    }

    // Compare SplatData instances with tolerance for quantization error
    static void compareSplatData(const lfs::core::SplatData& compressed,
                                  const lfs::core::SplatData& reference,
                                  float tol = 0.01f) {
        ASSERT_EQ(compressed.size(), reference.size())
            << "Splat count mismatch";

        const size_t N = compressed.size();

        // Compare means (positions) - quantized to 32 bits per chunk
        auto comp_means = compressed.means().cpu();
        auto ref_means = reference.means().cpu();
        const float* comp_m = comp_means.ptr<float>();
        const float* ref_m = ref_means.ptr<float>();

        float max_pos_diff = 0.0f;
        for (size_t i = 0; i < N * 3; ++i) {
            max_pos_diff = std::max(max_pos_diff, std::abs(comp_m[i] - ref_m[i]));
        }
        std::cout << "  Max position difference: " << max_pos_diff << std::endl;
        EXPECT_LT(max_pos_diff, tol) << "Position reconstruction error too high";

        // Compare scales
        auto comp_scales = compressed.get_scaling().cpu();
        auto ref_scales = reference.get_scaling().cpu();
        const float* comp_s = comp_scales.ptr<float>();
        const float* ref_s = ref_scales.ptr<float>();

        float max_scale_diff = 0.0f;
        for (size_t i = 0; i < N * 3; ++i) {
            max_scale_diff = std::max(max_scale_diff, std::abs(comp_s[i] - ref_s[i]));
        }
        std::cout << "  Max scale difference: " << max_scale_diff << std::endl;
        EXPECT_LT(max_scale_diff, tol) << "Scale reconstruction error too high";

        // Compare rotations (quaternions)
        auto comp_rot = compressed.get_rotation().cpu();
        auto ref_rot = reference.get_rotation().cpu();
        const float* comp_r = comp_rot.ptr<float>();
        const float* ref_r = ref_rot.ptr<float>();

        float max_rot_diff = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            // Quaternions can be negated and still represent the same rotation
            float diff_pos = 0.0f, diff_neg = 0.0f;
            for (int j = 0; j < 4; ++j) {
                diff_pos += std::abs(comp_r[i * 4 + j] - ref_r[i * 4 + j]);
                diff_neg += std::abs(comp_r[i * 4 + j] + ref_r[i * 4 + j]);
            }
            max_rot_diff = std::max(max_rot_diff, std::min(diff_pos, diff_neg));
        }
        std::cout << "  Max rotation difference: " << max_rot_diff << std::endl;
        EXPECT_LT(max_rot_diff, tol * 4) << "Rotation reconstruction error too high";

        // Compare opacity (logit space)
        auto comp_op = compressed.get_opacity().cpu();
        auto ref_op = reference.get_opacity().cpu();
        const float* comp_o = comp_op.ptr<float>();
        const float* ref_o = ref_op.ptr<float>();

        float max_opacity_diff = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            max_opacity_diff = std::max(max_opacity_diff, std::abs(comp_o[i] - ref_o[i]));
        }
        std::cout << "  Max opacity difference: " << max_opacity_diff << std::endl;
        // Opacity has higher tolerance due to 8-bit quantization
        EXPECT_LT(max_opacity_diff, 0.5f) << "Opacity reconstruction error too high";

        // Compare SH0 (colors)
        auto comp_sh0 = compressed.sh0().cpu();
        auto ref_sh0 = reference.sh0().cpu();
        const float* comp_c = comp_sh0.ptr<float>();
        const float* ref_c = ref_sh0.ptr<float>();

        float max_color_diff = 0.0f;
        for (size_t i = 0; i < N * 3; ++i) {
            max_color_diff = std::max(max_color_diff, std::abs(comp_c[i] - ref_c[i]));
        }
        std::cout << "  Max color difference: " << max_color_diff << std::endl;
        EXPECT_LT(max_color_diff, 0.5f) << "Color reconstruction error too high";
    }

    fs::path test_dir;
    fs::path compressed_ply;
    fs::path decompressed_ply;
};

// Test: Detection of compressed PLY format
TEST_F(CompressedPlyTest, DetectCompressedFormat) {
    if (!fs::exists(compressed_ply)) {
        GTEST_SKIP() << "Test file not found: " << compressed_ply
            << "\nRun: node splat-transform/bin/cli.mjs output/splat_30000.ply test_formats/test.compressed.ply";
    }

    EXPECT_TRUE(lfs::io::is_compressed_ply(compressed_ply))
        << "Failed to detect compressed PLY format";

    // Standard PLY should not be detected as compressed
    if (fs::exists(decompressed_ply)) {
        EXPECT_FALSE(lfs::io::is_compressed_ply(decompressed_ply))
            << "Standard PLY incorrectly detected as compressed";
    }
}

// Test: Load compressed PLY
TEST_F(CompressedPlyTest, LoadCompressedPly) {
    if (!fs::exists(compressed_ply)) {
        GTEST_SKIP() << "Test file not found: " << compressed_ply;
    }

    auto result = lfs::io::load_compressed_ply(compressed_ply);
    ASSERT_TRUE(result.has_value()) << "Failed to load: " << result.error();

    const auto& splat = *result;
    std::cout << "Loaded compressed PLY: " << splat.size() << " splats" << std::endl;

    EXPECT_GT(splat.size(), 0) << "No splats loaded";
    EXPECT_TRUE(splat.means().is_valid()) << "Means tensor invalid";
    EXPECT_TRUE(splat.sh0().is_valid()) << "SH0 tensor invalid";
    EXPECT_TRUE(splat.get_scaling().is_valid()) << "Scaling tensor invalid";
    EXPECT_TRUE(splat.get_rotation().is_valid()) << "Rotation tensor invalid";
    EXPECT_TRUE(splat.get_opacity().is_valid()) << "Opacity tensor invalid";
}

// Test: Compare with reference decompression
TEST_F(CompressedPlyTest, CompareWithReference) {
    if (!fs::exists(compressed_ply)) {
        GTEST_SKIP() << "Compressed test file not found: " << compressed_ply;
    }
    if (!fs::exists(decompressed_ply)) {
        GTEST_SKIP() << "Reference file not found: " << decompressed_ply
            << "\nRun: node splat-transform/bin/cli.mjs test_formats/test.compressed.ply test_formats/test_decompressed.ply";
    }

    std::cout << "Loading compressed PLY with our loader..." << std::endl;
    auto comp_result = lfs::io::load_compressed_ply(compressed_ply);
    ASSERT_TRUE(comp_result.has_value()) << "Failed to load compressed: " << comp_result.error();

    std::cout << "Loading reference PLY from splat-transform..." << std::endl;
    auto ref_result = lfs::io::load_ply(decompressed_ply);
    ASSERT_TRUE(ref_result.has_value()) << "Failed to load reference: " << ref_result.error();

    std::cout << "Comparing " << comp_result->size() << " splats..." << std::endl;
    compareSplatData(*comp_result, *ref_result);

    std::cout << "Compressed PLY roundtrip verification PASSED" << std::endl;
}

// Test: File not found handling
TEST_F(CompressedPlyTest, FileNotFound) {
    auto result = lfs::io::load_compressed_ply("/nonexistent/path/file.ply");
    EXPECT_FALSE(result.has_value()) << "Should fail for nonexistent file";
}

// Test: Invalid file format
TEST_F(CompressedPlyTest, InvalidFormat) {
    // Try to load a regular PLY as compressed
    if (fs::exists(decompressed_ply)) {
        auto result = lfs::io::load_compressed_ply(decompressed_ply);
        EXPECT_FALSE(result.has_value()) << "Should fail for non-compressed PLY";
    }
}

// Test: Export and reimport roundtrip
TEST_F(CompressedPlyTest, ExportRoundtrip) {
    // Load original PLY
    fs::path original = "/home/paja/projects/gaussian-splatting-cuda/output/splat_30000.ply";
    if (!fs::exists(original)) {
        GTEST_SKIP() << "Original PLY not found: " << original;
    }

    auto orig_result = lfs::io::load_ply(original);
    ASSERT_TRUE(orig_result.has_value()) << "Failed to load original: " << orig_result.error();

    // Export as compressed PLY
    fs::path export_path = test_dir / "roundtrip_test.compressed.ply";
    lfs::io::CompressedPlyWriteOptions options{
        .output_path = export_path,
        .include_sh = false  // SH0 only for this test
    };

    auto write_result = lfs::io::write_compressed_ply(*orig_result, options);
    ASSERT_TRUE(write_result.has_value()) << "Failed to write: " << write_result.error();
    ASSERT_TRUE(fs::exists(export_path)) << "Export file not created";

    // Reimport the exported file
    auto reimport_result = lfs::io::load_compressed_ply(export_path);
    ASSERT_TRUE(reimport_result.has_value()) << "Failed to reimport: " << reimport_result.error();

    ASSERT_EQ(reimport_result->size(), orig_result->size()) << "Splat count mismatch after roundtrip";

    // Compare statistics (lossy compression, so exact match not expected)
    auto compute_stats = [](const float* data, size_t n) {
        float min_val = data[0], max_val = data[0];
        double sum = 0;
        for (size_t i = 0; i < n; ++i) {
            min_val = std::min(min_val, data[i]);
            max_val = std::max(max_val, data[i]);
            sum += data[i];
        }
        return std::make_tuple(min_val, max_val, sum / n);
    };

    size_t N = orig_result->size();

    // Compare positions
    auto orig_means = orig_result->means().cpu();
    auto reimp_means = reimport_result->means().cpu();
    auto [orig_pos_min, orig_pos_max, orig_pos_avg] = compute_stats(orig_means.ptr<float>(), N * 3);
    auto [reimp_pos_min, reimp_pos_max, reimp_pos_avg] = compute_stats(reimp_means.ptr<float>(), N * 3);

    std::cout << "\nRoundtrip position statistics:" << std::endl;
    std::cout << "  Original: min=" << orig_pos_min << ", max=" << orig_pos_max << ", avg=" << orig_pos_avg << std::endl;
    std::cout << "  Reimport: min=" << reimp_pos_min << ", max=" << reimp_pos_max << ", avg=" << reimp_pos_avg << std::endl;

    // Positions should be very close (lossy but chunk-based quantization)
    EXPECT_NEAR(reimp_pos_avg, orig_pos_avg, std::abs(orig_pos_avg) * 0.01f + 0.01f)
        << "Position average should be close after roundtrip";

    // Compare scales (log-space)
    auto orig_scales = orig_result->scaling_raw().cpu();
    auto reimp_scales = reimport_result->scaling_raw().cpu();
    auto [orig_scale_min, orig_scale_max, orig_scale_avg] = compute_stats(orig_scales.ptr<float>(), N * 3);
    auto [reimp_scale_min, reimp_scale_max, reimp_scale_avg] = compute_stats(reimp_scales.ptr<float>(), N * 3);

    std::cout << "\nRoundtrip scale statistics (log-space):" << std::endl;
    std::cout << "  Original: min=" << orig_scale_min << ", max=" << orig_scale_max << ", avg=" << orig_scale_avg << std::endl;
    std::cout << "  Reimport: min=" << reimp_scale_min << ", max=" << reimp_scale_max << ", avg=" << reimp_scale_avg << std::endl;

    // Scales should be very close
    EXPECT_NEAR(reimp_scale_avg, orig_scale_avg, std::abs(orig_scale_avg) * 0.05f + 0.1f)
        << "Scale average should be close after roundtrip";

    // Clean up
    fs::remove(export_path);

    std::cout << "Compressed PLY export roundtrip PASSED" << std::endl;
}

// Test: Debug print actual color values
TEST_F(CompressedPlyTest, DebugColorValues) {
    if (!fs::exists(compressed_ply)) {
        GTEST_SKIP() << "Compressed test file not found: " << compressed_ply;
    }
    if (!fs::exists(decompressed_ply)) {
        GTEST_SKIP() << "Reference file not found: " << decompressed_ply;
    }

    // Load all three: compressed, reference decompressed, and original
    auto comp_result = lfs::io::load_compressed_ply(compressed_ply);
    auto ref_result = lfs::io::load_ply(decompressed_ply);

    fs::path original = "/home/paja/projects/gaussian-splatting-cuda/output/splat_30000.ply";
    std::optional<lfs::core::SplatData> orig_result;
    if (fs::exists(original)) {
        auto r = lfs::io::load_ply(original);
        if (r) orig_result = std::move(*r);
    }

    ASSERT_TRUE(comp_result.has_value()) << "Failed to load compressed: " << comp_result.error();
    ASSERT_TRUE(ref_result.has_value()) << "Failed to load reference: " << ref_result.error();

    auto comp_sh0 = comp_result->sh0().cpu();
    auto ref_sh0 = ref_result->sh0().cpu();

    std::cout << "\nCompressed sh0 shape: [";
    for (size_t i = 0; i < comp_sh0.shape().rank(); ++i) {
        std::cout << comp_sh0.shape()[i];
        if (i < comp_sh0.shape().rank() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Reference sh0 shape: [";
    for (size_t i = 0; i < ref_sh0.shape().rank(); ++i) {
        std::cout << ref_sh0.shape()[i];
        if (i < ref_sh0.shape().rank() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    const float* comp_data = comp_sh0.ptr<float>();
    const float* ref_data = ref_sh0.ptr<float>();

    lfs::core::Tensor orig_sh0_cpu;
    if (orig_result) {
        orig_sh0_cpu = orig_result->sh0().cpu();
        std::cout << "Original sh0 shape: [";
        for (size_t i = 0; i < orig_sh0_cpu.shape().rank(); ++i) {
            std::cout << orig_sh0_cpu.shape()[i];
            if (i < orig_sh0_cpu.shape().rank() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    const float* orig_data = orig_sh0_cpu.is_valid() ? orig_sh0_cpu.ptr<float>() : nullptr;

    std::cout << "\nFirst 5 splats SH0 (R, G, B):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Splat " << i << ":" << std::endl;
        std::cout << "  Comp: " << comp_data[i*3+0] << ", " << comp_data[i*3+1] << ", " << comp_data[i*3+2] << std::endl;
        std::cout << "  Ref:  " << ref_data[i*3+0] << ", " << ref_data[i*3+1] << ", " << ref_data[i*3+2] << std::endl;
        if (orig_data) {
            std::cout << "  Orig: " << orig_data[i*3+0] << ", " << orig_data[i*3+1] << ", " << orig_data[i*3+2] << std::endl;
        }
    }

    // Also check opacity
    auto comp_op = comp_result->get_opacity().cpu();
    auto ref_op = ref_result->get_opacity().cpu();
    const float* comp_o = comp_op.ptr<float>();
    const float* ref_o = ref_op.ptr<float>();

    std::cout << "\nFirst 5 splats opacity:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  Splat " << i << ": Comp=" << comp_o[i] << ", Ref=" << ref_o[i] << std::endl;
    }

    // Compute statistics to verify overall distribution
    std::cout << "\n--- Overall Statistics (should match) ---\n";

    auto compute_stats = [](const float* data, size_t n) {
        float min_val = data[0], max_val = data[0], sum = 0;
        for (size_t i = 0; i < n; ++i) {
            min_val = std::min(min_val, data[i]);
            max_val = std::max(max_val, data[i]);
            sum += data[i];
        }
        return std::make_tuple(min_val, max_val, sum / n);
    };

    size_t N = comp_result->size();
    auto [comp_min, comp_max, comp_avg] = compute_stats(comp_data, N * 3);
    auto [ref_min, ref_max, ref_avg] = compute_stats(ref_data, N * 3);

    std::cout << "SH0 colors (Compressed): min=" << comp_min << ", max=" << comp_max << ", avg=" << comp_avg << std::endl;
    std::cout << "SH0 colors (Reference):  min=" << ref_min << ", max=" << ref_max << ", avg=" << ref_avg << std::endl;

    if (orig_data) {
        auto [orig_min, orig_max, orig_avg] = compute_stats(orig_data, N * 3);
        std::cout << "SH0 colors (Original):   min=" << orig_min << ", max=" << orig_max << ", avg=" << orig_avg << std::endl;
    }

    EXPECT_NEAR(comp_avg, ref_avg, 0.01f) << "Average SH0 should match between compressed and reference";
}
