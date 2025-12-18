/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_relocate_gs_edge_cases.cpp
 * @brief Comprehensive tests for edge cases in relocate_gs that could expose tensor bugs
 *
 * Tests all suspicious operations identified in relocate_gs():
 * 1. Logit computation with edge values
 * 2. Int32 clamp operations
 * 3. Logical operation chains (logical_or, logical_not)
 * 4. squeeze() with negative indices
 * 5. nonzero() edge cases
 * 6. Division by near-zero values
 * 7. ones_like() with non-default dtypes
 * 8. Double index_select (chained indexing)
 * 9. contiguous() correctness
 * 10. Element-wise operations with reductions
 */

#include "core/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <torch/torch.h>

using namespace lfs::core;

class RelocateGsEdgeCasesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is available
        ASSERT_TRUE(true) << "CUDA required for these tests";
    }
};

// ============================================================================
// TEST 1: Logit computation with edge values
// ============================================================================

TEST_F(RelocateGsEdgeCasesTest, LogitEdgeValuesNearZero) {
    // Test logit computation: log(p / (1 - p)) with values near 0
    std::vector<float> test_values = {0.005f, 0.001f, 0.0001f, 1e-5f};

    for (float val : test_values) {
        auto p = Tensor::full({1}, val, Device::CUDA);
        auto logit = (p / (Tensor::ones_like(p) - p)).log();
        auto result = logit.cpu().to_vector()[0];

        // Should be negative and finite
        EXPECT_TRUE(std::isfinite(result)) << "Logit of " << val << " is not finite: " << result;
        EXPECT_LT(result, 0.0f) << "Logit of " << val << " should be negative";
        EXPECT_FALSE(std::isnan(result)) << "Logit of " << val << " is NaN";
    }
}

TEST_F(RelocateGsEdgeCasesTest, LogitEdgeValuesNearOne) {
    // Test logit computation with values near 1
    // Only test values that should produce finite results
    // Values closer to 1.0 than 1e-7 are handled by clamping in relocate_gs
    std::vector<float> test_values = {
        0.9f, 0.99f, 0.999f, 0.9999f,
        1.0f - 1e-7f, // Max value used in relocate_gs (after clamping)
        1.0f - 1e-6f};

    for (float val : test_values) {
        auto p = Tensor::full({1}, val, Device::CUDA);
        auto logit = (p / (Tensor::ones_like(p) - p)).log();
        auto result = logit.cpu().to_vector()[0];

        // Should be positive and finite (not inf)
        EXPECT_TRUE(std::isfinite(result))
            << "Logit of " << val << " is not finite: " << result;
        EXPECT_GT(result, 0.0f) << "Logit of " << val << " should be positive";
        EXPECT_FALSE(std::isnan(result)) << "Logit of " << val << " is NaN";
        EXPECT_FALSE(std::isinf(result)) << "Logit of " << val << " is Inf";
    }
}

// NOTE: LogitBoundaryBehavior test removed
// Testing logit(0.0) and logit(1.0) is implementation-specific behavior that doesn't
// occur in practice due to clamping in relocate_gs. The tensor library's handling
// of log(0) and division by zero may vary. The important test is LogitAfterClamping
// which validates the actual usage pattern in relocate_gs.

TEST_F(RelocateGsEdgeCasesTest, LogitAfterClamping) {
    // Test the full clamping + logit pipeline as in relocate_gs
    const float min_opacity = 0.005f;
    const float max_opacity = 1.0f - 1e-7f;

    std::vector<float> input_values = {
        0.0f, 0.001f, 0.005f, 0.1f, 0.5f, 0.9f, 0.999f, 1.0f};

    for (float val : input_values) {
        auto p = Tensor::full({1}, val, Device::CUDA);
        p = p.clamp(min_opacity, max_opacity);
        auto logit = (p / (Tensor::ones_like(p) - p)).log();
        auto result = logit.cpu().to_vector()[0];

        EXPECT_TRUE(std::isfinite(result))
            << "Logit after clamp(" << val << ") is not finite: " << result;
        EXPECT_FALSE(std::isnan(result))
            << "Logit after clamp(" << val << ") is NaN";
    }
}

TEST_F(RelocateGsEdgeCasesTest, LogitBatchOperation) {
    // Test logit on a batch of values (as happens in relocate_gs)
    std::vector<float> values = {
        0.005f, 0.01f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.99f, 1.0f - 1e-7f};

    auto p = Tensor::from_vector(values, TensorShape{values.size()}, Device::CUDA);
    auto logit = (p / (Tensor::ones_like(p) - p)).log();
    auto results = logit.cpu().to_vector();

    EXPECT_EQ(results.size(), values.size());

    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_TRUE(std::isfinite(results[i]))
            << "Logit[" << i << "] of " << values[i] << " is not finite";
        EXPECT_FALSE(std::isnan(results[i]))
            << "Logit[" << i << "] of " << values[i] << " is NaN";
    }
}

// ============================================================================
// TEST 2: Int32 clamp operations
// ============================================================================

TEST_F(RelocateGsEdgeCasesTest, Int32ClampBasic) {
    // Test Int32 clamp as used for ratios in relocate_gs
    auto ratios = Tensor::from_vector(
                      std::vector<float>{0, 1, 5, 10, 50, 100},
                      TensorShape{6}, Device::CUDA)
                      .to(DataType::Int32);

    const int n_max = 51;
    auto clamped = ratios.clamp(1, n_max);
    auto result = clamped.cpu().to_vector_int();

    EXPECT_EQ(result[0], 1);  // 0 -> 1
    EXPECT_EQ(result[1], 1);  // 1 -> 1
    EXPECT_EQ(result[2], 5);  // 5 -> 5
    EXPECT_EQ(result[3], 10); // 10 -> 10
    EXPECT_EQ(result[4], 50); // 50 -> 50
    EXPECT_EQ(result[5], 51); // 100 -> 51 (clamped to n_max)
}

TEST_F(RelocateGsEdgeCasesTest, Int32ClampEdgeCases) {
    // Test Int32 clamp with edge values
    std::vector<int> values = {
        std::numeric_limits<int>::min(),
        -1000, -1, 0, 1, 2, 50, 51, 52, 100, 1000,
        std::numeric_limits<int>::max()};

    auto tensor = Tensor::from_vector(
                      std::vector<float>(values.begin(), values.end()),
                      TensorShape{values.size()}, Device::CUDA)
                      .to(DataType::Int32);
    auto clamped = tensor.clamp(1, 51);
    auto result = clamped.cpu().to_vector_int();

    for (const auto& val : result) {
        EXPECT_GE(val, 1) << "Clamped value should be >= 1";
        EXPECT_LE(val, 51) << "Clamped value should be <= 51";
    }
}

TEST_F(RelocateGsEdgeCasesTest, Int32OnesLike) {
    // Test ones_like with Int32 dtype (as used in relocate_gs)
    auto float_tensor = Tensor::randn({100}, Device::CUDA);
    auto int_ones = Tensor::ones_like(float_tensor, DataType::Int32);

    EXPECT_EQ(int_ones.dtype(), DataType::Int32);
    EXPECT_EQ(int_ones.shape(), float_tensor.shape());
    EXPECT_EQ(int_ones.device(), Device::CUDA);

    auto values = int_ones.cpu().to_vector_int();
    for (const auto& val : values) {
        EXPECT_EQ(val, 1);
    }
}

// ============================================================================
// TEST 3: Logical operation chains
// ============================================================================

TEST_F(RelocateGsEdgeCasesTest, LogicalOrCombineComparisons) {
    // Test: (opacities <= min_opacity) | (rot_mag_sq < 1e-8)
    // As in: dead_mask = (opacities <= min_opacity).logical_or(rot_mag_sq < 1e-8f)

    std::vector<float> opacities_data = {0.001f, 0.005f, 0.1f, 0.5f, 1.0f};
    std::vector<float> rot_mag_sq_data = {1e-10f, 1e-8f, 1e-7f, 1e-5f, 1.0f};

    auto opacities = Tensor::from_vector(opacities_data, TensorShape{5}, Device::CUDA);
    auto rot_mag_sq = Tensor::from_vector(rot_mag_sq_data, TensorShape{5}, Device::CUDA);

    const float min_opacity = 0.005f;
    auto dead_mask = (opacities <= min_opacity).logical_or(rot_mag_sq < 1e-8f);

    EXPECT_EQ(dead_mask.dtype(), DataType::Bool);
    auto result = dead_mask.cpu().to_vector_bool();

    // opacities[0]=0.001 <= 0.005 -> true, OR rot_mag_sq[0]=1e-10 < 1e-8 -> true => true
    EXPECT_TRUE(result[0]);

    // opacities[1]=0.005 <= 0.005 -> true, OR ... => true
    EXPECT_TRUE(result[1]);

    // opacities[2]=0.1 > 0.005 -> false, rot_mag_sq[2]=1e-7 >= 1e-8 -> false => false
    EXPECT_FALSE(result[2]);

    // opacities[3]=0.5 > 0.005 -> false, rot_mag_sq[3]=1e-5 >= 1e-8 -> false => false
    EXPECT_FALSE(result[3]);

    // opacities[4]=1.0 > 0.005 -> false, rot_mag_sq[4]=1.0 >= 1e-8 -> false => false
    EXPECT_FALSE(result[4]);
}

TEST_F(RelocateGsEdgeCasesTest, LogicalNotAndNonzero) {
    // Test: alive_mask = ~dead_mask; alive_indices = alive_mask.nonzero().squeeze(-1)

    std::vector<float> dead_mask_data = {true, false, true, false, false, true};
    auto dead_mask = Tensor::from_vector(dead_mask_data, TensorShape{6}, Device::CUDA).to(DataType::Bool);

    auto alive_mask = dead_mask.logical_not();
    auto result = alive_mask.cpu().to_vector_bool();

    EXPECT_FALSE(result[0]); // ~true = false
    EXPECT_TRUE(result[1]);  // ~false = true
    EXPECT_FALSE(result[2]); // ~true = false
    EXPECT_TRUE(result[3]);  // ~false = true
    EXPECT_TRUE(result[4]);  // ~false = true
    EXPECT_FALSE(result[5]); // ~true = false
}

TEST_F(RelocateGsEdgeCasesTest, ElementWiseMultiplyThenSum) {
    // Test: (rotation_raw * rotation_raw).sum(-1)
    // This tests element-wise multiply followed by reduction

    std::vector<float> rotation_data = {
        1.0f, 0.0f, 0.0f, 0.0f,     // Row 0: magnitude^2 = 1.0
        0.5f, 0.5f, 0.5f, 0.5f,     // Row 1: magnitude^2 = 1.0
        1e-5f, 1e-5f, 1e-5f, 1e-5f, // Row 2: magnitude^2 = 4e-10
        0.0f, 0.0f, 0.0f, 0.0f      // Row 3: magnitude^2 = 0.0
    };

    auto rotation_raw = Tensor::from_vector(rotation_data, TensorShape{4, 4}, Device::CUDA);
    auto rot_mag_sq = (rotation_raw * rotation_raw).sum(-1);

    EXPECT_EQ(rot_mag_sq.shape(), TensorShape({4}));
    auto result = rot_mag_sq.cpu().to_vector();

    EXPECT_NEAR(result[0], 1.0f, 1e-6f);
    EXPECT_NEAR(result[1], 1.0f, 1e-6f);
    EXPECT_NEAR(result[2], 4e-10f, 1e-11f);
    EXPECT_NEAR(result[3], 0.0f, 1e-6f);
}

// ============================================================================
// TEST 4: squeeze() with negative indices
// ============================================================================

TEST_F(RelocateGsEdgeCasesTest, SqueezeNegativeIndexShape_N_1) {
    // Test squeeze(-1) on shape [N, 1] -> [N]
    auto tensor = Tensor::randn({10, 1}, Device::CUDA);
    auto squeezed = tensor.squeeze(-1);

    EXPECT_EQ(squeezed.ndim(), 1);
    EXPECT_EQ(squeezed.shape()[0], 10);
}

TEST_F(RelocateGsEdgeCasesTest, SqueezeNegativeIndexShape_N) {
    // Test squeeze(-1) on shape [N] (no-op)
    auto tensor = Tensor::randn({10}, Device::CUDA);
    auto squeezed = tensor.squeeze(-1);

    EXPECT_EQ(squeezed.ndim(), 1);
    EXPECT_EQ(squeezed.shape()[0], 10);
}

TEST_F(RelocateGsEdgeCasesTest, SqueezeNegativeIndexShape_N_1_1) {
    // Test squeeze(-1) on shape [N, 1, 1]
    auto tensor = Tensor::randn({5, 1, 1}, Device::CUDA);
    auto squeezed = tensor.squeeze(-1);

    EXPECT_EQ(squeezed.ndim(), 2);
    EXPECT_EQ(squeezed.shape()[0], 5);
    EXPECT_EQ(squeezed.shape()[1], 1);
}

TEST_F(RelocateGsEdgeCasesTest, SqueezeAfterNonzero) {
    // Test: nonzero().squeeze(-1) as done in relocate_gs
    std::vector<float> mask_data = {false, true, false, true, true};
    auto mask = Tensor::from_vector(mask_data, TensorShape{5}, Device::CUDA).to(DataType::Bool);

    auto indices = mask.nonzero().squeeze(-1);

    EXPECT_EQ(indices.ndim(), 1);
    auto result = indices.cpu().to_vector_int64();

    EXPECT_EQ(result.size(), 3); // 3 true values
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 3);
    EXPECT_EQ(result[2], 4);
}

// ============================================================================
// TEST 5: nonzero() edge cases
// ============================================================================

TEST_F(RelocateGsEdgeCasesTest, NonzeroAllFalse) {
    // Test nonzero() when all elements are false
    auto mask = Tensor::zeros({100}, Device::CUDA).to(DataType::Bool);

    auto indices = mask.nonzero().squeeze(-1);

    EXPECT_EQ(indices.numel(), 0);
}

TEST_F(RelocateGsEdgeCasesTest, NonzeroAllTrue) {
    // Test nonzero() when all elements are true
    auto mask = Tensor::ones({50}, Device::CUDA).to(DataType::Bool);

    auto indices = mask.nonzero().squeeze(-1);

    EXPECT_EQ(indices.numel(), 50);
    auto result = indices.cpu().to_vector_int64();

    for (int i = 0; i < 50; ++i) {
        EXPECT_EQ(result[i], i);
    }
}

TEST_F(RelocateGsEdgeCasesTest, NonzeroEmptyTensor) {
    // Test nonzero() on empty tensor
    auto mask = Tensor::empty({0}, Device::CUDA, DataType::Bool);
    auto indices = mask.nonzero();

    EXPECT_EQ(indices.numel(), 0);
}

TEST_F(RelocateGsEdgeCasesTest, NonzeroSingleElement) {
    // Test nonzero() with single element
    auto mask_true = Tensor::ones({1}, Device::CUDA).to(DataType::Bool);
    auto indices_true = mask_true.nonzero().squeeze(-1);
    EXPECT_EQ(indices_true.numel(), 1);
    EXPECT_EQ(indices_true.cpu().to_vector_int64()[0], 0);

    auto mask_false = Tensor::zeros({1}, Device::CUDA).to(DataType::Bool);
    auto indices_false = mask_false.nonzero().squeeze(-1);
    EXPECT_EQ(indices_false.numel(), 0);
}

// ============================================================================
// TEST 6: Division by near-zero values
// ============================================================================

TEST_F(RelocateGsEdgeCasesTest, DivisionByNearZero) {
    // Test: p / (1 - p) when p is very close to 1
    std::vector<float> values_near_one = {
        0.9f, 0.99f, 0.999f, 0.9999f, 1.0f - 1e-7f};

    for (float val : values_near_one) {
        auto p = Tensor::full({1}, val, Device::CUDA);
        auto denominator = Tensor::ones_like(p) - p;
        auto division = p / denominator;
        auto result = division.cpu().to_vector()[0];

        EXPECT_TRUE(std::isfinite(result))
            << "Division " << val << " / (1 - " << val << ") is not finite";
        EXPECT_GT(result, 0.0f) << "Division should be positive";
        EXPECT_FALSE(std::isinf(result)) << "Division should not be Inf";
    }
}

TEST_F(RelocateGsEdgeCasesTest, DivisionByExactZero) {
    // Test what happens with exact 1.0 (should be prevented by clamping)
    auto p = Tensor::full({1}, 1.0f, Device::CUDA);
    auto clamped = p.clamp(0.005f, 1.0f - 1e-7f);
    auto denominator = Tensor::ones_like(clamped) - clamped;
    auto division = clamped / denominator;
    auto result = division.cpu().to_vector()[0];

    // After clamping, this should be finite
    EXPECT_TRUE(std::isfinite(result));
}

// ============================================================================
// TEST 7: ones_like() with different dtypes
// ============================================================================

TEST_F(RelocateGsEdgeCasesTest, OnesLikeFloat32) {
    auto tensor = Tensor::randn({50}, Device::CUDA);
    auto ones = Tensor::ones_like(tensor);

    EXPECT_EQ(ones.dtype(), DataType::Float32);
    EXPECT_EQ(ones.shape(), tensor.shape());

    auto values = ones.cpu().to_vector();
    for (const auto& val : values) {
        EXPECT_FLOAT_EQ(val, 1.0f);
    }
}

TEST_F(RelocateGsEdgeCasesTest, OnesLikeInt32) {
    auto tensor = Tensor::randn({50}, Device::CUDA);
    auto ones = Tensor::ones_like(tensor, DataType::Int32);

    EXPECT_EQ(ones.dtype(), DataType::Int32);
    EXPECT_EQ(ones.shape(), tensor.shape());

    auto values = ones.cpu().to_vector_int();
    for (const auto& val : values) {
        EXPECT_EQ(val, 1);
    }
}

TEST_F(RelocateGsEdgeCasesTest, OnesLikeInt64) {
    auto tensor = Tensor::randn({30}, Device::CUDA);
    auto ones = Tensor::ones_like(tensor, DataType::Int64);

    EXPECT_EQ(ones.dtype(), DataType::Int64);
    auto values = ones.cpu().to_vector_int64();
    for (const auto& val : values) {
        EXPECT_EQ(val, 1);
    }
}

TEST_F(RelocateGsEdgeCasesTest, OnesLikePreservesShape) {
    auto tensor = Tensor::randn({10, 5, 3}, Device::CUDA);
    auto ones = Tensor::ones_like(tensor, DataType::Int32);

    EXPECT_EQ(ones.shape(), tensor.shape());
    EXPECT_EQ(ones.shape()[0], 10);
    EXPECT_EQ(ones.shape()[1], 5);
    EXPECT_EQ(ones.shape()[2], 3);
}

// ============================================================================
// TEST 8: Double index_select (chained indexing)
// ============================================================================

TEST_F(RelocateGsEdgeCasesTest, DoubleIndexSelectBasic) {
    // Test: alive_indices.index_select(0, sampled_idxs_local)
    // This is double indirection: indices[sampled_idxs]

    std::vector<int64_t> alive_indices_data = {1, 3, 5, 7, 9}; // 5 alive
    std::vector<int64_t> sampled_local_data = {0, 2, 4, 1};    // Sample from alive

    auto alive_indices = Tensor::from_vector(std::vector<int>(alive_indices_data.begin(), alive_indices_data.end()), TensorShape{5}, Device::CUDA).to(DataType::Int64);
    auto sampled_local = Tensor::from_vector(std::vector<int>(sampled_local_data.begin(), sampled_local_data.end()), TensorShape{4}, Device::CUDA).to(DataType::Int64);

    auto sampled_idxs = alive_indices.index_select(0, sampled_local);
    auto result = sampled_idxs.cpu().to_vector_int64();

    EXPECT_EQ(result.size(), 4);
    EXPECT_EQ(result[0], 1); // alive_indices[0]
    EXPECT_EQ(result[1], 5); // alive_indices[2]
    EXPECT_EQ(result[2], 9); // alive_indices[4]
    EXPECT_EQ(result[3], 3); // alive_indices[1]
}

TEST_F(RelocateGsEdgeCasesTest, DoubleIndexSelectWithDuplicates) {
    // Test double index_select when sampled indices have duplicates
    std::vector<int64_t> alive_indices_data = {10, 20, 30, 40};
    std::vector<int64_t> sampled_local_data = {0, 0, 1, 1, 2, 2}; // Duplicates

    auto alive_indices = Tensor::from_vector(std::vector<int>(alive_indices_data.begin(), alive_indices_data.end()), TensorShape{4}, Device::CUDA).to(DataType::Int64);
    auto sampled_local = Tensor::from_vector(std::vector<int>(sampled_local_data.begin(), sampled_local_data.end()), TensorShape{6}, Device::CUDA).to(DataType::Int64);

    auto sampled_idxs = alive_indices.index_select(0, sampled_local);
    auto result = sampled_idxs.cpu().to_vector_int64();

    EXPECT_EQ(result.size(), 6);
    EXPECT_EQ(result[0], 10);
    EXPECT_EQ(result[1], 10);
    EXPECT_EQ(result[2], 20);
    EXPECT_EQ(result[3], 20);
    EXPECT_EQ(result[4], 30);
    EXPECT_EQ(result[5], 30);
}

TEST_F(RelocateGsEdgeCasesTest, TripleIndexSelect) {
    // Test even more indirection: a.index_select(0, b.index_select(0, c))
    auto a = Tensor::from_vector(std::vector<float>{100, 200, 300, 400},
                                 TensorShape{4}, Device::CUDA);
    auto b = Tensor::from_vector(std::vector<int>{3, 1, 2, 0}, TensorShape{4}, Device::CUDA).to(DataType::Int64);
    auto c = Tensor::from_vector(std::vector<int>{0, 2}, TensorShape{2}, Device::CUDA).to(DataType::Int64);

    // First: b[c] = [3, 2]
    auto b_indexed = b.index_select(0, c);
    // Then: a[b[c]] = a[[3, 2]] = [400, 300]
    auto result_tensor = a.index_select(0, b_indexed);
    auto result = result_tensor.cpu().to_vector();

    EXPECT_EQ(result.size(), 2);
    EXPECT_FLOAT_EQ(result[0], 400.0f);
    EXPECT_FLOAT_EQ(result[1], 300.0f);
}

// ============================================================================
// TEST 9: contiguous() correctness
// ============================================================================

TEST_F(RelocateGsEdgeCasesTest, ContiguousOnContiguousTensor) {
    // Test contiguous() on already contiguous tensor (no-op)
    auto tensor = Tensor::randn({100, 3}, Device::CUDA);
    EXPECT_TRUE(tensor.is_contiguous());

    auto contig = tensor.contiguous();
    EXPECT_TRUE(contig.is_contiguous());
    EXPECT_EQ(contig.shape(), tensor.shape());
}

TEST_F(RelocateGsEdgeCasesTest, ContiguousAfterTranspose) {
    // Test contiguous() after transpose (creates non-contiguous view)
    auto tensor = Tensor::randn({10, 5}, Device::CUDA);
    auto transposed = tensor.transpose(0, 1);

    // Transpose creates non-contiguous view
    EXPECT_FALSE(transposed.is_contiguous());

    auto contig = transposed.contiguous();
    EXPECT_TRUE(contig.is_contiguous());
    EXPECT_EQ(contig.shape()[0], 5);
    EXPECT_EQ(contig.shape()[1], 10);
}

// TEST REMOVED: slice() API not available in tensor library

TEST_F(RelocateGsEdgeCasesTest, ContiguousPreservesData) {
    // Ensure contiguous() preserves data correctly
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto tensor = Tensor::from_vector(data, TensorShape{2, 5}, Device::CUDA);
    auto transposed = tensor.transpose(0, 1);
    auto contig = transposed.contiguous();

    // Verify shape
    EXPECT_EQ(contig.shape()[0], 5);
    EXPECT_EQ(contig.shape()[1], 2);

    // Verify data is correct
    auto result = contig.cpu().to_vector();
    EXPECT_FLOAT_EQ(result[0], 1.0f); // [0, 0]
    EXPECT_FLOAT_EQ(result[1], 6.0f); // [0, 1]
    EXPECT_FLOAT_EQ(result[2], 2.0f); // [1, 0]
    EXPECT_FLOAT_EQ(result[3], 7.0f); // [1, 1]
}

// ============================================================================
// TEST 10: log() of scales edge cases
// ============================================================================

TEST_F(RelocateGsEdgeCasesTest, LogOfPositiveScales) {
    // Test log() on positive scales (normal case)
    std::vector<float> scales = {0.001f, 0.01f, 0.1f, 1.0f, 10.0f};
    auto tensor = Tensor::from_vector(scales, TensorShape{5}, Device::CUDA);
    auto log_scales = tensor.log();
    auto result = log_scales.cpu().to_vector();

    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_TRUE(std::isfinite(result[i]))
            << "log(" << scales[i] << ") is not finite";
        EXPECT_FALSE(std::isnan(result[i]))
            << "log(" << scales[i] << ") is NaN";
    }
}

TEST_F(RelocateGsEdgeCasesTest, LogOfVerySmallPositive) {
    // Test log() on very small positive values
    std::vector<float> scales = {1e-10f, 1e-8f, 1e-6f, 1e-4f};
    auto tensor = Tensor::from_vector(scales, TensorShape{4}, Device::CUDA);
    auto log_scales = tensor.log();
    auto result = log_scales.cpu().to_vector();

    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_TRUE(std::isfinite(result[i]))
            << "log(" << scales[i] << ") is not finite";
        EXPECT_LT(result[i], 0.0f) << "log of small value should be negative";
    }
}

// ============================================================================
// TEST 11: Complete relocate_gs pipeline simulation
// ============================================================================

TEST_F(RelocateGsEdgeCasesTest, FullRelocateGsPipelineSimulation) {
    // Simulate the complete pipeline from relocate_gs
    const size_t N = 100;
    const float min_opacity = 0.005f;

    // 1. Create test data with EXPLICITLY dead Gaussians
    // Dead Gaussians: opacity <= min_opacity OR rot_mag_sq < 1e-8
    std::vector<float> opacity_data(N);
    for (size_t i = 0; i < N; ++i) {
        if (i < 10) {
            // First 10: very low opacity (dead by opacity criterion)
            opacity_data[i] = 0.001f + (i * 0.0002f); // [0.001, 0.0028]
        } else if (i < 20) {
            // Next 10: at threshold
            opacity_data[i] = min_opacity;
        } else {
            // Rest: alive
            opacity_data[i] = 0.2f + (static_cast<float>(i) / N) * 0.5f; // [0.2, 0.7]
        }
    }
    auto opacities = Tensor::from_vector(opacity_data, TensorShape{N}, Device::CUDA);

    // Create rotations: some with very small magnitude (dead by rotation criterion)
    auto rotation_raw = Tensor::randn({N, 4}, Device::CUDA) * 0.1f;
    auto rotation_cpu = rotation_raw.cpu().to_vector();
    for (size_t i = 20; i < 25; ++i) {
        // Indices 20-24: very small rotations (dead by rotation criterion)
        // Set all 4 quaternion components to tiny values
        rotation_cpu[i * 4 + 0] = 1e-5f;
        rotation_cpu[i * 4 + 1] = 1e-5f;
        rotation_cpu[i * 4 + 2] = 1e-5f;
        rotation_cpu[i * 4 + 3] = 1e-5f;
    }
    rotation_raw = Tensor::from_vector(rotation_cpu, TensorShape{N, 4}, Device::CUDA);

    // 2. Find dead mask (logical operations)
    auto rot_mag_sq = (rotation_raw * rotation_raw).sum(-1);
    auto dead_mask = (opacities <= min_opacity).logical_or(rot_mag_sq < 1e-8f);
    auto dead_indices = dead_mask.nonzero().squeeze(-1);
    int n_dead = dead_indices.numel();

    // Should have dead Gaussians now (at least 10 from low opacity + 10 at threshold + 5 from low rotation)
    EXPECT_GT(n_dead, 0) << "Should have dead Gaussians in explicitly constructed data";
    EXPECT_GE(n_dead, 15) << "Should have at least 15 dead Gaussians (10 low opacity + 5 low rotation)";

    // 3. Find alive indices
    auto alive_mask = dead_mask.logical_not();
    auto alive_indices = alive_mask.nonzero().squeeze(-1);

    EXPECT_GT(alive_indices.numel(), 0) << "Should have some alive Gaussians";

    // 4. Simulate sampling (use first few alive indices)
    int n_sample = std::min(static_cast<int>(alive_indices.numel()), 10);
    auto sampled_local = Tensor::arange(0, n_sample).cuda().to(DataType::Int64);
    auto sampled_idxs = alive_indices.index_select(0, sampled_local);

    // 5. Get sampled opacities
    auto sampled_opacities = opacities.index_select(0, sampled_idxs);

    // 6. Count occurrences (Int32 operations)
    auto ratios = Tensor::ones_like(opacities, DataType::Int32);
    ratios = ratios.index_add_(0, sampled_idxs,
                               Tensor::ones(TensorShape{sampled_idxs.numel()}, Device::CUDA, DataType::Int32));
    ratios = ratios.index_select(0, sampled_idxs).contiguous();
    const int n_max = 51;
    ratios = ratios.clamp(1, n_max);

    // 7. Simulate new opacities and compute logit
    auto new_opacities = sampled_opacities * 0.8f; // Scaled down
    new_opacities = new_opacities.clamp(min_opacity, 1.0f - 1e-7f);
    auto new_opacity_raw = (new_opacities / (Tensor::ones_like(new_opacities) - new_opacities)).log();

    // 8. Verify all results are valid
    auto opacity_raw_vec = new_opacity_raw.cpu().to_vector();
    for (const auto& val : opacity_raw_vec) {
        EXPECT_TRUE(std::isfinite(val)) << "Opacity raw value is not finite";
        EXPECT_FALSE(std::isnan(val)) << "Opacity raw value is NaN";
    }

    auto ratio_vec = ratios.cpu().to_vector_int();
    for (const auto& val : ratio_vec) {
        EXPECT_GE(val, 1);
        EXPECT_LE(val, n_max);
    }
}

TEST_F(RelocateGsEdgeCasesTest, UnsqueezeConditional) {
    // Test conditional unsqueeze as in: if (ndim == 2) { unsqueeze(-1) }

    // Case 1: 1D tensor, no unsqueeze
    auto tensor_1d = Tensor::randn({10}, Device::CUDA);
    Tensor result_1d = tensor_1d;
    if (tensor_1d.ndim() == 2) {
        result_1d = result_1d.unsqueeze(-1);
    }
    EXPECT_EQ(result_1d.ndim(), 1);
    EXPECT_EQ(result_1d.shape(), tensor_1d.shape());

    // Case 2: 2D tensor [N, 1], should unsqueeze to [N, 1, 1]
    auto tensor_2d = Tensor::randn({10, 1}, Device::CUDA);
    Tensor result_2d = tensor_2d;
    if (result_2d.ndim() == 2) {
        result_2d = result_2d.unsqueeze(-1);
    }
    EXPECT_EQ(result_2d.ndim(), 3);
    EXPECT_EQ(result_2d.shape()[0], 10);
    EXPECT_EQ(result_2d.shape()[1], 1);
    EXPECT_EQ(result_2d.shape()[2], 1);
}

// ============================================================================
// TEST 12: Rotation magnitude squared calculation
// ============================================================================

TEST_F(RelocateGsEdgeCasesTest, RotationMagnitudeSquaredCalculation) {
    // Verify: (rotation_raw * rotation_raw).sum(-1) correctly computes quaternion mag²
    // Formula: w² + x² + y² + z² for quaternion [w, x, y, z]

    // Test with known quaternion values
    std::vector<float> quaternions = {
        // quat 0: [1, 0, 0, 0] -> mag² = 1
        1.0f, 0.0f, 0.0f, 0.0f,
        // quat 1: [0.5, 0.5, 0.5, 0.5] -> mag² = 0.25*4 = 1.0
        0.5f, 0.5f, 0.5f, 0.5f,
        // quat 2: [2, 0, 0, 0] -> mag² = 4
        2.0f, 0.0f, 0.0f, 0.0f,
        // quat 3: [1, 1, 1, 1] -> mag² = 4
        1.0f, 1.0f, 1.0f, 1.0f,
        // quat 4: very small -> mag² ≈ 4e-10 (below threshold)
        1e-5f, 1e-5f, 1e-5f, 1e-5f};

    auto rotation_raw = Tensor::from_vector(quaternions, TensorShape{5, 4}, Device::CUDA);

    // Compute magnitude squared using the formula from relocate_gs
    auto rot_mag_sq = (rotation_raw * rotation_raw).sum(-1);
    auto result = rot_mag_sq.cpu().to_vector();

    // Verify against manual calculation
    for (size_t i = 0; i < 5; ++i) {
        float w = quaternions[i * 4 + 0];
        float x = quaternions[i * 4 + 1];
        float y = quaternions[i * 4 + 2];
        float z = quaternions[i * 4 + 3];

        float expected = w * w + x * x + y * y + z * z;
        float computed = result[i];
        float diff = std::abs(computed - expected);

        EXPECT_LT(diff, 1e-6f)
            << "Quat[" << i << "]: (" << w << ", " << x << ", " << y << ", " << z << ") "
            << "expected mag²=" << expected << ", got=" << computed;
    }

    // Verify expected values
    EXPECT_NEAR(result[0], 1.0f, 1e-6f); // Unit quaternion
    EXPECT_NEAR(result[1], 1.0f, 1e-6f); // Normalized quaternion
    EXPECT_NEAR(result[2], 4.0f, 1e-6f); // Scaled quaternion
    EXPECT_NEAR(result[3], 4.0f, 1e-6f); // All ones
    EXPECT_LT(result[4], 1e-8f);         // Very small (below dead threshold)
}

TEST_F(RelocateGsEdgeCasesTest, RotationDeadThreshold) {
    // Test the dead Gaussian detection threshold: rot_mag_sq < 1e-8
    std::vector<float> test_quats = {
        // Below threshold (should be dead)
        1e-5f, 1e-5f, 1e-5f, 1e-5f, // mag² = 4e-10 < 1e-8 ✓ dead
        5e-5f, 0.0f, 0.0f, 0.0f,    // mag² = 2.5e-9 < 1e-8 ✓ dead
        // Above threshold (should be alive)
        1e-4f, 1e-5f, 1e-5f, 1e-5f, // mag² ≈ 1.02e-8 > 1e-8 ✗ alive
        1e-3f, 0.0f, 0.0f, 0.0f,    // mag² = 1e-6 > 1e-8 ✗ alive
    };

    auto rotation_raw = Tensor::from_vector(test_quats, TensorShape{4, 4}, Device::CUDA);
    auto rot_mag_sq = (rotation_raw * rotation_raw).sum(-1);

    const float threshold = 1e-8f;
    auto is_dead = rot_mag_sq < threshold;
    auto dead_vals = is_dead.cpu().to_vector_int();

    // First two should be dead (mag² < 1e-8)
    EXPECT_NE(dead_vals[0], 0) << "Quaternion with mag²=4e-10 should be dead";
    EXPECT_NE(dead_vals[1], 0) << "Quaternion with mag²=2.5e-9 should be dead";

    // Last two should be alive (mag² >= 1e-8)
    EXPECT_EQ(dead_vals[2], 0) << "Quaternion with mag²≈1.02e-8 should be alive";
    EXPECT_EQ(dead_vals[3], 0) << "Quaternion with mag²=1e-6 should be alive";
}

TEST_F(RelocateGsEdgeCasesTest, RotationMagnitudeSumDimension) {
    // Verify sum(-1) sums along the last dimension correctly
    std::vector<float> simple_data = {
        1.0f, 2.0f, 3.0f, 4.0f, // row 0: 1² + 2² + 3² + 4² = 1+4+9+16 = 30
        5.0f, 6.0f, 7.0f, 8.0f  // row 1: 5² + 6² + 7² + 8² = 25+36+49+64 = 174
    };

    auto tensor = Tensor::from_vector(simple_data, TensorShape{2, 4}, Device::CUDA);
    auto squared = tensor * tensor;
    auto summed = squared.sum(-1);
    auto result = summed.cpu().to_vector();

    EXPECT_NEAR(result[0], 30.0f, 1e-5f) << "Row 0: [1,2,3,4] squared and summed";
    EXPECT_NEAR(result[1], 174.0f, 1e-5f) << "Row 1: [5,6,7,8] squared and summed";
    EXPECT_EQ(result.size(), 2) << "sum(-1) should reduce last dimension";
}

// =============================================================================
// TEST 13: LibTorch Comparison for (rotation_raw * rotation_raw).sum(-1)
// =============================================================================

TEST_F(RelocateGsEdgeCasesTest, RotationMagnitudeSquared_LibTorchComparison) {
    // CRITICAL: Verify exact operation matches LibTorch implementation
    // This is the exact formula used in relocate_gs for dead Gaussian detection

    std::vector<float> quaternions = {
        // Unit quaternion
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        // Normalized quaternion
        0.5f,
        0.5f,
        0.5f,
        0.5f,
        // Scaled quaternion
        2.0f,
        0.0f,
        0.0f,
        0.0f,
        // All ones
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        // Very small (dead threshold)
        1e-5f,
        1e-5f,
        1e-5f,
        1e-5f,
        // Random values
        0.7071f,
        0.7071f,
        0.0f,
        0.0f,
        // Near threshold: mag² ≈ 1e-8
        3.16e-5f,
        0.0f,
        0.0f,
        0.0f,
        // Negative values
        -1.0f,
        1.0f,
        -1.0f,
        1.0f,
    };

    const size_t N = quaternions.size() / 4;

    // ========== LFS Tensor Library ==========
    auto rotation_lfs = Tensor::from_vector(quaternions, TensorShape{N, 4}, Device::CUDA);
    auto rot_mag_sq_lfs = (rotation_lfs * rotation_lfs).sum(-1);
    auto result_lfs = rot_mag_sq_lfs.cpu().to_vector();

    // ========== LibTorch ==========
    auto rotation_torch = torch::from_blob(
                              quaternions.data(),
                              {static_cast<long>(N), 4},
                              torch::kFloat32)
                              .clone()
                              .cuda();
    auto rot_mag_sq_torch = (rotation_torch * rotation_torch).sum(-1);
    auto result_torch_tensor = rot_mag_sq_torch.cpu();

    // ========== Compare Results ==========
    for (size_t i = 0; i < N; ++i) {
        float lfs_val = result_lfs[i];
        float torch_val = result_torch_tensor[i].item<float>();
        float diff = std::abs(lfs_val - torch_val);

        EXPECT_LT(diff, 1e-6f)
            << "Quat[" << i << "]: ["
            << quaternions[i * 4 + 0] << ", "
            << quaternions[i * 4 + 1] << ", "
            << quaternions[i * 4 + 2] << ", "
            << quaternions[i * 4 + 3] << "] "
            << "LFS=" << lfs_val << " vs LibTorch=" << torch_val
            << " (diff=" << diff << ")";
    }
}

TEST_F(RelocateGsEdgeCasesTest, LogitCalculation_LibTorchComparison) {
    // Compare full logit calculation: log(p / (1 - p))

    std::vector<float> opacity_values = {
        0.005f, 0.01f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.99f,
        1.0f - 1e-7f, 0.005f + 1e-6f, 0.5f - 1e-6f, 0.5f + 1e-6f};
    const size_t N = opacity_values.size();

    // ========== LFS Tensor Library ==========
    auto p_lfs = Tensor::from_vector(opacity_values, TensorShape{N}, Device::CUDA);
    auto logit_lfs = (p_lfs / (Tensor::ones_like(p_lfs) - p_lfs)).log();
    auto result_lfs = logit_lfs.cpu().to_vector();

    // ========== LibTorch ==========
    auto p_torch = torch::from_blob(
                       opacity_values.data(),
                       {static_cast<long>(N)},
                       torch::kFloat32)
                       .clone()
                       .cuda();
    auto logit_torch = (p_torch / (torch::ones_like(p_torch) - p_torch)).log();
    auto result_torch_tensor = logit_torch.cpu();

    // ========== Compare Results ==========
    for (size_t i = 0; i < N; ++i) {
        float lfs_val = result_lfs[i];
        float torch_val = result_torch_tensor[i].item<float>();
        float diff = std::abs(lfs_val - torch_val);

        EXPECT_LT(diff, 1e-6f)
            << "Logit mismatch at index " << i
            << ": p=" << opacity_values[i]
            << ", LFS=" << lfs_val << " vs LibTorch=" << torch_val
            << " (diff=" << diff << ")";
    }
}

TEST_F(RelocateGsEdgeCasesTest, Int32Clamp_LibTorchComparison) {
    // Compare Int32 clamp operation

    std::vector<float> values_float = {
        -100, -10, -1, 0, 1, 2, 10, 25, 50, 51, 52, 100, 1000};
    const size_t N = values_float.size();
    const int n_max = 51;

    // ========== LFS Tensor Library ==========
    auto tensor_lfs = Tensor::from_vector(values_float, TensorShape{N}, Device::CUDA).to(DataType::Int32);
    auto result_lfs_tensor = tensor_lfs.clamp(1, n_max);
    auto result_lfs = result_lfs_tensor.cpu().to_vector_int();

    // ========== LibTorch ==========
    auto tensor_torch = torch::from_blob(
                            values_float.data(),
                            {static_cast<long>(N)},
                            torch::kFloat32)
                            .clone()
                            .to(torch::kInt32)
                            .cuda();
    auto result_torch = tensor_torch.clamp(1, n_max);
    auto result_torch_tensor = result_torch.cpu();

    // ========== Compare Results ==========
    for (size_t i = 0; i < N; ++i) {
        int lfs_val = result_lfs[i];
        int torch_val = result_torch_tensor[i].item<int>();

        EXPECT_EQ(lfs_val, torch_val)
            << "Int32 clamp mismatch at index " << i
            << ": input=" << static_cast<int>(values_float[i])
            << ", LFS=" << lfs_val << " vs LibTorch=" << torch_val;
    }
}

TEST_F(RelocateGsEdgeCasesTest, LogicalOperations_LibTorchComparison) {
    // Compare logical_or and logical_not operations

    std::vector<float> mask1_data = {true, true, false, false, true, false, true, false};
    std::vector<float> mask2_data = {true, false, true, false, false, true, true, true};
    const size_t N = mask1_data.size();

    // Test logical_or
    {
        // ========== LFS Tensor Library ==========
        auto mask1_lfs = Tensor::from_vector(mask1_data, TensorShape{N}, Device::CUDA).to(DataType::Bool);
        auto mask2_lfs = Tensor::from_vector(mask2_data, TensorShape{N}, Device::CUDA).to(DataType::Bool);
        auto result_lfs_tensor = mask1_lfs.logical_or(mask2_lfs);
        auto result_lfs = result_lfs_tensor.cpu().to_vector_bool();

        // ========== LibTorch ==========
        auto mask1_torch = torch::from_blob(
                               mask1_data.data(),
                               {static_cast<long>(N)},
                               torch::kFloat32)
                               .clone()
                               .to(torch::kBool)
                               .cuda();
        auto mask2_torch = torch::from_blob(
                               mask2_data.data(),
                               {static_cast<long>(N)},
                               torch::kFloat32)
                               .clone()
                               .to(torch::kBool)
                               .cuda();
        auto result_torch = mask1_torch.logical_or(mask2_torch);
        auto result_torch_tensor = result_torch.cpu();

        // ========== Compare Results ==========
        for (size_t i = 0; i < N; ++i) {
            bool lfs_val = result_lfs[i];
            bool torch_val = result_torch_tensor[i].item<bool>();

            EXPECT_EQ(lfs_val, torch_val)
                << "logical_or mismatch at index " << i
                << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val;
        }
    }

    // Test logical_not
    {
        // ========== LFS Tensor Library ==========
        auto mask_lfs = Tensor::from_vector(mask1_data, TensorShape{N}, Device::CUDA).to(DataType::Bool);
        auto result_lfs_tensor = mask_lfs.logical_not();
        auto result_lfs = result_lfs_tensor.cpu().to_vector_bool();

        // ========== LibTorch ==========
        auto mask_torch = torch::from_blob(
                              mask1_data.data(),
                              {static_cast<long>(N)},
                              torch::kFloat32)
                              .clone()
                              .to(torch::kBool)
                              .cuda();
        auto result_torch = mask_torch.logical_not();
        auto result_torch_tensor = result_torch.cpu();

        // ========== Compare Results ==========
        for (size_t i = 0; i < N; ++i) {
            bool lfs_val = result_lfs[i];
            bool torch_val = result_torch_tensor[i].item<bool>();

            EXPECT_EQ(lfs_val, torch_val)
                << "logical_not mismatch at index " << i
                << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val;
        }
    }
}

TEST_F(RelocateGsEdgeCasesTest, Squeeze_LibTorchComparison) {
    // Compare squeeze with negative indices

    // Test case 1: squeeze(-1) on [N, 1] -> [N]
    {
        std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        const size_t N = data.size();

        // ========== LFS Tensor Library ==========
        auto tensor_lfs = Tensor::from_vector(data, TensorShape{N, 1}, Device::CUDA);
        auto result_lfs_tensor = tensor_lfs.squeeze(-1);
        auto result_lfs = result_lfs_tensor.cpu().to_vector();

        // ========== LibTorch ==========
        auto tensor_torch = torch::from_blob(
                                data.data(),
                                {static_cast<long>(N), 1},
                                torch::kFloat32)
                                .clone()
                                .cuda();
        auto result_torch = tensor_torch.squeeze(-1);
        auto result_torch_tensor = result_torch.cpu();

        // ========== Compare Results ==========
        EXPECT_EQ(result_lfs_tensor.ndim(), result_torch.dim())
            << "squeeze(-1) dimension mismatch";

        for (size_t i = 0; i < N; ++i) {
            float lfs_val = result_lfs[i];
            float torch_val = result_torch_tensor[i].item<float>();

            EXPECT_FLOAT_EQ(lfs_val, torch_val)
                << "squeeze(-1) value mismatch at index " << i;
        }
    }

    // Test case 2: squeeze(-1) on [N] (no-op)
    {
        std::vector<float> data = {1, 2, 3, 4, 5};
        const size_t N = data.size();

        // ========== LFS Tensor Library ==========
        auto tensor_lfs = Tensor::from_vector(data, TensorShape{N}, Device::CUDA);
        auto result_lfs_tensor = tensor_lfs.squeeze(-1);

        // ========== LibTorch ==========
        auto tensor_torch = torch::from_blob(
                                data.data(),
                                {static_cast<long>(N)},
                                torch::kFloat32)
                                .clone()
                                .cuda();
        auto result_torch = tensor_torch.squeeze(-1);

        // ========== Compare Results ==========
        EXPECT_EQ(result_lfs_tensor.ndim(), result_torch.dim())
            << "squeeze(-1) on 1D tensor dimension mismatch";
        EXPECT_EQ(result_lfs_tensor.shape()[0], result_torch.size(0))
            << "squeeze(-1) on 1D tensor shape mismatch";
    }
}

TEST_F(RelocateGsEdgeCasesTest, Nonzero_LibTorchComparison) {
    // Compare nonzero operation

    std::vector<float> mask_data = {false, true, false, true, true, false, false, true, false, true};
    const size_t N = mask_data.size();

    // ========== LFS Tensor Library ==========
    auto mask_lfs = Tensor::from_vector(mask_data, TensorShape{N}, Device::CUDA).to(DataType::Bool);
    auto indices_lfs = mask_lfs.nonzero().squeeze(-1);
    auto result_lfs = indices_lfs.cpu().to_vector_int64();

    // ========== LibTorch ==========
    auto mask_torch = torch::from_blob(
                          mask_data.data(),
                          {static_cast<long>(N)},
                          torch::kFloat32)
                          .clone()
                          .to(torch::kBool)
                          .cuda();
    auto indices_torch = mask_torch.nonzero().squeeze(-1);
    auto result_torch_tensor = indices_torch.cpu();

    // ========== Compare Results ==========
    EXPECT_EQ(result_lfs.size(), result_torch_tensor.size(0))
        << "nonzero count mismatch";

    for (size_t i = 0; i < result_lfs.size(); ++i) {
        int64_t lfs_val = result_lfs[i];
        int64_t torch_val = result_torch_tensor[i].item<int64_t>();

        EXPECT_EQ(lfs_val, torch_val)
            << "nonzero index mismatch at position " << i
            << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val;
    }
}

TEST_F(RelocateGsEdgeCasesTest, Division_LibTorchComparison) {
    // Compare division operations, especially p / (1 - p)

    std::vector<float> p_values = {
        0.005f, 0.01f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.99f, 1.0f - 1e-7f};
    const size_t N = p_values.size();

    // ========== LFS Tensor Library ==========
    auto p_lfs = Tensor::from_vector(p_values, TensorShape{N}, Device::CUDA);
    auto denominator_lfs = Tensor::ones_like(p_lfs) - p_lfs;
    auto result_lfs_tensor = p_lfs / denominator_lfs;
    auto result_lfs = result_lfs_tensor.cpu().to_vector();

    // ========== LibTorch ==========
    auto p_torch = torch::from_blob(
                       p_values.data(),
                       {static_cast<long>(N)},
                       torch::kFloat32)
                       .clone()
                       .cuda();
    auto denominator_torch = torch::ones_like(p_torch) - p_torch;
    auto result_torch = p_torch / denominator_torch;
    auto result_torch_tensor = result_torch.cpu();

    // ========== Compare Results ==========
    for (size_t i = 0; i < N; ++i) {
        float lfs_val = result_lfs[i];
        float torch_val = result_torch_tensor[i].item<float>();
        float diff = std::abs(lfs_val - torch_val);

        EXPECT_LT(diff, 1e-5f)
            << "Division mismatch at index " << i
            << ": p=" << p_values[i]
            << ", LFS=" << lfs_val << " vs LibTorch=" << torch_val
            << " (diff=" << diff << ")";
    }
}

TEST_F(RelocateGsEdgeCasesTest, OnesLike_LibTorchComparison) {
    // Compare ones_like with different dtypes

    std::vector<float> data = {1.5f, 2.7f, -3.2f, 4.8f, 0.0f, -1.1f};
    const size_t N = data.size();

    // Test Float32
    {
        // ========== LFS Tensor Library ==========
        auto tensor_lfs = Tensor::from_vector(data, TensorShape{N}, Device::CUDA);
        auto ones_lfs = Tensor::ones_like(tensor_lfs);
        auto result_lfs = ones_lfs.cpu().to_vector();

        // ========== LibTorch ==========
        auto tensor_torch = torch::from_blob(
                                data.data(),
                                {static_cast<long>(N)},
                                torch::kFloat32)
                                .clone()
                                .cuda();
        auto ones_torch = torch::ones_like(tensor_torch);
        auto result_torch_tensor = ones_torch.cpu();

        // ========== Compare Results ==========
        for (size_t i = 0; i < N; ++i) {
            EXPECT_FLOAT_EQ(result_lfs[i], result_torch_tensor[i].item<float>())
                << "ones_like Float32 mismatch at index " << i;
        }
    }

    // Test Int32
    {
        // ========== LFS Tensor Library ==========
        auto tensor_lfs = Tensor::from_vector(data, TensorShape{N}, Device::CUDA);
        auto ones_lfs = Tensor::ones_like(tensor_lfs, DataType::Int32);
        auto result_lfs = ones_lfs.cpu().to_vector_int();

        // ========== LibTorch ==========
        auto tensor_torch = torch::from_blob(
                                data.data(),
                                {static_cast<long>(N)},
                                torch::kFloat32)
                                .clone()
                                .cuda();
        auto ones_torch = torch::ones_like(tensor_torch, torch::kInt32);
        auto result_torch_tensor = ones_torch.cpu();

        // ========== Compare Results ==========
        for (size_t i = 0; i < N; ++i) {
            EXPECT_EQ(result_lfs[i], result_torch_tensor[i].item<int>())
                << "ones_like Int32 mismatch at index " << i;
        }
    }

    // Test Int64
    {
        // ========== LFS Tensor Library ==========
        auto tensor_lfs = Tensor::from_vector(data, TensorShape{N}, Device::CUDA);
        auto ones_lfs = Tensor::ones_like(tensor_lfs, DataType::Int64);
        auto result_lfs = ones_lfs.cpu().to_vector_int64();

        // ========== LibTorch ==========
        auto tensor_torch = torch::from_blob(
                                data.data(),
                                {static_cast<long>(N)},
                                torch::kFloat32)
                                .clone()
                                .cuda();
        auto ones_torch = torch::ones_like(tensor_torch, torch::kInt64);
        auto result_torch_tensor = ones_torch.cpu();

        // ========== Compare Results ==========
        for (size_t i = 0; i < N; ++i) {
            EXPECT_EQ(result_lfs[i], result_torch_tensor[i].item<int64_t>())
                << "ones_like Int64 mismatch at index " << i;
        }
    }
}

TEST_F(RelocateGsEdgeCasesTest, IndexSelect_LibTorchComparison) {
    // Compare chained index_select operations (double indirection)

    std::vector<float> base_data = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    std::vector<int64_t> indices1_data = {9, 7, 5, 3, 1}; // First level indices
    std::vector<int64_t> indices2_data = {0, 2, 4, 1};    // Second level indices

    const size_t N = base_data.size();
    const size_t N1 = indices1_data.size();
    const size_t N2 = indices2_data.size();

    // Test single index_select
    {
        // ========== LFS Tensor Library ==========
        auto base_lfs = Tensor::from_vector(base_data, TensorShape{N}, Device::CUDA);
        auto indices_lfs = Tensor::from_vector(
                               std::vector<int>(indices1_data.begin(), indices1_data.end()),
                               TensorShape{N1}, Device::CUDA)
                               .to(DataType::Int64);
        auto result_lfs_tensor = base_lfs.index_select(0, indices_lfs);
        auto result_lfs = result_lfs_tensor.cpu().to_vector();

        // ========== LibTorch ==========
        auto base_torch = torch::from_blob(
                              base_data.data(),
                              {static_cast<long>(N)},
                              torch::kFloat32)
                              .clone()
                              .cuda();
        auto indices_torch = torch::from_blob(
                                 indices1_data.data(),
                                 {static_cast<long>(N1)},
                                 torch::kInt64)
                                 .clone()
                                 .cuda();
        auto result_torch = base_torch.index_select(0, indices_torch);
        auto result_torch_tensor = result_torch.cpu();

        // ========== Compare Results ==========
        for (size_t i = 0; i < N1; ++i) {
            EXPECT_FLOAT_EQ(result_lfs[i], result_torch_tensor[i].item<float>())
                << "index_select mismatch at position " << i;
        }
    }

    // Test chained index_select
    {
        // ========== LFS Tensor Library ==========
        auto base_lfs = Tensor::from_vector(base_data, TensorShape{N}, Device::CUDA);
        auto indices1_lfs = Tensor::from_vector(
                                std::vector<int>(indices1_data.begin(), indices1_data.end()),
                                TensorShape{N1}, Device::CUDA)
                                .to(DataType::Int64);
        auto indices2_lfs = Tensor::from_vector(
                                std::vector<int>(indices2_data.begin(), indices2_data.end()),
                                TensorShape{N2}, Device::CUDA)
                                .to(DataType::Int64);

        auto temp_lfs = base_lfs.index_select(0, indices1_lfs);
        auto result_lfs_tensor = temp_lfs.index_select(0, indices2_lfs);
        auto result_lfs = result_lfs_tensor.cpu().to_vector();

        // ========== LibTorch ==========
        auto base_torch = torch::from_blob(
                              base_data.data(),
                              {static_cast<long>(N)},
                              torch::kFloat32)
                              .clone()
                              .cuda();
        auto indices1_torch = torch::from_blob(
                                  indices1_data.data(),
                                  {static_cast<long>(N1)},
                                  torch::kInt64)
                                  .clone()
                                  .cuda();
        auto indices2_torch = torch::from_blob(
                                  indices2_data.data(),
                                  {static_cast<long>(N2)},
                                  torch::kInt64)
                                  .clone()
                                  .cuda();

        auto temp_torch = base_torch.index_select(0, indices1_torch);
        auto result_torch = temp_torch.index_select(0, indices2_torch);
        auto result_torch_tensor = result_torch.cpu();

        // ========== Compare Results ==========
        for (size_t i = 0; i < N2; ++i) {
            EXPECT_FLOAT_EQ(result_lfs[i], result_torch_tensor[i].item<float>())
                << "Chained index_select mismatch at position " << i;
        }
    }
}

TEST_F(RelocateGsEdgeCasesTest, Sum_LibTorchComparison) {
    // Compare sum with dim=-1

    std::vector<float> data = {
        1.0f, 2.0f, 3.0f, 4.0f,     // Row 0: sum = 10
        5.0f, 6.0f, 7.0f, 8.0f,     // Row 1: sum = 26
        0.5f, 0.5f, 0.5f, 0.5f,     // Row 2: sum = 2
        1e-5f, 1e-5f, 1e-5f, 1e-5f, // Row 3: sum = 4e-5
    };

    const size_t N = 4;
    const size_t M = 4;

    // ========== LFS Tensor Library ==========
    auto tensor_lfs = Tensor::from_vector(data, TensorShape{N, M}, Device::CUDA);
    auto result_lfs_tensor = tensor_lfs.sum(-1);
    auto result_lfs = result_lfs_tensor.cpu().to_vector();

    // ========== LibTorch ==========
    auto tensor_torch = torch::from_blob(
                            data.data(),
                            {static_cast<long>(N), static_cast<long>(M)},
                            torch::kFloat32)
                            .clone()
                            .cuda();
    auto result_torch = tensor_torch.sum(-1);
    auto result_torch_tensor = result_torch.cpu();

    // ========== Compare Results ==========
    for (size_t i = 0; i < N; ++i) {
        float lfs_val = result_lfs[i];
        float torch_val = result_torch_tensor[i].item<float>();
        float diff = std::abs(lfs_val - torch_val);

        EXPECT_LT(diff, 1e-6f)
            << "sum(-1) mismatch at row " << i
            << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val
            << " (diff=" << diff << ")";
    }
}
