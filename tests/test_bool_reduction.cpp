/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <gtest/gtest.h>

using namespace lfs::core;

TEST(BoolReduction, SumScalarAllTrue) {
    const auto t = Tensor::full_bool({10000}, true, Device::CUDA);
    ASSERT_FLOAT_EQ(t.sum_scalar(), 10000.0f);
}

TEST(BoolReduction, SumScalarAllFalse) {
    const auto t = Tensor::full_bool({10000}, false, Device::CUDA);
    ASSERT_FLOAT_EQ(t.sum_scalar(), 0.0f);
}

TEST(BoolReduction, SumScalarMixed) {
    auto t = Tensor::full_bool({100, 100}, true, Device::CUDA);
    t.slice(0, 0, 50).fill_(false);
    ASSERT_FLOAT_EQ(t.sum_scalar(), 5000.0f);
}

TEST(BoolReduction, LargeSum) {
    const auto t = Tensor::full_bool({10000000}, true, Device::CUDA);
    ASSERT_FLOAT_EQ(t.sum_scalar(), 10000000.0f);
}

TEST(BoolReduction, MeanOperation) {
    auto t = Tensor::full_bool({1000}, true, Device::CUDA);
    t.slice(0, 0, 500).fill_(false);
    // Mean of bool returns Int64, integer division: 500/1000 = 0
    ASSERT_EQ(t.mean().item<int64_t>(), 0);
}

TEST(BoolReduction, MaxOperation) {
    const auto t_all_false = Tensor::full_bool({1000}, false, Device::CUDA);
    ASSERT_EQ(t_all_false.max().item<int64_t>(), 0);

    auto t_some_true = Tensor::full_bool({1000}, false, Device::CUDA);
    t_some_true.slice(0, 500, 501).fill_(true);
    ASSERT_EQ(t_some_true.max().item<int64_t>(), 1);
}

TEST(BoolReduction, MinOperation) {
    const auto t_all_true = Tensor::full_bool({1000}, true, Device::CUDA);
    ASSERT_EQ(t_all_true.min().item<int64_t>(), 1);

    auto t_some_false = Tensor::full_bool({1000}, true, Device::CUDA);
    t_some_false.slice(0, 500, 501).fill_(false);
    ASSERT_EQ(t_some_false.min().item<int64_t>(), 0);
}

TEST(BoolReduction, ComparisonResult) {
    const auto t = Tensor::arange(0, 100, 1);
    const auto mask = t > 50.0f;
    ASSERT_EQ(mask.dtype(), DataType::Bool);
    ASSERT_FLOAT_EQ(mask.sum_scalar(), 49.0f); // Values 51-99
}

TEST(BoolReduction, NoDuplicatesBugFix) {
    const auto zeros = Tensor::zeros({1000000}, Device::CUDA, DataType::Bool);
    ASSERT_FLOAT_EQ(zeros.sum_scalar(), 0.0f);
}

// Axis-specific reduction tests for all(dim) / any(dim)

TEST(BoolReduction, AllDim1_AllTrue) {
    const auto t = Tensor::full_bool({100, 3}, true, Device::CUDA);
    const std::vector<int> axes = {1};
    const auto result = t.all(std::span<const int>(axes), false);

    ASSERT_EQ(result.ndim(), 1);
    ASSERT_EQ(result.shape()[0], 100);
    ASSERT_EQ(result.dtype(), DataType::Bool);
    EXPECT_EQ(result.sum_scalar(), 100.0f);
}

TEST(BoolReduction, AllDim1_AllFalse) {
    const auto t = Tensor::full_bool({100, 3}, false, Device::CUDA);
    const std::vector<int> axes = {1};
    const auto result = t.all(std::span<const int>(axes), false);

    ASSERT_EQ(result.ndim(), 1);
    ASSERT_EQ(result.shape()[0], 100);
    EXPECT_EQ(result.sum_scalar(), 0.0f);
}

TEST(BoolReduction, AllDim1_MixedRows) {
    // First 30 rows: all True, next 40: partial True, last 30: all False
    auto t = Tensor::full_bool({100, 3}, false, Device::CUDA);
    t.slice(0, 0, 30).fill_(true);
    t.slice(0, 30, 70).slice(1, 0, 1).fill_(true); // Only first column

    const std::vector<int> axes = {1};
    const auto result = t.all(std::span<const int>(axes), false);

    ASSERT_EQ(result.ndim(), 1);
    ASSERT_EQ(result.shape()[0], 100);
    EXPECT_EQ(result.sum_scalar(), 30.0f); // Only first 30 rows all True
}

TEST(BoolReduction, AllDim1_CropBoxSimulation) {
    // Simulates crop_by_cropbox: inside_both.all(dim=1)
    constexpr int N = 1000000;
    auto inside_both = Tensor::full_bool({N, 3}, false, Device::CUDA);
    inside_both.slice(0, 0, 300000).fill_(true);

    EXPECT_EQ(inside_both.sum_scalar(), 300000.0f * 3);

    const std::vector<int> axes = {1};
    const auto inside_mask = inside_both.all(std::span<const int>(axes), false);

    ASSERT_EQ(inside_mask.ndim(), 1);
    ASSERT_EQ(inside_mask.shape()[0], N);
    EXPECT_EQ(inside_mask.sum_scalar(), 300000.0f);
}

TEST(BoolReduction, AllDim1_SingleFalseBreaksRow) {
    auto t = Tensor::full_bool({10, 5}, true, Device::CUDA);
    t.slice(0, 5, 10).slice(1, 2, 3).fill_(false);

    const std::vector<int> axes = {1};
    const auto result = t.all(std::span<const int>(axes), false);

    EXPECT_EQ(result.sum_scalar(), 5.0f); // First 5 rows all True
}

TEST(BoolReduction, AnyDim1_Basic) {
    auto t = Tensor::full_bool({100, 3}, false, Device::CUDA);
    t.slice(0, 0, 30).slice(1, 0, 1).fill_(true);

    const std::vector<int> axes = {1};
    const auto result = t.any(std::span<const int>(axes), false);

    EXPECT_EQ(result.sum_scalar(), 30.0f); // First 30 rows have at least one True
}

TEST(BoolReduction, AllDim0_Basic) {
    auto t = Tensor::full_bool({100, 3}, true, Device::CUDA);
    t.slice(0, 50, 51).slice(1, 0, 1).fill_(false);

    const std::vector<int> axes = {0};
    const auto result = t.all(std::span<const int>(axes), false);

    ASSERT_EQ(result.ndim(), 1);
    ASSERT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.sum_scalar(), 2.0f); // Columns 1,2 all True, column 0 has False
}
