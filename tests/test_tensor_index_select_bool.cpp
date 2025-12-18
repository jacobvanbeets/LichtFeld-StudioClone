/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace lfs::core;

namespace {

    void compare_tensors(const Tensor& lfs, const torch::Tensor& ref, const std::string& ctx = "") {
        const auto torch_cpu = ref.cpu().contiguous();
        const auto lfs_cpu = lfs.cpu();

        ASSERT_EQ(lfs_cpu.ndim(), static_cast<size_t>(torch_cpu.dim())) << ctx;
        for (size_t i = 0; i < lfs_cpu.ndim(); ++i) {
            ASSERT_EQ(lfs_cpu.size(i), static_cast<size_t>(torch_cpu.size(i))) << ctx;
        }
        if (lfs_cpu.numel() == 0)
            return;

        const auto* lfs_ptr = lfs_cpu.ptr<float>();
        const auto torch_flat = torch_cpu.flatten();
        const auto acc = torch_flat.accessor<float, 1>();
        for (size_t i = 0; i < lfs_cpu.numel(); ++i) {
            EXPECT_FLOAT_EQ(lfs_ptr[i], acc[i]) << ctx << " at " << i;
        }
    }

    torch::Tensor to_torch_bool(const std::vector<bool>& v) {
        auto t = torch::zeros({static_cast<int64_t>(v.size())}, torch::kUInt8);
        auto* p = t.data_ptr<uint8_t>();
        for (size_t i = 0; i < v.size(); ++i)
            p[i] = v[i] ? 1 : 0;
        return t.to(torch::kBool);
    }

    Tensor make_bool_mask(const std::vector<bool>& v, Device dev = Device::CUDA) {
        auto m = Tensor::zeros({v.size()}, Device::CPU, DataType::Bool);
        auto* p = m.ptr<bool>();
        for (size_t i = 0; i < v.size(); ++i)
            p[i] = v[i];
        return dev == Device::CUDA ? m.cuda() : m;
    }

} // namespace

class TensorIndexSelectBoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(Tensor::zeros({1}, Device::CUDA).is_valid());
    }
};

// Basic functionality tests

TEST_F(TensorIndexSelectBoolTest, Basic_1D) {
    const auto data = Tensor::arange(0.0f, 5.0f);
    const auto mask = make_bool_mask({true, false, true, false, true});
    const auto result = data.index_select(0, mask);

    ASSERT_EQ(result.size(0), 3);
    const auto cpu = result.cpu();
    const auto* p = cpu.ptr<float>();
    EXPECT_FLOAT_EQ(p[0], 0.0f);
    EXPECT_FLOAT_EQ(p[1], 2.0f);
    EXPECT_FLOAT_EQ(p[2], 4.0f);
}

TEST_F(TensorIndexSelectBoolTest, Basic_2D) {
    const auto data = Tensor::arange(0.0f, 12.0f).reshape({4, 3});
    const auto mask = make_bool_mask({true, false, true, false});
    const auto result = data.index_select(0, mask);

    ASSERT_EQ(result.size(0), 2);
    ASSERT_EQ(result.size(1), 3);

    const auto cpu = result.cpu();
    const auto* p = cpu.ptr<float>();
    EXPECT_FLOAT_EQ(p[0], 0.0f);
    EXPECT_FLOAT_EQ(p[1], 1.0f);
    EXPECT_FLOAT_EQ(p[2], 2.0f);
    EXPECT_FLOAT_EQ(p[3], 6.0f);
    EXPECT_FLOAT_EQ(p[4], 7.0f);
    EXPECT_FLOAT_EQ(p[5], 8.0f);
}

TEST_F(TensorIndexSelectBoolTest, AllTrue) {
    const auto data = Tensor::arange(0.0f, 5.0f);
    const auto mask = Tensor::ones({5}, Device::CUDA, DataType::Bool);
    ASSERT_EQ(data.index_select(0, mask).size(0), 5);
}

TEST_F(TensorIndexSelectBoolTest, AllFalse) {
    const auto data = Tensor::arange(0.0f, 5.0f);
    const auto mask = Tensor::zeros({5}, Device::CUDA, DataType::Bool);
    ASSERT_EQ(data.index_select(0, mask).size(0), 0);
}

TEST_F(TensorIndexSelectBoolTest, LogicalNot_ApplyDeletedPattern) {
    const auto data = Tensor::arange(0.0f, 10.0f);
    auto deleted = Tensor::zeros({10}, Device::CPU, DataType::Bool);
    auto* p = deleted.ptr<bool>();
    p[2] = p[5] = p[7] = true;

    const auto keep = deleted.cuda().logical_not();
    const auto result = data.index_select(0, keep);

    ASSERT_EQ(result.size(0), 7);
    const auto cpu = result.cpu();
    const auto* r = cpu.ptr<float>();
    const std::vector<float> expected = {0, 1, 3, 4, 6, 8, 9};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(r[i], expected[i]);
    }
}

TEST_F(TensorIndexSelectBoolTest, LargeScale) {
    constexpr size_t N = 100000;
    const auto data = Tensor::randn({N, 3}, Device::CUDA);

    auto deleted = Tensor::zeros({N}, Device::CPU, DataType::Bool);
    auto* p = deleted.ptr<bool>();
    size_t num_del = 0;
    for (size_t i = 0; i < N; i += 10) {
        p[i] = true;
        ++num_del;
    }

    const auto keep = deleted.cuda().logical_not();
    const auto result = data.index_select(0, keep);

    ASSERT_EQ(result.size(0), N - num_del);
    ASSERT_EQ(result.size(1), 3);
}

// LibTorch comparison tests

TEST_F(TensorIndexSelectBoolTest, VsTorch_1D) {
    const auto t_data = torch::arange(10, torch::kFloat32).cuda();
    const auto lfs_data = Tensor::arange(0.0f, 10.0f);

    const std::vector<bool> mv = {true, false, true, false, true, false, true, false, true, false};
    const auto t_mask = to_torch_bool(mv).cuda();
    const auto lfs_mask = make_bool_mask(mv);

    const auto t_idx = torch::nonzero(t_mask).flatten().to(torch::kLong);
    compare_tensors(lfs_data.index_select(0, lfs_mask), torch::index_select(t_data, 0, t_idx), "1D");
}

TEST_F(TensorIndexSelectBoolTest, VsTorch_2D) {
    const auto t_data = torch::arange(12, torch::kFloat32).reshape({4, 3}).cuda();
    const auto lfs_data = Tensor::arange(0.0f, 12.0f).reshape({4, 3});

    const std::vector<bool> mv = {false, true, false, true};
    const auto t_mask = to_torch_bool(mv).cuda();
    const auto lfs_mask = make_bool_mask(mv);

    const auto t_idx = torch::nonzero(t_mask).flatten().to(torch::kLong);
    compare_tensors(lfs_data.index_select(0, lfs_mask), torch::index_select(t_data, 0, t_idx), "2D");
}

TEST_F(TensorIndexSelectBoolTest, VsTorch_AllTrue) {
    const auto t_data = torch::arange(5, torch::kFloat32).cuda();
    const auto lfs_data = Tensor::arange(0.0f, 5.0f);

    const auto t_mask = torch::ones({5}, torch::kBool).cuda();
    const auto lfs_mask = Tensor::ones({5}, Device::CUDA, DataType::Bool);

    const auto t_idx = torch::nonzero(t_mask).flatten().to(torch::kLong);
    compare_tensors(lfs_data.index_select(0, lfs_mask), torch::index_select(t_data, 0, t_idx), "all true");
}

TEST_F(TensorIndexSelectBoolTest, VsTorch_AllFalse) {
    const auto t_data = torch::arange(5, torch::kFloat32).cuda();
    const auto lfs_data = Tensor::arange(0.0f, 5.0f);

    const auto t_mask = torch::zeros({5}, torch::kBool).cuda();
    const auto lfs_mask = Tensor::zeros({5}, Device::CUDA, DataType::Bool);

    const auto t_idx = torch::nonzero(t_mask).flatten().to(torch::kLong);
    ASSERT_EQ(lfs_data.index_select(0, lfs_mask).size(0), 0);
    ASSERT_EQ(torch::index_select(t_data, 0, t_idx).size(0), 0);
}

TEST_F(TensorIndexSelectBoolTest, VsTorch_LogicalNot) {
    const auto t_data = torch::arange(10, torch::kFloat32).cuda();
    const auto lfs_data = Tensor::arange(0.0f, 10.0f);

    std::vector<bool> dv(10, false);
    dv[2] = dv[5] = dv[7] = true;

    const auto t_del = to_torch_bool(dv).cuda();
    const auto t_keep = torch::logical_not(t_del);

    auto lfs_del = Tensor::zeros({10}, Device::CPU, DataType::Bool);
    auto* p = lfs_del.ptr<bool>();
    for (size_t i = 0; i < 10; ++i)
        p[i] = dv[i];
    const auto lfs_keep = lfs_del.cuda().logical_not();

    const auto t_idx = torch::nonzero(t_keep).flatten().to(torch::kLong);
    compare_tensors(lfs_data.index_select(0, lfs_keep), torch::index_select(t_data, 0, t_idx), "logical_not");
}

TEST_F(TensorIndexSelectBoolTest, VsTorch_3D) {
    const auto t_data = torch::arange(24, torch::kFloat32).reshape({4, 3, 2}).cuda();
    const auto lfs_data = Tensor::arange(0.0f, 24.0f).reshape({4, 3, 2});

    const std::vector<bool> mv = {true, false, true, false};
    const auto t_mask = to_torch_bool(mv).cuda();
    const auto lfs_mask = make_bool_mask(mv);

    const auto t_idx = torch::nonzero(t_mask).flatten().to(torch::kLong);
    compare_tensors(lfs_data.index_select(0, lfs_mask), torch::index_select(t_data, 0, t_idx), "3D");
}

TEST_F(TensorIndexSelectBoolTest, VsTorch_LargeRandom) {
    constexpr int64_t N = 10000, M = 16;
    const auto t_data = torch::randn({N, M}, torch::kFloat32).cuda();

    const auto t_cpu = t_data.cpu().contiguous();
    auto lfs_data = Tensor::zeros({static_cast<size_t>(N), static_cast<size_t>(M)}, Device::CUDA);
    cudaMemcpy(lfs_data.data_ptr(), t_cpu.data_ptr<float>(), N * M * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<bool> mv(N);
    std::srand(42);
    for (int64_t i = 0; i < N; ++i)
        mv[i] = (std::rand() % 100) < 30;

    const auto t_mask = to_torch_bool(mv).cuda();
    const auto lfs_mask = make_bool_mask(mv);

    const auto t_idx = torch::nonzero(t_mask).flatten().to(torch::kLong);
    compare_tensors(lfs_data.index_select(0, lfs_mask), torch::index_select(t_data, 0, t_idx), "large random");
}
