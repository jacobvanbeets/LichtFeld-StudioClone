/* Test to verify tensor cat operation works correctly */

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>

using namespace lfs::core;

TEST(TensorCatBugTest, SimpleConcatenation) {
    // Create first tensor [2, 3] with values 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
    auto t1 = Tensor::zeros({2, 3}, Device::CUDA);
    {
        float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        cudaMemcpy(t1.ptr<float>(), data, 6 * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Create second tensor [1, 3] with values 7.0, 8.0, 9.0
    auto t2 = Tensor::zeros({1, 3}, Device::CUDA);
    {
        float data[3] = {7.0f, 8.0f, 9.0f};
        cudaMemcpy(t2.ptr<float>(), data, 3 * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Concatenate along dimension 0 -> should get [3, 3]
    std::vector<Tensor> tensors = {t1, t2};
    auto result = Tensor::cat(tensors, 0);

    // Verify shape
    ASSERT_EQ(result.shape().rank(), 2);
    ASSERT_EQ(result.shape()[0], 3);
    ASSERT_EQ(result.shape()[1], 3);

    // Verify values
    float result_data[9];
    cudaMemcpy(result_data, result.ptr<float>(), 9 * sizeof(float), cudaMemcpyDeviceToHost);

    float expected[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    std::cout << "Expected: ";
    for (int i = 0; i < 9; i++)
        std::cout << expected[i] << " ";
    std::cout << std::endl;

    std::cout << "Got:      ";
    for (int i = 0; i < 9; i++)
        std::cout << result_data[i] << " ";
    std::cout << std::endl;

    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(result_data[i], expected[i])
            << "Mismatch at index " << i;
    }
}

TEST(TensorCatBugTest, TrimmedTensorConcatenation) {
    // Simulate what happens in extend_state_for_new_params:
    // 1. We have an old tensor with some capacity
    // 2. We create a "trimmed" tensor by copying first N rows
    // 3. We concatenate with zeros

    // Create "old" tensor [5, 3] (simulating over-allocated state)
    auto old_tensor = Tensor::zeros({5, 3}, Device::CUDA);
    {
        float data[15] = {
            1.0f, 2.0f, 3.0f,    // row 0
            4.0f, 5.0f, 6.0f,    // row 1
            7.0f, 8.0f, 9.0f,    // row 2
            10.0f, 11.0f, 12.0f, // row 3 - UNUSED
            13.0f, 14.0f, 15.0f  // row 4 - UNUSED
        };
        cudaMemcpy(old_tensor.ptr<float>(), data, 15 * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Simulate: state.size = 3, but tensor has shape [5, 3]
    size_t state_size = 3;
    size_t n_new = 2;

    // Create trimmed tensor (copy first 3 rows only)
    auto trimmed = Tensor::zeros({state_size, 3}, Device::CUDA);
    size_t bytes_to_copy = state_size * 3 * sizeof(float);
    cudaMemcpy(trimmed.ptr<float>(), old_tensor.ptr<float>(), bytes_to_copy, cudaMemcpyDeviceToDevice);

    // Create new zeros
    auto new_zeros = Tensor::zeros({n_new, 3}, Device::CUDA);

    // Concatenate
    std::vector<Tensor> parts = {trimmed, new_zeros};
    auto result = Tensor::cat(parts, 0);

    // Verify shape
    ASSERT_EQ(result.shape()[0], 5); // 3 + 2
    ASSERT_EQ(result.shape()[1], 3);

    // Verify first 3 rows match trimmed data
    float result_data[15];
    cudaMemcpy(result_data, result.ptr<float>(), 15 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nTrimmed tensor concatenation:" << std::endl;
    for (size_t i = 0; i < 5; i++) {
        std::cout << "  Row " << i << ": "
                  << result_data[i * 3 + 0] << ", "
                  << result_data[i * 3 + 1] << ", "
                  << result_data[i * 3 + 2] << std::endl;
    }

    // First 3 rows should match
    float expected_first_9[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(result_data[i], expected_first_9[i])
            << "Mismatch in first 9 elements at index " << i;
    }

    // Last 2 rows should be zeros
    for (int i = 9; i < 15; i++) {
        EXPECT_FLOAT_EQ(result_data[i], 0.0f)
            << "New rows should be zero, but index " << i << " is " << result_data[i];
    }
}

TEST(TensorCatBugTest, RepeatedConcatenations) {
    // Test that repeated concatenations preserve data correctly
    // This simulates multiple add_new_gs operations

    auto state = Tensor::zeros({2, 3}, Device::CUDA);
    {
        float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        cudaMemcpy(state.ptr<float>(), data, 6 * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::cout << "\nRepeated concatenations test:" << std::endl;

    // First concatenation: add 1 row
    {
        auto new_zeros = Tensor::zeros({1, 3}, Device::CUDA);
        std::vector<Tensor> parts = {state, new_zeros};
        state = Tensor::cat(parts, 0);

        float data[9];
        cudaMemcpy(data, state.ptr<float>(), 9 * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "After 1st cat [3,3]: ";
        for (int i = 0; i < 9; i++)
            std::cout << data[i] << " ";
        std::cout << std::endl;

        // Check first 6 values preserved
        EXPECT_FLOAT_EQ(data[0], 1.0f);
        EXPECT_FLOAT_EQ(data[1], 2.0f);
        EXPECT_FLOAT_EQ(data[2], 3.0f);
        EXPECT_FLOAT_EQ(data[3], 4.0f);
        EXPECT_FLOAT_EQ(data[4], 5.0f);
        EXPECT_FLOAT_EQ(data[5], 6.0f);
    }

    // Second concatenation: add 2 more rows
    {
        auto new_zeros = Tensor::zeros({2, 3}, Device::CUDA);
        std::vector<Tensor> parts = {state, new_zeros};
        state = Tensor::cat(parts, 0);

        float data[15];
        cudaMemcpy(data, state.ptr<float>(), 15 * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "After 2nd cat [5,3]: ";
        for (int i = 0; i < 15; i++)
            std::cout << data[i] << " ";
        std::cout << std::endl;

        // Check first 6 values STILL preserved
        EXPECT_FLOAT_EQ(data[0], 1.0f) << "First value corrupted after 2nd concatenation!";
        EXPECT_FLOAT_EQ(data[1], 2.0f) << "Second value corrupted after 2nd concatenation!";
        EXPECT_FLOAT_EQ(data[2], 3.0f);
        EXPECT_FLOAT_EQ(data[3], 4.0f);
        EXPECT_FLOAT_EQ(data[4], 5.0f);
        EXPECT_FLOAT_EQ(data[5], 6.0f);
    }
}
