/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <gtest/gtest.h>

using namespace lfs::core;

/**
 * Test suite for CPU tensor bugs found during mural dataset loading:
 *
 * Bug 1: UInt8->Float32 conversion segfaults on CPU with 4M+ points
 * Bug 2: Large CPU tensor .cuda() transfer segfaults with 4M+ points
 *
 * These tests systematically check various tensor sizes to identify
 * the exact failure point and validate fixes.
 */

class CPULargeTensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Empty - tests will create their own tensors
    }

    void TearDown() override {
        // Empty - RAII handles cleanup
    }

    // Helper to create a point cloud-like tensor
    Tensor create_point_cloud_positions(size_t num_points, Device device) {
        std::vector<float> data(num_points * 3);
        for (size_t i = 0; i < num_points * 3; ++i) {
            data[i] = static_cast<float>(i % 100) / 100.0f;
        }
        return Tensor::from_vector(data, {num_points, 3}, device);
    }

    // Helper to create a point cloud color tensor (UInt8)
    Tensor create_point_cloud_colors(size_t num_points, Device device) {
        std::vector<uint8_t> data(num_points * 3);
        for (size_t i = 0; i < num_points * 3; ++i) {
            data[i] = static_cast<uint8_t>(i % 256);
        }
        return Tensor::from_blob(data.data(), {num_points, 3}, device, DataType::UInt8).clone();
    }
};

// ============================================================================
// Test 1: Small tensors (baseline - should always work)
// ============================================================================

TEST_F(CPULargeTensorTest, SmallTensor_CPUtoGPU_Float32) {
    const size_t num_points = 1000;
    auto cpu_tensor = create_point_cloud_positions(num_points, Device::CPU);

    EXPECT_TRUE(cpu_tensor.is_valid());
    EXPECT_EQ(cpu_tensor.device(), Device::CPU);
    EXPECT_EQ(cpu_tensor.size(0), num_points);

    // Transfer to GPU
    auto gpu_tensor = cpu_tensor.cuda();

    EXPECT_TRUE(gpu_tensor.is_valid());
    EXPECT_EQ(gpu_tensor.device(), Device::CUDA);
    EXPECT_EQ(gpu_tensor.size(0), num_points);

    // Verify data integrity
    auto back_to_cpu = gpu_tensor.cpu();
    EXPECT_TRUE(back_to_cpu.all_close(cpu_tensor, 1e-5));
}

TEST_F(CPULargeTensorTest, SmallTensor_UInt8ToFloat32_CPU) {
    const size_t num_points = 1000;
    auto uint8_tensor = create_point_cloud_colors(num_points, Device::CPU);

    EXPECT_TRUE(uint8_tensor.is_valid());
    EXPECT_EQ(uint8_tensor.dtype(), DataType::UInt8);
    EXPECT_EQ(uint8_tensor.device(), Device::CPU);

    // Convert to Float32 on CPU
    auto float32_tensor = uint8_tensor.to(DataType::Float32);

    EXPECT_TRUE(float32_tensor.is_valid());
    EXPECT_EQ(float32_tensor.dtype(), DataType::Float32);
    EXPECT_EQ(float32_tensor.device(), Device::CPU);
    EXPECT_EQ(float32_tensor.size(0), num_points);
}

TEST_F(CPULargeTensorTest, SmallTensor_UInt8ToFloat32_GPU) {
    const size_t num_points = 1000;
    auto uint8_cpu = create_point_cloud_colors(num_points, Device::CPU);
    auto uint8_gpu = uint8_cpu.cuda();

    EXPECT_EQ(uint8_gpu.device(), Device::CUDA);
    EXPECT_EQ(uint8_gpu.dtype(), DataType::UInt8);

    // Convert to Float32 on GPU
    auto float32_gpu = uint8_gpu.to(DataType::Float32);

    EXPECT_TRUE(float32_gpu.is_valid());
    EXPECT_EQ(float32_gpu.dtype(), DataType::Float32);
    EXPECT_EQ(float32_gpu.device(), Device::CUDA);
    EXPECT_EQ(float32_gpu.size(0), num_points);
}

// ============================================================================
// Test 2: Medium tensors (100K points - bicycle dataset scale)
// ============================================================================

TEST_F(CPULargeTensorTest, MediumTensor_CPUtoGPU_Float32) {
    const size_t num_points = 100000;  // 100K points
    auto cpu_tensor = create_point_cloud_positions(num_points, Device::CPU);

    EXPECT_TRUE(cpu_tensor.is_valid());

    // Transfer to GPU
    auto gpu_tensor = cpu_tensor.cuda();

    EXPECT_TRUE(gpu_tensor.is_valid());
    EXPECT_EQ(gpu_tensor.device(), Device::CUDA);
    EXPECT_EQ(gpu_tensor.size(0), num_points);
}

TEST_F(CPULargeTensorTest, MediumTensor_UInt8ToFloat32_CPU) {
    const size_t num_points = 100000;
    auto uint8_tensor = create_point_cloud_colors(num_points, Device::CPU);

    EXPECT_TRUE(uint8_tensor.is_valid());

    // Convert to Float32 on CPU
    auto float32_tensor = uint8_tensor.to(DataType::Float32);

    EXPECT_TRUE(float32_tensor.is_valid());
    EXPECT_EQ(float32_tensor.dtype(), DataType::Float32);
    EXPECT_EQ(float32_tensor.size(0), num_points);
}

TEST_F(CPULargeTensorTest, MediumTensor_UInt8ToFloat32_GPU) {
    const size_t num_points = 100000;
    auto uint8_cpu = create_point_cloud_colors(num_points, Device::CPU);
    auto uint8_gpu = uint8_cpu.cuda();

    // Convert to Float32 on GPU
    auto float32_gpu = uint8_gpu.to(DataType::Float32);

    EXPECT_TRUE(float32_gpu.is_valid());
    EXPECT_EQ(float32_gpu.dtype(), DataType::Float32);
    EXPECT_EQ(float32_gpu.size(0), num_points);
}

// ============================================================================
// Test 3: Large tensors (1M points)
// ============================================================================

TEST_F(CPULargeTensorTest, LargeTensor_CPUtoGPU_Float32) {
    const size_t num_points = 1000000;  // 1M points
    auto cpu_tensor = create_point_cloud_positions(num_points, Device::CPU);

    EXPECT_TRUE(cpu_tensor.is_valid());

    // Transfer to GPU
    auto gpu_tensor = cpu_tensor.cuda();

    EXPECT_TRUE(gpu_tensor.is_valid());
    EXPECT_EQ(gpu_tensor.device(), Device::CUDA);
    EXPECT_EQ(gpu_tensor.size(0), num_points);
}

TEST_F(CPULargeTensorTest, LargeTensor_UInt8ToFloat32_CPU) {
    const size_t num_points = 1000000;
    auto uint8_tensor = create_point_cloud_colors(num_points, Device::CPU);

    EXPECT_TRUE(uint8_tensor.is_valid());

    // Convert to Float32 on CPU
    auto float32_tensor = uint8_tensor.to(DataType::Float32);

    EXPECT_TRUE(float32_tensor.is_valid());
    EXPECT_EQ(float32_tensor.dtype(), DataType::Float32);
    EXPECT_EQ(float32_tensor.size(0), num_points);
}

TEST_F(CPULargeTensorTest, LargeTensor_UInt8ToFloat32_GPU) {
    const size_t num_points = 1000000;
    auto uint8_cpu = create_point_cloud_colors(num_points, Device::CPU);
    auto uint8_gpu = uint8_cpu.cuda();

    // Convert to Float32 on GPU
    auto float32_gpu = uint8_gpu.to(DataType::Float32);

    EXPECT_TRUE(float32_gpu.is_valid());
    EXPECT_EQ(float32_gpu.dtype(), DataType::Float32);
    EXPECT_EQ(float32_gpu.size(0), num_points);
}

// ============================================================================
// Test 4: Very large tensors (4M points - mural dataset scale) - THE FAILING CASE
// ============================================================================

TEST_F(CPULargeTensorTest, VeryLargeTensor_CPUtoGPU_Float32) {
    const size_t num_points = 4042850;  // Exact mural dataset size

    std::cout << "Creating CPU tensor with " << num_points << " points..." << std::endl;
    auto cpu_tensor = create_point_cloud_positions(num_points, Device::CPU);

    EXPECT_TRUE(cpu_tensor.is_valid());
    EXPECT_EQ(cpu_tensor.device(), Device::CPU);
    EXPECT_EQ(cpu_tensor.size(0), num_points);
    std::cout << "CPU tensor created successfully" << std::endl;

    // Transfer to GPU - THIS IS WHERE IT CRASHES IN PRODUCTION
    std::cout << "Transferring to GPU..." << std::endl;
    auto gpu_tensor = cpu_tensor.cuda();

    EXPECT_TRUE(gpu_tensor.is_valid());
    EXPECT_EQ(gpu_tensor.device(), Device::CUDA);
    EXPECT_EQ(gpu_tensor.size(0), num_points);
    std::cout << "Transfer successful!" << std::endl;
}

TEST_F(CPULargeTensorTest, VeryLargeTensor_UInt8ToFloat32_CPU) {
    const size_t num_points = 4042850;  // Exact mural dataset size

    std::cout << "Creating UInt8 tensor with " << num_points << " points..." << std::endl;
    auto uint8_tensor = create_point_cloud_colors(num_points, Device::CPU);

    EXPECT_TRUE(uint8_tensor.is_valid());
    EXPECT_EQ(uint8_tensor.dtype(), DataType::UInt8);
    std::cout << "UInt8 tensor created successfully" << std::endl;

    // Convert to Float32 on CPU - THIS IS WHERE IT CRASHES IN PRODUCTION
    std::cout << "Converting UInt8->Float32 on CPU..." << std::endl;
    auto float32_tensor = uint8_tensor.to(DataType::Float32);

    EXPECT_TRUE(float32_tensor.is_valid());
    EXPECT_EQ(float32_tensor.dtype(), DataType::Float32);
    EXPECT_EQ(float32_tensor.size(0), num_points);
    std::cout << "Conversion successful!" << std::endl;
}

TEST_F(CPULargeTensorTest, VeryLargeTensor_UInt8ToFloat32_GPU) {
    const size_t num_points = 4042850;  // Exact mural dataset size

    std::cout << "Creating UInt8 tensor on CPU..." << std::endl;
    auto uint8_cpu = create_point_cloud_colors(num_points, Device::CPU);

    EXPECT_TRUE(uint8_cpu.is_valid());
    std::cout << "Moving UInt8 to GPU..." << std::endl;
    auto uint8_gpu = uint8_cpu.cuda();

    EXPECT_EQ(uint8_gpu.device(), Device::CUDA);
    std::cout << "Converting UInt8->Float32 on GPU..." << std::endl;

    // Convert to Float32 on GPU - This should work (workaround)
    auto float32_gpu = uint8_gpu.to(DataType::Float32);

    EXPECT_TRUE(float32_gpu.is_valid());
    EXPECT_EQ(float32_gpu.dtype(), DataType::Float32);
    EXPECT_EQ(float32_gpu.size(0), num_points);
    std::cout << "GPU conversion successful!" << std::endl;
}

// ============================================================================
// Test 5: Exact production workflow - mural point cloud loading
// ============================================================================

TEST_F(CPULargeTensorTest, Production_MuralPointCloudWorkflow) {
    const size_t num_points = 4042850;

    std::cout << "\n=== Simulating mural point cloud loading ===" << std::endl;

    // Step 1: Load point cloud on CPU (like colmap.cpp does now)
    std::cout << "1. Loading point cloud on CPU..." << std::endl;
    auto means_cpu = create_point_cloud_positions(num_points, Device::CPU);
    auto colors_cpu = create_point_cloud_colors(num_points, Device::CPU);

    EXPECT_TRUE(means_cpu.is_valid());
    EXPECT_TRUE(colors_cpu.is_valid());
    std::cout << "   ✓ Point cloud loaded on CPU" << std::endl;

    // Step 2: OLD APPROACH (BROKEN) - Convert colors on CPU then move to GPU
    std::cout << "\n2. Testing OLD approach (CPU conversion)..." << std::endl;
    try {
        auto colors_float_cpu = colors_cpu.to(DataType::Float32);
        std::cout << "   ✓ UInt8->Float32 on CPU succeeded (unexpected!)" << std::endl;
        EXPECT_TRUE(colors_float_cpu.is_valid());
    } catch (...) {
        std::cout << "   ✗ UInt8->Float32 on CPU failed (expected)" << std::endl;
        FAIL() << "CPU conversion should not throw exceptions";
    }

    // Step 3: NEW APPROACH (WORKAROUND) - Move to GPU first, then convert
    std::cout << "\n3. Testing NEW approach (GPU conversion)..." << std::endl;
    auto means_gpu = means_cpu.cuda();
    std::cout << "   ✓ Means moved to GPU" << std::endl;

    auto colors_uint8_gpu = colors_cpu.cuda();
    std::cout << "   ✓ Colors (UInt8) moved to GPU" << std::endl;

    auto colors_float_gpu = colors_uint8_gpu.to(DataType::Float32);
    std::cout << "   ✓ UInt8->Float32 on GPU succeeded" << std::endl;

    auto colors_normalized = colors_float_gpu.div(255.0f);
    std::cout << "   ✓ Colors normalized" << std::endl;

    EXPECT_TRUE(means_gpu.is_valid());
    EXPECT_TRUE(colors_normalized.is_valid());
    EXPECT_EQ(means_gpu.device(), Device::CUDA);
    EXPECT_EQ(colors_normalized.device(), Device::CUDA);

    std::cout << "\n=== Workflow validation complete ===" << std::endl;
}

// ============================================================================
// Test 6: Binary search for exact failure threshold
// ============================================================================

TEST_F(CPULargeTensorTest, FindFailureThreshold_UInt8ToFloat32) {
    std::cout << "\n=== Binary search for CPU conversion failure threshold ===" << std::endl;

    // Test various sizes to find where it breaks
    std::vector<size_t> test_sizes = {
        10000,      // 10K
        50000,      // 50K
        100000,     // 100K
        500000,     // 500K
        1000000,    // 1M
        2000000,    // 2M
        3000000,    // 3M
        4000000,    // 4M
        4042850,    // Exact mural size
    };

    for (size_t num_points : test_sizes) {
        std::cout << "\nTesting " << num_points << " points..." << std::endl;

        try {
            auto uint8_tensor = create_point_cloud_colors(num_points, Device::CPU);
            auto float32_tensor = uint8_tensor.to(DataType::Float32);

            if (float32_tensor.is_valid()) {
                std::cout << "  ✓ " << num_points << " points: SUCCESS" << std::endl;
            } else {
                std::cout << "  ✗ " << num_points << " points: INVALID RESULT" << std::endl;
                FAIL() << "Conversion produced invalid tensor at " << num_points << " points";
            }
        } catch (const std::exception& e) {
            std::cout << "  ✗ " << num_points << " points: EXCEPTION - " << e.what() << std::endl;
            FAIL() << "Conversion threw exception at " << num_points << " points: " << e.what();
        }
    }
}

// ============================================================================
// Test 7: Memory integrity check
// ============================================================================

TEST_F(CPULargeTensorTest, MemoryIntegrity_LargeConversion) {
    const size_t num_points = 100000;  // Use medium size for integrity check

    // Create UInt8 tensor with known pattern
    std::vector<uint8_t> data(num_points * 3);
    for (size_t i = 0; i < num_points * 3; ++i) {
        data[i] = static_cast<uint8_t>((i * 7) % 256);  // Deterministic pattern
    }

    auto uint8_cpu = Tensor::from_blob(data.data(), {num_points, 3}, Device::CPU, DataType::UInt8).clone();

    // Convert on CPU
    auto float32_cpu = uint8_cpu.to(DataType::Float32);

    // Convert on GPU (workaround)
    auto uint8_gpu = uint8_cpu.cuda();
    auto float32_gpu = uint8_gpu.to(DataType::Float32).cpu();

    // Both should produce the same result
    EXPECT_TRUE(float32_cpu.all_close(float32_gpu, 1e-5))
        << "CPU and GPU conversions produced different results";
}
