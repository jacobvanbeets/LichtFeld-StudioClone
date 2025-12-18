/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <gtest/gtest.h>

using namespace lfs::core;

/**
 * Comprehensive CPU dtype conversion tests for large tensors
 *
 * Tests all dtype conversion pairs with various sizes to find:
 * 1. Which conversions fail with large tensors
 * 2. What the failure threshold is
 * 3. Whether the issue is CPU-specific or affects GPU too
 *
 * Dtype coverage:
 * - Float32
 * - Int32
 * - Int64
 * - UInt8
 * - Bool
 */

class CPUDtypeConversionTest : public ::testing::Test {
protected:
    // Test sizes: small, medium, large, very large (mural scale)
    const std::vector<size_t> test_sizes = {
        1000,    // 1K - small baseline
        10000,   // 10K
        100000,  // 100K - bicycle scale
        1000000, // 1M
        4000000, // 4M - mural scale (KNOWN FAILURE)
    };

    template <typename T>
    Tensor create_tensor(size_t num_elements, Device device, DataType dtype) {
        if constexpr (std::is_same_v<T, bool>) {
            // Special handling for bool - can't use vector<bool>::data()
            std::vector<uint8_t> data(num_elements);
            for (size_t i = 0; i < num_elements; ++i) {
                data[i] = (i % 2 == 0) ? 1 : 0;
            }
            return Tensor::from_blob(data.data(), {num_elements}, device, DataType::Bool).clone();
        } else {
            std::vector<T> data(num_elements);
            for (size_t i = 0; i < num_elements; ++i) {
                if constexpr (std::is_same_v<T, uint8_t>) {
                    data[i] = static_cast<T>(i % 256);
                } else if constexpr (std::is_integral_v<T>) {
                    data[i] = static_cast<T>(i % 1000);
                } else {
                    data[i] = static_cast<T>(i) / 1000.0f;
                }
            }
            return Tensor::from_blob(data.data(), {num_elements}, device, dtype).clone();
        }
    }

    void test_conversion(const std::string& test_name,
                         DataType from_dtype,
                         DataType to_dtype,
                         size_t num_elements,
                         Device device) {
        std::cout << "\n  Testing " << test_name
                  << " [" << num_elements << " elements on "
                  << (device == Device::CPU ? "CPU" : "GPU") << "]... ";

        try {
            // Create source tensor
            Tensor src;
            if (from_dtype == DataType::Float32) {
                src = create_tensor<float>(num_elements, device, from_dtype);
            } else if (from_dtype == DataType::Int32) {
                src = create_tensor<int>(num_elements, device, from_dtype);
            } else if (from_dtype == DataType::Int64) {
                src = create_tensor<int64_t>(num_elements, device, from_dtype);
            } else if (from_dtype == DataType::UInt8) {
                src = create_tensor<uint8_t>(num_elements, device, from_dtype);
            } else if (from_dtype == DataType::Bool) {
                src = create_tensor<bool>(num_elements, device, from_dtype);
            } else {
                FAIL() << "Unsupported source dtype";
            }

            ASSERT_TRUE(src.is_valid()) << "Failed to create source tensor";
            ASSERT_EQ(src.dtype(), from_dtype);
            ASSERT_EQ(src.device(), device);

            // Perform conversion
            auto dst = src.to(to_dtype);

            // Validate result
            ASSERT_TRUE(dst.is_valid()) << "Conversion produced invalid tensor";
            ASSERT_EQ(dst.dtype(), to_dtype) << "Wrong output dtype";
            ASSERT_EQ(dst.device(), device) << "Device changed during conversion";
            ASSERT_EQ(dst.numel(), src.numel()) << "Element count changed";

            std::cout << "✓ PASS";
        } catch (const std::exception& e) {
            std::cout << "✗ FAIL: " << e.what();
            FAIL() << "Conversion threw exception: " << e.what();
        }
    }
};

// ============================================================================
// Float32 conversions
// ============================================================================

TEST_F(CPUDtypeConversionTest, Float32_to_Int32_CPU_AllSizes) {
    std::cout << "\n=== Float32 -> Int32 (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("Float32->Int32", DataType::Float32, DataType::Int32, size, Device::CPU);
    }
}

TEST_F(CPUDtypeConversionTest, Float32_to_Int64_CPU_AllSizes) {
    std::cout << "\n=== Float32 -> Int64 (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("Float32->Int64", DataType::Float32, DataType::Int64, size, Device::CPU);
    }
}

TEST_F(CPUDtypeConversionTest, Float32_to_UInt8_CPU_AllSizes) {
    std::cout << "\n=== Float32 -> UInt8 (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("Float32->UInt8", DataType::Float32, DataType::UInt8, size, Device::CPU);
    }
}

TEST_F(CPUDtypeConversionTest, Float32_to_Bool_CPU_AllSizes) {
    std::cout << "\n=== Float32 -> Bool (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("Float32->Bool", DataType::Float32, DataType::Bool, size, Device::CPU);
    }
}

// ============================================================================
// Int32 conversions
// ============================================================================

TEST_F(CPUDtypeConversionTest, Int32_to_Float32_CPU_AllSizes) {
    std::cout << "\n=== Int32 -> Float32 (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("Int32->Float32", DataType::Int32, DataType::Float32, size, Device::CPU);
    }
}

TEST_F(CPUDtypeConversionTest, Int32_to_Int64_CPU_AllSizes) {
    std::cout << "\n=== Int32 -> Int64 (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("Int32->Int64", DataType::Int32, DataType::Int64, size, Device::CPU);
    }
}

TEST_F(CPUDtypeConversionTest, Int32_to_UInt8_CPU_AllSizes) {
    std::cout << "\n=== Int32 -> UInt8 (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("Int32->UInt8", DataType::Int32, DataType::UInt8, size, Device::CPU);
    }
}

TEST_F(CPUDtypeConversionTest, Int32_to_Bool_CPU_AllSizes) {
    std::cout << "\n=== Int32 -> Bool (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("Int32->Bool", DataType::Int32, DataType::Bool, size, Device::CPU);
    }
}

// ============================================================================
// Int64 conversions
// ============================================================================

TEST_F(CPUDtypeConversionTest, Int64_to_Float32_CPU_AllSizes) {
    std::cout << "\n=== Int64 -> Float32 (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("Int64->Float32", DataType::Int64, DataType::Float32, size, Device::CPU);
    }
}

TEST_F(CPUDtypeConversionTest, Int64_to_Int32_CPU_AllSizes) {
    std::cout << "\n=== Int64 -> Int32 (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("Int64->Int32", DataType::Int64, DataType::Int32, size, Device::CPU);
    }
}

TEST_F(CPUDtypeConversionTest, Int64_to_UInt8_CPU_AllSizes) {
    std::cout << "\n=== Int64 -> UInt8 (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("Int64->UInt8", DataType::Int64, DataType::UInt8, size, Device::CPU);
    }
}

// ============================================================================
// UInt8 conversions (THE KNOWN FAILURE CASES)
// ============================================================================

TEST_F(CPUDtypeConversionTest, UInt8_to_Float32_CPU_AllSizes) {
    std::cout << "\n=== UInt8 -> Float32 (CPU) [KNOWN FAILURE @ 4M] ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("UInt8->Float32", DataType::UInt8, DataType::Float32, size, Device::CPU);
    }
}

TEST_F(CPUDtypeConversionTest, UInt8_to_Int32_CPU_AllSizes) {
    std::cout << "\n=== UInt8 -> Int32 (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("UInt8->Int32", DataType::UInt8, DataType::Int32, size, Device::CPU);
    }
}

TEST_F(CPUDtypeConversionTest, UInt8_to_Int64_CPU_AllSizes) {
    std::cout << "\n=== UInt8 -> Int64 (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("UInt8->Int64", DataType::UInt8, DataType::Int64, size, Device::CPU);
    }
}

TEST_F(CPUDtypeConversionTest, UInt8_to_Bool_CPU_AllSizes) {
    std::cout << "\n=== UInt8 -> Bool (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("UInt8->Bool", DataType::UInt8, DataType::Bool, size, Device::CPU);
    }
}

// ============================================================================
// Bool conversions
// ============================================================================

TEST_F(CPUDtypeConversionTest, Bool_to_Float32_CPU_AllSizes) {
    std::cout << "\n=== Bool -> Float32 (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("Bool->Float32", DataType::Bool, DataType::Float32, size, Device::CPU);
    }
}

TEST_F(CPUDtypeConversionTest, Bool_to_Int32_CPU_AllSizes) {
    std::cout << "\n=== Bool -> Int32 (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("Bool->Int32", DataType::Bool, DataType::Int32, size, Device::CPU);
    }
}

TEST_F(CPUDtypeConversionTest, Bool_to_UInt8_CPU_AllSizes) {
    std::cout << "\n=== Bool -> UInt8 (CPU) ===" << std::endl;
    for (size_t size : test_sizes) {
        test_conversion("Bool->UInt8", DataType::Bool, DataType::UInt8, size, Device::CPU);
    }
}

// ============================================================================
// GPU conversions for comparison (should work)
// ============================================================================

TEST_F(CPUDtypeConversionTest, UInt8_to_Float32_GPU_AllSizes) {
    std::cout << "\n=== UInt8 -> Float32 (GPU) [Should work] ===" << std::endl;
    for (size_t size : test_sizes) {
        // Create on CPU, move to GPU
        auto cpu_tensor = create_tensor<uint8_t>(size, Device::CPU, DataType::UInt8);
        auto gpu_tensor = cpu_tensor.cuda();

        std::cout << "\n  Testing UInt8->Float32 [" << size << " elements on GPU]... ";
        try {
            auto result = gpu_tensor.to(DataType::Float32);
            ASSERT_TRUE(result.is_valid());
            ASSERT_EQ(result.dtype(), DataType::Float32);
            ASSERT_EQ(result.device(), Device::CUDA);
            std::cout << "✓ PASS";
        } catch (const std::exception& e) {
            std::cout << "✗ FAIL: " << e.what();
            FAIL() << "GPU conversion failed: " << e.what();
        }
    }
}

// Note: Multi-dimensional tests removed for simplicity - 1D tests are sufficient
// to identify the CPU conversion bugs

// ============================================================================
// CPU->GPU transfer tests (separate from conversion)
// ============================================================================

TEST_F(CPUDtypeConversionTest, CPUtoGPU_Transfer_AllDtypes) {
    std::cout << "\n=== CPU->GPU Transfer (All dtypes, 4M elements) ===" << std::endl;

    const size_t num_elements = 4000000;

    // Float32
    std::cout << "\n  Float32... ";
    {
        auto cpu = create_tensor<float>(num_elements, Device::CPU, DataType::Float32);
        auto gpu = cpu.cuda();
        ASSERT_TRUE(gpu.is_valid());
        ASSERT_EQ(gpu.device(), Device::CUDA);
        std::cout << "✓";
    }

    // Int32
    std::cout << "\n  Int32... ";
    {
        auto cpu = create_tensor<int>(num_elements, Device::CPU, DataType::Int32);
        auto gpu = cpu.cuda();
        ASSERT_TRUE(gpu.is_valid());
        ASSERT_EQ(gpu.device(), Device::CUDA);
        std::cout << "✓";
    }

    // Int64
    std::cout << "\n  Int64... ";
    {
        auto cpu = create_tensor<int64_t>(num_elements, Device::CPU, DataType::Int64);
        auto gpu = cpu.cuda();
        ASSERT_TRUE(gpu.is_valid());
        ASSERT_EQ(gpu.device(), Device::CUDA);
        std::cout << "✓";
    }

    // UInt8
    std::cout << "\n  UInt8... ";
    {
        auto cpu = create_tensor<uint8_t>(num_elements, Device::CPU, DataType::UInt8);
        auto gpu = cpu.cuda();
        ASSERT_TRUE(gpu.is_valid());
        ASSERT_EQ(gpu.device(), Device::CUDA);
        std::cout << "✓";
    }

    // Bool
    std::cout << "\n  Bool... ";
    {
        auto cpu = create_tensor<bool>(num_elements, Device::CPU, DataType::Bool);
        auto gpu = cpu.cuda();
        ASSERT_TRUE(gpu.is_valid());
        ASSERT_EQ(gpu.device(), Device::CUDA);
        std::cout << "✓";
    }

    std::cout << std::endl;
}

// ============================================================================
// Stress test: Find exact failure point for UInt8->Float32
// ============================================================================

TEST_F(CPUDtypeConversionTest, UInt8_to_Float32_CPU_BinarySearch) {
    std::cout << "\n=== Binary Search for UInt8->Float32 CPU Failure Point ===" << std::endl;

    // Known working: 1M, Known failing: 4M
    // Binary search between them
    size_t working_max = 1000000;
    size_t failing_min = 4000000;

    std::vector<size_t> probe_sizes = {
        1500000, // 1.5M
        2000000, // 2M
        2500000, // 2.5M
        3000000, // 3M
        3500000, // 3.5M
    };

    for (size_t size : probe_sizes) {
        std::cout << "\n  Probing " << size << " elements... ";
        try {
            auto uint8_tensor = create_tensor<uint8_t>(size, Device::CPU, DataType::UInt8);
            auto float32_tensor = uint8_tensor.to(DataType::Float32);

            if (float32_tensor.is_valid()) {
                std::cout << "✓ SUCCESS (failure threshold > " << size << ")";
                working_max = std::max(working_max, size);
            } else {
                std::cout << "✗ INVALID RESULT";
                failing_min = std::min(failing_min, size);
            }
        } catch (const std::exception& e) {
            std::cout << "✗ EXCEPTION: " << e.what();
            failing_min = std::min(failing_min, size);
        }
    }

    std::cout << "\n\n  Result: Works up to " << working_max
              << ", fails at " << failing_min << std::endl;
}

// ============================================================================
// Memory corruption test
// ============================================================================

TEST_F(CPUDtypeConversionTest, UInt8_to_Float32_CPU_DataIntegrity) {
    std::cout << "\n=== UInt8->Float32 CPU Data Integrity Check ===" << std::endl;

    const size_t test_size = 100000; // Use safe size

    // Create deterministic data
    std::vector<uint8_t> data(test_size);
    for (size_t i = 0; i < test_size; ++i) {
        data[i] = static_cast<uint8_t>((i * 7 + 13) % 256);
    }

    auto uint8_cpu = Tensor::from_blob(data.data(), {test_size}, Device::CPU, DataType::UInt8).clone();

    // Convert on CPU
    auto float32_cpu = uint8_cpu.to(DataType::Float32);

    // Verify each value
    auto acc_u8 = uint8_cpu.accessor<uint8_t, 1>();
    auto acc_f32 = float32_cpu.accessor<float, 1>();

    bool all_correct = true;
    size_t num_errors = 0;
    for (size_t i = 0; i < test_size && num_errors < 10; ++i) {
        float expected = static_cast<float>(acc_u8(i));
        float actual = acc_f32(i);
        if (std::abs(expected - actual) > 1e-5) {
            if (num_errors < 5) {
                std::cout << "\n  ERROR at index " << i << ": expected "
                          << expected << ", got " << actual;
            }
            all_correct = false;
            num_errors++;
        }
    }

    if (all_correct) {
        std::cout << "\n  ✓ All " << test_size << " values correct" << std::endl;
    } else {
        std::cout << "\n  ✗ Found " << num_errors << " errors (showing first 5)" << std::endl;
        FAIL() << "Data integrity check failed";
    }
}
