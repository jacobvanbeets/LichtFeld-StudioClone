/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Standalone SPZ verification test - doesn't require LibTorch or GTest
// Build: g++ -std=c++23 -I../include -I../external/spz -I../src -o verify_spz verify_spz_format.cpp
// Run: ./verify_spz

#include "load-spz.h"
#include <cmath>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

bool approxEqual(float a, float b, float eps = 0.01f) {
    return std::abs(a - b) < eps;
}

int main() {
    std::cout << "=== SPZ Format Verification ===\n\n";

    // Test 1: Create a known GaussianCloud and verify layout
    std::cout << "Test 1: GaussianCloud layout verification\n";
    {
        spz::GaussianCloud cloud;
        cloud.numPoints = 2;
        cloud.shDegree = 1;

        // Positions
        cloud.positions = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

        // Scales
        cloud.scales = {-2.0f, -2.0f, -2.0f, -3.0f, -3.0f, -3.0f};

        // Rotations (xyzw format)
        cloud.rotations = {0.0f, 0.0f, 0.0f, 1.0f,  // Identity
                           0.5f, 0.5f, 0.5f, 0.5f}; // Non-identity

        // Alphas
        cloud.alphas = {0.0f, 1.0f};

        // Colors (DC)
        cloud.colors = {0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f};

        // SH coefficients - 3 per channel, interleaved RGB
        // Point 0: sh0=[0.1, 0.2, 0.3], sh1=[0.4, 0.5, 0.6], sh2=[0.7, 0.8, 0.9]
        // Point 1: all zeros
        cloud.sh = {
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, // Point 0
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f  // Point 1
        };

        // Save to temp file
        fs::path tmp = fs::temp_directory_path() / "verify_spz_test.spz";
        spz::PackOptions pack_opts;
        pack_opts.from = spz::CoordinateSystem::RDF;

        if (!spz::saveSpz(cloud, pack_opts, tmp.string())) {
            std::cerr << "FAIL: Could not save SPZ\n";
            return 1;
        }

        // Load back
        spz::UnpackOptions unpack_opts;
        unpack_opts.to = spz::CoordinateSystem::RDF;
        auto loaded = spz::loadSpz(tmp.string(), unpack_opts);

        if (loaded.numPoints != 2) {
            std::cerr << "FAIL: Point count mismatch: " << loaded.numPoints << " != 2\n";
            return 1;
        }

        if (loaded.shDegree != 1) {
            std::cerr << "FAIL: SH degree mismatch: " << loaded.shDegree << " != 1\n";
            return 1;
        }

        // Verify rotation of point 0 (should be identity)
        std::cout << "  Point 0 rotation (xyzw): ["
                  << loaded.rotations[0] << ", " << loaded.rotations[1] << ", "
                  << loaded.rotations[2] << ", " << loaded.rotations[3] << "]\n";
        std::cout << "  Expected: [0, 0, 0, 1]\n";

        // Check SH layout
        std::cout << "  Point 0 SH coefficients:\n";
        for (int j = 0; j < 3; ++j) {
            std::cout << "    coeff " << j << ": ["
                      << loaded.sh[j * 3 + 0] << ", "
                      << loaded.sh[j * 3 + 1] << ", "
                      << loaded.sh[j * 3 + 2] << "]\n";
        }
        std::cout << "  Expected: coeff 0=[0.1, 0.2, 0.3], coeff 1=[0.4, 0.5, 0.6], coeff 2=[0.7, 0.8, 0.9]\n";

        // Verify SH values (with SPZ lossy compression tolerance)
        bool sh_ok = true;
        float expected_sh[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
        for (int i = 0; i < 9; ++i) {
            if (!approxEqual(loaded.sh[i], expected_sh[i], 0.1f)) {
                std::cout << "  WARNING: SH[" << i << "] = " << loaded.sh[i]
                          << " vs expected " << expected_sh[i] << "\n";
                sh_ok = false;
            }
        }

        if (sh_ok) {
            std::cout << "  PASS: SH coefficients match (within tolerance)\n";
        }

        fs::remove(tmp);
    }

    // Test 2: Verify coordinate conversion matches PLY conventions
    std::cout << "\nTest 2: Coordinate system verification\n";
    {
        spz::GaussianCloud cloud;
        cloud.numPoints = 1;
        cloud.shDegree = 0;
        cloud.positions = {1.0f, 2.0f, 3.0f};
        cloud.scales = {0.0f, 0.0f, 0.0f};
        cloud.rotations = {0.0f, 0.0f, 0.0f, 1.0f}; // xyzw
        cloud.alphas = {0.0f};
        cloud.colors = {0.5f, 0.5f, 0.5f};

        fs::path tmp = fs::temp_directory_path() / "coord_test.spz";

        // Save with RDF (PLY convention)
        spz::PackOptions pack_opts;
        pack_opts.from = spz::CoordinateSystem::RDF;
        spz::saveSpz(cloud, pack_opts, tmp.string());

        // Load with RDF (PLY convention)
        spz::UnpackOptions unpack_opts;
        unpack_opts.to = spz::CoordinateSystem::RDF;
        auto loaded = spz::loadSpz(tmp.string(), unpack_opts);

        std::cout << "  Original position: [1, 2, 3]\n";
        std::cout << "  Loaded position: [" << loaded.positions[0] << ", "
                  << loaded.positions[1] << ", " << loaded.positions[2] << "]\n";

        if (approxEqual(loaded.positions[0], 1.0f, 0.01f) &&
            approxEqual(loaded.positions[1], 2.0f, 0.01f) &&
            approxEqual(loaded.positions[2], 3.0f, 0.01f)) {
            std::cout << "  PASS: Position preserved\n";
        } else {
            std::cout << "  FAIL: Position changed\n";
        }

        fs::remove(tmp);
    }

    // Test 3: Quaternion format verification
    std::cout << "\nTest 3: Quaternion format (xyzw vs wxyz)\n";
    {
        // SPZ uses xyzw format
        // Our SplatData uses wxyz format
        // This is the key conversion we need to test

        // Create a 90-degree rotation around Z axis
        // In wxyz: [w=0.7071, x=0, y=0, z=0.7071]
        // In xyzw: [x=0, y=0, z=0.7071, w=0.7071]

        spz::GaussianCloud cloud;
        cloud.numPoints = 1;
        cloud.shDegree = 0;
        cloud.positions = {0.0f, 0.0f, 0.0f};
        cloud.scales = {0.0f, 0.0f, 0.0f};
        cloud.rotations = {0.0f, 0.0f, 0.7071f, 0.7071f}; // xyzw: 90deg around Z
        cloud.alphas = {0.0f};
        cloud.colors = {0.5f, 0.5f, 0.5f};

        fs::path tmp = fs::temp_directory_path() / "quat_test.spz";
        spz::PackOptions pack_opts;
        pack_opts.from = spz::CoordinateSystem::RDF;
        spz::saveSpz(cloud, pack_opts, tmp.string());

        spz::UnpackOptions unpack_opts;
        unpack_opts.to = spz::CoordinateSystem::RDF;
        auto loaded = spz::loadSpz(tmp.string(), unpack_opts);

        std::cout << "  Original (xyzw): [0, 0, 0.7071, 0.7071]\n";
        std::cout << "  Loaded (xyzw): ["
                  << loaded.rotations[0] << ", " << loaded.rotations[1] << ", "
                  << loaded.rotations[2] << ", " << loaded.rotations[3] << "]\n";

        // Our conversion should produce wxyz: [0.7071, 0, 0, 0.7071]
        std::cout << "  Our SplatData wxyz should be: [w=0.7071, x=0, y=0, z=0.7071]\n";

        if (approxEqual(loaded.rotations[0], 0.0f, 0.1f) &&
            approxEqual(loaded.rotations[1], 0.0f, 0.1f) &&
            std::abs(loaded.rotations[2]) > 0.5f && // z should be ~0.7071
            std::abs(loaded.rotations[3]) > 0.5f) { // w should be ~0.7071
            std::cout << "  PASS: Quaternion format correct\n";
        } else {
            std::cout << "  WARNING: Quaternion may be incorrect\n";
        }

        fs::remove(tmp);
    }

    std::cout << "\n=== Verification Complete ===\n";
    return 0;
}
