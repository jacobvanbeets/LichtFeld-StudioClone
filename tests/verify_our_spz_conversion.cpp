/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Verify our SPZ conversion by examining exported file with reference implementation
// This tests the actual data flow through our code

#include "load-spz.h"
#include <cmath>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <spz_file>\n";
        std::cerr << "Loads an SPZ file created by our implementation and dumps the data.\n";
        return 1;
    }

    fs::path spz_path = argv[1];
    if (!fs::exists(spz_path)) {
        std::cerr << "File not found: " << spz_path << "\n";
        return 1;
    }

    std::cout << "Loading SPZ file: " << spz_path << "\n";

    spz::UnpackOptions opts;
    opts.to = spz::CoordinateSystem::RDF; // Load in PLY coordinate system
    auto cloud = spz::loadSpz(spz_path.string(), opts);

    if (cloud.numPoints == 0) {
        std::cerr << "Failed to load SPZ or file is empty.\n";
        return 1;
    }

    std::cout << "\n=== SPZ File Contents ===\n";
    std::cout << "Number of points: " << cloud.numPoints << "\n";
    std::cout << "SH degree: " << cloud.shDegree << "\n";
    std::cout << "Antialiased: " << (cloud.antialiased ? "yes" : "no") << "\n";

    // Dump first few points
    int dump_count = std::min(5, cloud.numPoints);
    std::cout << "\nFirst " << dump_count << " points:\n";

    for (int i = 0; i < dump_count; ++i) {
        std::cout << "\n--- Point " << i << " ---\n";

        // Position
        std::cout << "Position: ["
                  << cloud.positions[i * 3 + 0] << ", "
                  << cloud.positions[i * 3 + 1] << ", "
                  << cloud.positions[i * 3 + 2] << "]\n";

        // Scale
        std::cout << "Scale (log): ["
                  << cloud.scales[i * 3 + 0] << ", "
                  << cloud.scales[i * 3 + 1] << ", "
                  << cloud.scales[i * 3 + 2] << "]\n";

        // Rotation (xyzw)
        std::cout << "Rotation (xyzw): ["
                  << cloud.rotations[i * 4 + 0] << ", "
                  << cloud.rotations[i * 4 + 1] << ", "
                  << cloud.rotations[i * 4 + 2] << ", "
                  << cloud.rotations[i * 4 + 3] << "]\n";

        // Alpha
        std::cout << "Alpha (logit): " << cloud.alphas[i] << "\n";

        // Color (SH DC)
        std::cout << "Color (SH DC): ["
                  << cloud.colors[i * 3 + 0] << ", "
                  << cloud.colors[i * 3 + 1] << ", "
                  << cloud.colors[i * 3 + 2] << "]\n";

        // SH coefficients
        if (cloud.shDegree > 0) {
            int sh_per_point = 0;
            switch (cloud.shDegree) {
            case 1: sh_per_point = 3; break;
            case 2: sh_per_point = 8; break;
            case 3: sh_per_point = 15; break;
            }

            std::cout << "SH coefficients (" << sh_per_point << " per channel):\n";
            for (int j = 0; j < std::min(3, sh_per_point); ++j) {
                size_t idx = i * sh_per_point * 3 + j * 3;
                std::cout << "  coeff " << j << ": ["
                          << cloud.sh[idx + 0] << ", "
                          << cloud.sh[idx + 1] << ", "
                          << cloud.sh[idx + 2] << "]\n";
            }
            if (sh_per_point > 3) {
                std::cout << "  ... (" << (sh_per_point - 3) << " more coefficients)\n";
            }
        }
    }

    // Statistics
    std::cout << "\n=== Statistics ===\n";

    float min_pos[3] = {1e10f, 1e10f, 1e10f};
    float max_pos[3] = {-1e10f, -1e10f, -1e10f};
    float sum_alpha = 0;

    for (int i = 0; i < cloud.numPoints; ++i) {
        for (int j = 0; j < 3; ++j) {
            min_pos[j] = std::min(min_pos[j], cloud.positions[i * 3 + j]);
            max_pos[j] = std::max(max_pos[j], cloud.positions[i * 3 + j]);
        }
        sum_alpha += cloud.alphas[i];
    }

    std::cout << "Position bounds:\n";
    std::cout << "  X: [" << min_pos[0] << ", " << max_pos[0] << "]\n";
    std::cout << "  Y: [" << min_pos[1] << ", " << max_pos[1] << "]\n";
    std::cout << "  Z: [" << min_pos[2] << ", " << max_pos[2] << "]\n";
    std::cout << "Average alpha (logit): " << (sum_alpha / cloud.numPoints) << "\n";

    return 0;
}
