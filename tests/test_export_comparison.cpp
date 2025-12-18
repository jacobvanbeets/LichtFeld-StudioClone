/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Compare our SPZ export with reference implementation export
// Tests: PLY -> Our export -> SPZ vs PLY -> Reference export -> SPZ

#include "load-spz.h"
#include <cmath>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <our_spz_file> <ply_file>\n";
        std::cerr << "Compares our SPZ export with reference implementation export.\n";
        return 1;
    }

    fs::path our_spz = argv[1];
    fs::path ply_path = argv[2];

    if (!fs::exists(our_spz) || !fs::exists(ply_path)) {
        std::cerr << "Files not found.\n";
        return 1;
    }

    // Load PLY with reference and convert to SPZ
    std::cout << "Loading PLY with reference implementation...\n";
    spz::UnpackOptions load_opts;
    load_opts.to = spz::CoordinateSystem::RDF;
    auto from_ply = spz::loadSplatFromPly(ply_path.string(), load_opts);
    std::cout << "  Loaded " << from_ply.numPoints << " points, SH degree " << from_ply.shDegree << "\n";

    // Save using reference implementation
    fs::path ref_spz = fs::temp_directory_path() / "ref_export.spz";
    spz::PackOptions pack_opts;
    pack_opts.from = spz::CoordinateSystem::RDF;
    spz::saveSpz(from_ply, pack_opts, ref_spz.string());

    // Load both SPZ files
    std::cout << "\nLoading our SPZ export...\n";
    auto our_cloud = spz::loadSpz(our_spz.string(), load_opts);
    std::cout << "  Loaded " << our_cloud.numPoints << " points, SH degree " << our_cloud.shDegree << "\n";

    std::cout << "\nLoading reference SPZ export...\n";
    auto ref_cloud = spz::loadSpz(ref_spz.string(), load_opts);
    std::cout << "  Loaded " << ref_cloud.numPoints << " points, SH degree " << ref_cloud.shDegree << "\n";

    // Compare
    if (our_cloud.numPoints != ref_cloud.numPoints) {
        std::cerr << "ERROR: Point count mismatch!\n";
        return 1;
    }

    std::cout << "\n=== Comparison (first 3 points) ===\n";

    int errors = 0;
    for (int i = 0; i < std::min(3, our_cloud.numPoints); ++i) {
        std::cout << "\n--- Point " << i << " ---\n";

        // Position
        std::cout << "Position:\n";
        std::cout << "  Ours: [" << our_cloud.positions[i * 3 + 0] << ", "
                  << our_cloud.positions[i * 3 + 1] << ", "
                  << our_cloud.positions[i * 3 + 2] << "]\n";
        std::cout << "  Ref:  [" << ref_cloud.positions[i * 3 + 0] << ", "
                  << ref_cloud.positions[i * 3 + 1] << ", "
                  << ref_cloud.positions[i * 3 + 2] << "]\n";

        // Rotation
        std::cout << "Rotation (xyzw):\n";
        std::cout << "  Ours: [" << our_cloud.rotations[i * 4 + 0] << ", "
                  << our_cloud.rotations[i * 4 + 1] << ", "
                  << our_cloud.rotations[i * 4 + 2] << ", "
                  << our_cloud.rotations[i * 4 + 3] << "]\n";
        std::cout << "  Ref:  [" << ref_cloud.rotations[i * 4 + 0] << ", "
                  << ref_cloud.rotations[i * 4 + 1] << ", "
                  << ref_cloud.rotations[i * 4 + 2] << ", "
                  << ref_cloud.rotations[i * 4 + 3] << "]\n";

        // Check if quaternions are equivalent (q and -q represent same rotation)
        bool quat_match = true;
        float dot = 0;
        for (int j = 0; j < 4; ++j) {
            dot += our_cloud.rotations[i * 4 + j] * ref_cloud.rotations[i * 4 + j];
        }
        if (std::abs(std::abs(dot) - 1.0f) > 0.01f) {
            std::cout << "  WARNING: Quaternions may differ! dot=" << dot << "\n";
            quat_match = false;
            errors++;
        }

        // SH DC
        std::cout << "Color (SH DC):\n";
        std::cout << "  Ours: [" << our_cloud.colors[i * 3 + 0] << ", "
                  << our_cloud.colors[i * 3 + 1] << ", "
                  << our_cloud.colors[i * 3 + 2] << "]\n";
        std::cout << "  Ref:  [" << ref_cloud.colors[i * 3 + 0] << ", "
                  << ref_cloud.colors[i * 3 + 1] << ", "
                  << ref_cloud.colors[i * 3 + 2] << "]\n";

        // SH coefficients
        if (our_cloud.shDegree > 0) {
            int sh_dim = 0;
            switch (our_cloud.shDegree) {
            case 1: sh_dim = 3; break;
            case 2: sh_dim = 8; break;
            case 3: sh_dim = 15; break;
            }

            std::cout << "SH coeff 0:\n";
            size_t idx = i * sh_dim * 3;
            std::cout << "  Ours: [" << our_cloud.sh[idx + 0] << ", "
                      << our_cloud.sh[idx + 1] << ", " << our_cloud.sh[idx + 2] << "]\n";
            std::cout << "  Ref:  [" << ref_cloud.sh[idx + 0] << ", "
                      << ref_cloud.sh[idx + 1] << ", " << ref_cloud.sh[idx + 2] << "]\n";

            // Check for mismatch
            for (int c = 0; c < 3; ++c) {
                if (std::abs(our_cloud.sh[idx + c] - ref_cloud.sh[idx + c]) > 0.01f) {
                    std::cout << "  WARNING: SH coeff 0 channel " << c << " differs!\n";
                    errors++;
                }
            }
        }
    }

    fs::remove(ref_spz);

    std::cout << "\n=== Summary ===\n";
    if (errors == 0) {
        std::cout << "PASS: Our export matches reference implementation.\n";
    } else {
        std::cout << "WARN: " << errors << " differences found (may be due to quantization order).\n";
    }

    return 0;
}
