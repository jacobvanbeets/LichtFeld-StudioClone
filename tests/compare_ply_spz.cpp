/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Compare PLY loaded directly vs PLY roundtripped through SPZ using reference implementation

#include "load-spz.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <ply_file>\n";
        return 1;
    }

    fs::path ply_path = argv[1];
    if (!fs::exists(ply_path)) {
        std::cerr << "File not found: " << ply_path << "\n";
        return 1;
    }

    std::cout << "Loading PLY file with reference implementation: " << ply_path << "\n";

    // Load PLY using SPZ's reference implementation
    spz::UnpackOptions load_opts;
    load_opts.to = spz::CoordinateSystem::RDF;
    auto from_ply = spz::loadSplatFromPly(ply_path.string(), load_opts);

    if (from_ply.numPoints == 0) {
        std::cerr << "Failed to load PLY.\n";
        return 1;
    }

    std::cout << "Loaded " << from_ply.numPoints << " points, SH degree " << from_ply.shDegree << "\n";

    // Save to SPZ
    fs::path spz_path = fs::temp_directory_path() / "ref_test.spz";
    spz::PackOptions pack_opts;
    pack_opts.from = spz::CoordinateSystem::RDF;
    if (!spz::saveSpz(from_ply, pack_opts, spz_path.string())) {
        std::cerr << "Failed to save SPZ.\n";
        return 1;
    }

    // Load SPZ back
    auto from_spz = spz::loadSpz(spz_path.string(), load_opts);

    std::cout << "\n=== Comparison (first 3 points) ===\n";

    for (int i = 0; i < std::min(3, from_ply.numPoints); ++i) {
        std::cout << "\n--- Point " << i << " ---\n";

        // Position
        std::cout << "Position:\n";
        std::cout << "  PLY: [" << from_ply.positions[i * 3 + 0] << ", "
                  << from_ply.positions[i * 3 + 1] << ", "
                  << from_ply.positions[i * 3 + 2] << "]\n";
        std::cout << "  SPZ: [" << from_spz.positions[i * 3 + 0] << ", "
                  << from_spz.positions[i * 3 + 1] << ", "
                  << from_spz.positions[i * 3 + 2] << "]\n";

        // Rotation (xyzw in both)
        std::cout << "Rotation (xyzw):\n";
        std::cout << "  PLY: [" << from_ply.rotations[i * 4 + 0] << ", "
                  << from_ply.rotations[i * 4 + 1] << ", "
                  << from_ply.rotations[i * 4 + 2] << ", "
                  << from_ply.rotations[i * 4 + 3] << "]\n";
        std::cout << "  SPZ: [" << from_spz.rotations[i * 4 + 0] << ", "
                  << from_spz.rotations[i * 4 + 1] << ", "
                  << from_spz.rotations[i * 4 + 2] << ", "
                  << from_spz.rotations[i * 4 + 3] << "]\n";

        // SH DC
        std::cout << "Color (SH DC):\n";
        std::cout << "  PLY: [" << from_ply.colors[i * 3 + 0] << ", "
                  << from_ply.colors[i * 3 + 1] << ", "
                  << from_ply.colors[i * 3 + 2] << "]\n";
        std::cout << "  SPZ: [" << from_spz.colors[i * 3 + 0] << ", "
                  << from_spz.colors[i * 3 + 1] << ", "
                  << from_spz.colors[i * 3 + 2] << "]\n";

        // Higher order SH (first few)
        if (from_ply.shDegree > 0) {
            int sh_dim = 0;
            switch (from_ply.shDegree) {
            case 1: sh_dim = 3; break;
            case 2: sh_dim = 8; break;
            case 3: sh_dim = 15; break;
            }

            std::cout << "SH coeff 0 (of " << sh_dim << "):\n";
            size_t idx = i * sh_dim * 3;
            std::cout << "  PLY: [" << from_ply.sh[idx + 0] << ", "
                      << from_ply.sh[idx + 1] << ", " << from_ply.sh[idx + 2] << "]\n";
            std::cout << "  SPZ: [" << from_spz.sh[idx + 0] << ", "
                      << from_spz.sh[idx + 1] << ", " << from_spz.sh[idx + 2] << "]\n";

            if (sh_dim > 1) {
                std::cout << "SH coeff 1:\n";
                idx += 3;
                std::cout << "  PLY: [" << from_ply.sh[idx + 0] << ", "
                          << from_ply.sh[idx + 1] << ", " << from_ply.sh[idx + 2] << "]\n";
                std::cout << "  SPZ: [" << from_spz.sh[idx + 0] << ", "
                          << from_spz.sh[idx + 1] << ", " << from_spz.sh[idx + 2] << "]\n";
            }
        }
    }

    // Clean up
    fs::remove(spz_path);

    std::cout << "\n=== Complete ===\n";
    return 0;
}
