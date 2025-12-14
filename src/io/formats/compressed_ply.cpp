/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "compressed_ply.hpp"
#include "core_new/logger.hpp"
#include "core_new/tensor.hpp"
#include "kernels/morton_encoding_new.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

#ifdef _WIN32
#define NOMINMAX  // Prevent Windows min/max macros
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <tbb/parallel_for.h>

namespace lfs::io {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    namespace {

        constexpr size_t CHUNK_SIZE = 256;
        constexpr float SH_C0 = 0.28209479177387814f;

        struct CompressedPlyLayout {
            size_t chunk_count = 0;
            size_t vertex_count = 0;
            size_t sh_count = 0;        // Number of SH coefficients (0, 9, 24, or 45)
            size_t header_size = 0;
            size_t chunk_data_offset = 0;
            size_t vertex_data_offset = 0;
            size_t sh_data_offset = 0;
        };

        struct MMappedFile {
            void* data = nullptr;
            size_t size = 0;
#ifdef _WIN32
            HANDLE file_handle = INVALID_HANDLE_VALUE;
            HANDLE mapping_handle = INVALID_HANDLE_VALUE;

            ~MMappedFile() {
                if (data) UnmapViewOfFile(data);
                if (mapping_handle != INVALID_HANDLE_VALUE) CloseHandle(mapping_handle);
                if (file_handle != INVALID_HANDLE_VALUE) CloseHandle(file_handle);
            }

            [[nodiscard]] bool map(const std::filesystem::path& filepath) {
                auto wide_path = filepath.wstring();
                file_handle = CreateFileW(wide_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                          nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
                if (file_handle == INVALID_HANDLE_VALUE) return false;

                LARGE_INTEGER file_size_li;
                if (!GetFileSizeEx(file_handle, &file_size_li)) return false;
                size = static_cast<size_t>(file_size_li.QuadPart);

                mapping_handle = CreateFileMappingW(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
                if (!mapping_handle) return false;

                data = MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
                return data != nullptr;
            }
#else
            int fd = -1;

            ~MMappedFile() {
                if (data && data != MAP_FAILED) munmap(data, size);
                if (fd >= 0) close(fd);
            }

            [[nodiscard]] bool map(const std::filesystem::path& filepath) {
                fd = open(filepath.c_str(), O_RDONLY);
                if (fd < 0) return false;

                struct stat st {};
                if (fstat(fd, &st) < 0) return false;
                size = st.st_size;

                data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
                return data != MAP_FAILED;
            }
#endif
        };

        // Unpack 11-10-11 bit normalized value
        inline void unpack_111011(uint32_t value, float& x, float& y, float& z) {
            constexpr float scale_11 = 1.0f / 2047.0f;  // (2^11 - 1)
            constexpr float scale_10 = 1.0f / 1023.0f;  // (2^10 - 1)
            x = static_cast<float>((value >> 21) & 0x7FF) * scale_11;
            y = static_cast<float>((value >> 11) & 0x3FF) * scale_10;
            z = static_cast<float>(value & 0x7FF) * scale_11;
        }

        // Unpack 8-8-8-8 bit normalized value
        inline void unpack_8888(uint32_t value, float& r, float& g, float& b, float& a) {
            constexpr float scale = 1.0f / 255.0f;
            r = static_cast<float>((value >> 24) & 0xFF) * scale;
            g = static_cast<float>((value >> 16) & 0xFF) * scale;
            b = static_cast<float>((value >> 8) & 0xFF) * scale;
            a = static_cast<float>(value & 0xFF) * scale;
        }

        // Unpack rotation from 10-10-10-2 bit format
        inline void unpack_rotation(uint32_t value, float& qx, float& qy, float& qz, float& qw) {
            constexpr float SQRT2 = 1.41421356f;  // std::sqrt not constexpr on MSVC
            constexpr float norm = 1.0f / (SQRT2 * 0.5f);
            constexpr float scale = 1.0f / 1023.0f;

            const float a = (static_cast<float>((value >> 20) & 0x3FF) * scale - 0.5f) * norm;
            const float b = (static_cast<float>((value >> 10) & 0x3FF) * scale - 0.5f) * norm;
            const float c = (static_cast<float>(value & 0x3FF) * scale - 0.5f) * norm;
            const float m = std::sqrt(std::max(0.0f, 1.0f - (a * a + b * b + c * c)));

            const int which = (value >> 30) & 0x3;
            switch (which) {
            case 0: qw = m; qx = a; qy = b; qz = c; break;
            case 1: qw = a; qx = m; qy = b; qz = c; break;
            case 2: qw = a; qx = b; qy = m; qz = c; break;
            default: qw = a; qx = b; qy = c; qz = m; break;
            }
        }

        std::expected<CompressedPlyLayout, std::string>
        parse_compressed_header(const char* data, size_t file_size) {
            if (file_size < 10 || std::strncmp(data, "ply", 3) != 0) {
                return std::unexpected("Invalid PLY file");
            }

            CompressedPlyLayout layout{};
            const char* ptr = data;
            const char* end = data + file_size;
            bool found_chunk = false, found_vertex = false;
            [[maybe_unused]] bool found_sh = false;

            // Skip to first newline
            while (ptr < end && *ptr != '\n') ++ptr;
            ++ptr;

            while (ptr < end) {
                const char* line_start = ptr;
                while (ptr < end && *ptr != '\n' && *ptr != '\r') ++ptr;
                const size_t line_len = ptr - line_start;
                while (ptr < end && (*ptr == '\n' || *ptr == '\r')) ++ptr;

                if (line_len >= 10 && std::strncmp(line_start, "end_header", 10) == 0) {
                    layout.header_size = ptr - data;
                    break;
                }

                if (line_len >= 14 && std::strncmp(line_start, "element chunk ", 14) == 0) {
                    layout.chunk_count = std::strtoull(line_start + 14, nullptr, 10);
                    found_chunk = true;
                } else if (line_len >= 15 && std::strncmp(line_start, "element vertex ", 15) == 0) {
                    layout.vertex_count = std::strtoull(line_start + 15, nullptr, 10);
                    found_vertex = true;
                } else if (line_len >= 11 && std::strncmp(line_start, "element sh ", 11) == 0) {
                    layout.sh_count = std::strtoull(line_start + 11, nullptr, 10);
                    found_sh = true;
                }
            }

            if (!found_chunk || !found_vertex) {
                return std::unexpected("Not a compressed PLY file");
            }

            // Calculate data offsets
            // Chunk data: 18 floats per chunk (min/max x,y,z, scale x,y,z, color r,g,b)
            layout.chunk_data_offset = layout.header_size;
            layout.vertex_data_offset = layout.chunk_data_offset + layout.chunk_count * 18 * sizeof(float);
            layout.sh_data_offset = layout.vertex_data_offset + layout.vertex_count * 4 * sizeof(uint32_t);

            return layout;
        }

    } // anonymous namespace

    bool is_compressed_ply(const std::filesystem::path& filepath) {
        std::ifstream file(filepath);
        if (!file) return false;

        std::string line;
        bool has_chunk = false, has_packed_position = false;

        for (int i = 0; i < 100 && std::getline(file, line); ++i) {
            if (line.find("element chunk ") == 0) has_chunk = true;
            if (line.find("property uint packed_position") != std::string::npos) has_packed_position = true;
            if (line.find("end_header") == 0) break;
        }

        return has_chunk && has_packed_position;
    }

    std::expected<SplatData, std::string>
    load_compressed_ply(const std::filesystem::path& filepath) {
        LOG_TIMER("Compressed PLY Loading");

        if (!std::filesystem::exists(filepath)) {
            return std::unexpected("File does not exist: " + filepath.string());
        }

        MMappedFile mapped;
        if (!mapped.map(filepath)) {
            return std::unexpected("Failed to memory map file: " + filepath.string());
        }

        const char* data = static_cast<const char*>(mapped.data);
        auto layout_result = parse_compressed_header(data, mapped.size);
        if (!layout_result) {
            return std::unexpected(layout_result.error());
        }

        const auto& layout = *layout_result;
        const size_t N = layout.vertex_count;

        LOG_INFO("Loading compressed PLY: {} splats, {} chunks", N, layout.chunk_count);

        // Read chunk data (18 floats per chunk)
        const float* chunk_data = reinterpret_cast<const float*>(data + layout.chunk_data_offset);

        // Read vertex data (4 uint32 per vertex)
        const uint32_t* vertex_data = reinterpret_cast<const uint32_t*>(data + layout.vertex_data_offset);

        // Read optional SH data
        const uint8_t* sh_data = nullptr;
        int sh_coeffs = 0;
        if (layout.sh_count > 0 && layout.sh_data_offset < mapped.size) {
            sh_data = reinterpret_cast<const uint8_t*>(data + layout.sh_data_offset);
            // Determine SH coefficient count from remaining file size
            const size_t remaining = mapped.size - layout.sh_data_offset;
            sh_coeffs = remaining / N;
            if (sh_coeffs != 9 && sh_coeffs != 24 && sh_coeffs != 45) {
                sh_coeffs = 0;  // Invalid SH count
            }
            LOG_DEBUG("Found {} SH coefficients per splat", sh_coeffs);
        }

        // Allocate host buffers
        std::vector<float> host_means(N * 3);
        std::vector<float> host_scales(N * 3);
        std::vector<float> host_rotations(N * 4);
        std::vector<float> host_opacity(N);
        std::vector<float> host_sh0(N * 3);

        int sh_degree = 0;
        int shN_coeffs = 0;
        if (sh_coeffs == 9) { sh_degree = 1; shN_coeffs = 3; }
        else if (sh_coeffs == 24) { sh_degree = 2; shN_coeffs = 8; }
        else if (sh_coeffs == 45) { sh_degree = 3; shN_coeffs = 15; }

        std::vector<float> host_shN(N * shN_coeffs * 3);

        // Decompress in parallel
        tbb::parallel_for(tbb::blocked_range<size_t>(0, N, 1024),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    const size_t ci = i / CHUNK_SIZE;  // Chunk index
                    const size_t chunk_base = ci * 18;

                    // Chunk bounds
                    const float min_x = chunk_data[chunk_base + 0];
                    const float min_y = chunk_data[chunk_base + 1];
                    const float min_z = chunk_data[chunk_base + 2];
                    const float max_x = chunk_data[chunk_base + 3];
                    const float max_y = chunk_data[chunk_base + 4];
                    const float max_z = chunk_data[chunk_base + 5];
                    const float min_sx = chunk_data[chunk_base + 6];
                    const float min_sy = chunk_data[chunk_base + 7];
                    const float min_sz = chunk_data[chunk_base + 8];
                    const float max_sx = chunk_data[chunk_base + 9];
                    const float max_sy = chunk_data[chunk_base + 10];
                    const float max_sz = chunk_data[chunk_base + 11];
                    const float min_r = chunk_data[chunk_base + 12];
                    const float min_g = chunk_data[chunk_base + 13];
                    const float min_b = chunk_data[chunk_base + 14];
                    const float max_r = chunk_data[chunk_base + 15];
                    const float max_g = chunk_data[chunk_base + 16];
                    const float max_b = chunk_data[chunk_base + 17];

                    // Unpack vertex data
                    const size_t vi = i * 4;
                    const uint32_t packed_pos = vertex_data[vi + 0];
                    const uint32_t packed_rot = vertex_data[vi + 1];
                    const uint32_t packed_scale = vertex_data[vi + 2];
                    const uint32_t packed_color = vertex_data[vi + 3];

                    // Position
                    float px, py, pz;
                    unpack_111011(packed_pos, px, py, pz);
                    host_means[i * 3 + 0] = min_x + px * (max_x - min_x);
                    host_means[i * 3 + 1] = min_y + py * (max_y - min_y);
                    host_means[i * 3 + 2] = min_z + pz * (max_z - min_z);

                    // Rotation (stored as w, x, y, z)
                    float qx, qy, qz, qw;
                    unpack_rotation(packed_rot, qx, qy, qz, qw);
                    host_rotations[i * 4 + 0] = qw;
                    host_rotations[i * 4 + 1] = qx;
                    host_rotations[i * 4 + 2] = qy;
                    host_rotations[i * 4 + 3] = qz;

                    // Scale (lerp in log space)
                    float sx, sy, sz;
                    unpack_111011(packed_scale, sx, sy, sz);
                    host_scales[i * 3 + 0] = min_sx + sx * (max_sx - min_sx);
                    host_scales[i * 3 + 1] = min_sy + sy * (max_sy - min_sy);
                    host_scales[i * 3 + 2] = min_sz + sz * (max_sz - min_sz);

                    // Color and opacity
                    float cr, cg, cb, ca;
                    unpack_8888(packed_color, cr, cg, cb, ca);
                    cr = min_r + cr * (max_r - min_r);
                    cg = min_g + cg * (max_g - min_g);
                    cb = min_b + cb * (max_b - min_b);

                    // Convert RGB [0,1] to f_dc (SH coefficient)
                    host_sh0[i * 3 + 0] = (cr - 0.5f) / SH_C0;
                    host_sh0[i * 3 + 1] = (cg - 0.5f) / SH_C0;
                    host_sh0[i * 3 + 2] = (cb - 0.5f) / SH_C0;

                    // Convert opacity [0,1] to logit
                    ca = std::clamp(ca, 1e-5f, 1.0f - 1e-5f);
                    host_opacity[i] = -std::log(1.0f / ca - 1.0f);

                    // Decompress SH coefficients
                    if (sh_data && shN_coeffs > 0) {
                        const size_t sh_base = i * sh_coeffs;
                        for (int j = 0; j < shN_coeffs * 3; ++j) {
                            const uint8_t v = sh_data[sh_base + j];
                            float n;
                            if (v == 0) n = 0.0f;
                            else if (v == 255) n = 1.0f;
                            else n = (v + 0.5f) / 256.0f;
                            host_shN[i * shN_coeffs * 3 + j] = (n - 0.5f) * 8.0f;
                        }
                    }
                }
            });

        // Create tensors
        Tensor means = Tensor::from_vector(host_means, {N, 3}, Device::CUDA);
        Tensor scales = Tensor::from_vector(host_scales, {N, 3}, Device::CUDA);
        Tensor rotations = Tensor::from_vector(host_rotations, {N, 4}, Device::CUDA);
        Tensor opacity = Tensor::from_vector(host_opacity, {N, 1}, Device::CUDA);
        Tensor sh0 = Tensor::from_vector(host_sh0, {N, 1, 3}, Device::CUDA);

        Tensor shN;
        if (shN_coeffs > 0) {
            shN = Tensor::from_vector(host_shN, {N, static_cast<size_t>(shN_coeffs), 3}, Device::CUDA);
        } else {
            shN = Tensor::zeros({N, 0, 3}, Device::CUDA);
        }

        LOG_INFO("Loaded compressed PLY: {} splats, SH degree {}", N, sh_degree);

        return SplatData(
            sh_degree,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scales),
            std::move(rotations),
            std::move(opacity),
            0.5f);
    }

    namespace {

        // Pack 11-10-11 bit normalized values into uint32
        inline uint32_t pack_111011(float x, float y, float z) {
            const uint32_t ix = static_cast<uint32_t>(std::clamp(x, 0.0f, 1.0f) * 2047.0f);
            const uint32_t iy = static_cast<uint32_t>(std::clamp(y, 0.0f, 1.0f) * 1023.0f);
            const uint32_t iz = static_cast<uint32_t>(std::clamp(z, 0.0f, 1.0f) * 2047.0f);
            return (ix << 21) | (iy << 11) | iz;
        }

        // Pack 8-8-8-8 bit normalized values into uint32
        inline uint32_t pack_8888(float r, float g, float b, float a) {
            const uint32_t ir = static_cast<uint32_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f);
            const uint32_t ig = static_cast<uint32_t>(std::clamp(g, 0.0f, 1.0f) * 255.0f);
            const uint32_t ib = static_cast<uint32_t>(std::clamp(b, 0.0f, 1.0f) * 255.0f);
            const uint32_t ia = static_cast<uint32_t>(std::clamp(a, 0.0f, 1.0f) * 255.0f);
            return (ir << 24) | (ig << 16) | (ib << 8) | ia;
        }

        // Pack rotation into 10-10-10-2 bit format
        inline uint32_t pack_rotation(float qw, float qx, float qy, float qz) {
            // Normalize
            float len = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            if (len > 1e-10f) {
                qw /= len; qx /= len; qy /= len; qz /= len;
            } else {
                qw = 1.0f; qx = 0.0f; qy = 0.0f; qz = 0.0f;
            }

            // Find largest component
            float abs_w = std::abs(qw), abs_x = std::abs(qx);
            float abs_y = std::abs(qy), abs_z = std::abs(qz);
            int which = 0;
            float max_val = abs_w;
            if (abs_x > max_val) { max_val = abs_x; which = 1; }
            if (abs_y > max_val) { max_val = abs_y; which = 2; }
            if (abs_z > max_val) { which = 3; }

            // Ensure largest is positive
            float sign = 1.0f;
            switch (which) {
                case 0: if (qw < 0) sign = -1.0f; break;
                case 1: if (qx < 0) sign = -1.0f; break;
                case 2: if (qy < 0) sign = -1.0f; break;
                case 3: if (qz < 0) sign = -1.0f; break;
            }
            qw *= sign; qx *= sign; qy *= sign; qz *= sign;

            // Get the 3 smaller components
            float a, b, c;
            switch (which) {
                case 0: a = qx; b = qy; c = qz; break;
                case 1: a = qw; b = qy; c = qz; break;
                case 2: a = qw; b = qx; c = qz; break;
                default: a = qw; b = qx; c = qy; break;
            }

            // Scale by sqrt(2)/2 and map to [0,1]
            constexpr float SQRT2 = 1.41421356f;  // std::sqrt not constexpr on MSVC
            constexpr float inv_norm = SQRT2 * 0.5f;
            a = a * inv_norm * 0.5f + 0.5f;
            b = b * inv_norm * 0.5f + 0.5f;
            c = c * inv_norm * 0.5f + 0.5f;

            const uint32_t ia = static_cast<uint32_t>(std::clamp(a, 0.0f, 1.0f) * 1023.0f);
            const uint32_t ib = static_cast<uint32_t>(std::clamp(b, 0.0f, 1.0f) * 1023.0f);
            const uint32_t ic = static_cast<uint32_t>(std::clamp(c, 0.0f, 1.0f) * 1023.0f);

            return (static_cast<uint32_t>(which) << 30) | (ia << 20) | (ib << 10) | ic;
        }

    } // anonymous namespace

    std::expected<void, std::string>
    write_compressed_ply(const SplatData& splat_data,
                         const CompressedPlyWriteOptions& options) {
        LOG_TIMER("Compressed PLY Export");

        try {
            const int64_t N = splat_data.size();
            if (N == 0) {
                return std::unexpected("No splats to export");
            }

            // Compute Morton order on GPU (much faster than CPU)
            LOG_DEBUG("Computing GPU Morton order for {} splats", N);
            Tensor morton_codes = morton_encode_new(splat_data.means());
            Tensor sort_indices = morton_sort_indices_new(morton_codes);

            // Copy sorted indices to CPU
            auto sort_indices_cpu = sort_indices.cpu();
            const int64_t* sorted_idx = sort_indices_cpu.ptr<int64_t>();

            // Get CPU data for packing
            auto means_cpu = splat_data.means().cpu();
            auto scales_cpu = splat_data.scaling_raw().cpu();  // Raw log-space scales
            auto rotations_cpu = splat_data.get_rotation().cpu();  // Normalized
            auto opacity_cpu = splat_data.get_opacity().cpu();  // Sigmoid applied
            auto sh0_cpu = splat_data.sh0().cpu();

            const float* means = means_cpu.ptr<float>();
            const float* scales = scales_cpu.ptr<float>();
            const float* rotations = rotations_cpu.ptr<float>();
            const float* opacity = opacity_cpu.ptr<float>();
            const float* sh0 = sh0_cpu.ptr<float>();

            // Calculate chunks
            const size_t chunk_count = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;
            LOG_DEBUG("Writing {} chunks", chunk_count);

            // Prepare chunk bounds (18 floats per chunk)
            std::vector<float> chunk_data(chunk_count * 18);

            // Prepare vertex data (4 uint32 per vertex)
            std::vector<uint32_t> vertex_data(N * 4);

            // Get SH data for higher-order coefficients
            Tensor shN_cpu;
            const float* shN = nullptr;
            int sh_degree = splat_data.get_max_sh_degree();
            int shN_coeffs = 0;
            if (sh_degree == 1) shN_coeffs = 3;
            else if (sh_degree == 2) shN_coeffs = 8;
            else if (sh_degree == 3) shN_coeffs = 15;

            std::vector<uint8_t> sh_data;
            if (options.include_sh && shN_coeffs > 0 && splat_data.shN().is_valid()) {
                shN_cpu = splat_data.shN().cpu();
                shN = shN_cpu.ptr<float>();
                sh_data.resize(N * shN_coeffs * 3);
            }

            // Process each chunk
            tbb::parallel_for(tbb::blocked_range<size_t>(0, chunk_count),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t ci = range.begin(); ci < range.end(); ++ci) {
                        const size_t chunk_start = ci * CHUNK_SIZE;
                        const size_t chunk_end = std::min(chunk_start + CHUNK_SIZE, static_cast<size_t>(N));

                        // Find min/max for this chunk
                        float min_pos[3], max_pos[3];
                        float min_scale[3], max_scale[3];
                        float min_color[3], max_color[3];

                        for (int j = 0; j < 3; ++j) {
                            min_pos[j] = std::numeric_limits<float>::max();
                            max_pos[j] = std::numeric_limits<float>::lowest();
                            min_scale[j] = std::numeric_limits<float>::max();
                            max_scale[j] = std::numeric_limits<float>::lowest();
                            min_color[j] = std::numeric_limits<float>::max();
                            max_color[j] = std::numeric_limits<float>::lowest();
                        }

                        // First pass: compute bounds
                        for (size_t i = chunk_start; i < chunk_end; ++i) {
                            const size_t orig_idx = static_cast<size_t>(sorted_idx[i]);

                            for (int j = 0; j < 3; ++j) {
                                min_pos[j] = std::min(min_pos[j], means[orig_idx * 3 + j]);
                                max_pos[j] = std::max(max_pos[j], means[orig_idx * 3 + j]);
                                min_scale[j] = std::min(min_scale[j], scales[orig_idx * 3 + j]);
                                max_scale[j] = std::max(max_scale[j], scales[orig_idx * 3 + j]);
                            }

                            // Use raw f_dc (SH0) values - viewer does the conversion
                            for (int j = 0; j < 3; ++j) {
                                float f_dc = sh0[orig_idx * 3 + j];
                                min_color[j] = std::min(min_color[j], f_dc);
                                max_color[j] = std::max(max_color[j], f_dc);
                            }
                        }

                        // Add small epsilon to prevent division by zero
                        for (int j = 0; j < 3; ++j) {
                            if (max_pos[j] - min_pos[j] < 1e-10f) max_pos[j] = min_pos[j] + 1e-10f;
                            if (max_scale[j] - min_scale[j] < 1e-10f) max_scale[j] = min_scale[j] + 1e-10f;
                            if (max_color[j] - min_color[j] < 1e-10f) max_color[j] = min_color[j] + 1e-10f;
                        }

                        // Store chunk bounds
                        const size_t cb = ci * 18;
                        for (int j = 0; j < 3; ++j) {
                            chunk_data[cb + j] = min_pos[j];
                            chunk_data[cb + 3 + j] = max_pos[j];
                            chunk_data[cb + 6 + j] = min_scale[j];
                            chunk_data[cb + 9 + j] = max_scale[j];
                            chunk_data[cb + 12 + j] = min_color[j];
                            chunk_data[cb + 15 + j] = max_color[j];
                        }

                        // Second pass: pack vertices
                        for (size_t i = chunk_start; i < chunk_end; ++i) {
                            const size_t orig_idx = static_cast<size_t>(sorted_idx[i]);

                            // Normalize position
                            float px = (means[orig_idx * 3 + 0] - min_pos[0]) / (max_pos[0] - min_pos[0]);
                            float py = (means[orig_idx * 3 + 1] - min_pos[1]) / (max_pos[1] - min_pos[1]);
                            float pz = (means[orig_idx * 3 + 2] - min_pos[2]) / (max_pos[2] - min_pos[2]);

                            // Normalize scale
                            float sx = (scales[orig_idx * 3 + 0] - min_scale[0]) / (max_scale[0] - min_scale[0]);
                            float sy = (scales[orig_idx * 3 + 1] - min_scale[1]) / (max_scale[1] - min_scale[1]);
                            float sz = (scales[orig_idx * 3 + 2] - min_scale[2]) / (max_scale[2] - min_scale[2]);

                            // Use raw f_dc values normalized to chunk range
                            float cr = (sh0[orig_idx * 3 + 0] - min_color[0]) / (max_color[0] - min_color[0]);
                            float cg = (sh0[orig_idx * 3 + 1] - min_color[1]) / (max_color[1] - min_color[1]);
                            float cb = (sh0[orig_idx * 3 + 2] - min_color[2]) / (max_color[2] - min_color[2]);

                            // Get opacity (already sigmoid)
                            float op = std::clamp(opacity[orig_idx], 0.0f, 1.0f);

                            // Get rotation (w, x, y, z format in SplatData)
                            float qw = rotations[orig_idx * 4 + 0];
                            float qx = rotations[orig_idx * 4 + 1];
                            float qy = rotations[orig_idx * 4 + 2];
                            float qz = rotations[orig_idx * 4 + 3];

                            // Pack into vertex data
                            const size_t vi = i * 4;
                            vertex_data[vi + 0] = pack_111011(px, py, pz);
                            vertex_data[vi + 1] = pack_rotation(qw, qx, qy, qz);
                            vertex_data[vi + 2] = pack_111011(sx, sy, sz);
                            vertex_data[vi + 3] = pack_8888(cr, cg, cb, op);

                            // Pack SH coefficients if present
                            if (shN && !sh_data.empty()) {
                                const size_t sh_base = i * shN_coeffs * 3;
                                for (int j = 0; j < shN_coeffs * 3; ++j) {
                                    float val = shN[orig_idx * shN_coeffs * 3 + j];
                                    // Map from [-4, 4] to [0, 1] then to [0, 255]
                                    float n = val / 8.0f + 0.5f;
                                    n = std::clamp(n, 0.0f, 1.0f);
                                    sh_data[sh_base + j] = static_cast<uint8_t>(n * 255.0f);
                                }
                            }
                        }
                    }
                });

            // Write PLY file
            std::ofstream file(options.output_path, std::ios::binary);
            if (!file) {
                return std::unexpected("Failed to open file: " + options.output_path.string());
            }

            // Write header
            std::ostringstream header;
            header << "ply\n";
            header << "format binary_little_endian 1.0\n";
            header << "element chunk " << chunk_count << "\n";
            header << "property float min_x\n";
            header << "property float min_y\n";
            header << "property float min_z\n";
            header << "property float max_x\n";
            header << "property float max_y\n";
            header << "property float max_z\n";
            header << "property float min_scale_x\n";
            header << "property float min_scale_y\n";
            header << "property float min_scale_z\n";
            header << "property float max_scale_x\n";
            header << "property float max_scale_y\n";
            header << "property float max_scale_z\n";
            header << "property float min_r\n";
            header << "property float min_g\n";
            header << "property float min_b\n";
            header << "property float max_r\n";
            header << "property float max_g\n";
            header << "property float max_b\n";
            header << "element vertex " << N << "\n";
            header << "property uint packed_position\n";
            header << "property uint packed_rotation\n";
            header << "property uint packed_scale\n";
            header << "property uint packed_color\n";

            if (!sh_data.empty()) {
                // Element count is number of splats, not coefficient count!
                header << "element sh " << N << "\n";
                for (int j = 0; j < shN_coeffs * 3; ++j) {
                    header << "property uchar f_rest_" << j << "\n";
                }
            }

            header << "end_header\n";

            std::string header_str = header.str();
            file.write(header_str.c_str(), header_str.size());

            // Write chunk data
            file.write(reinterpret_cast<const char*>(chunk_data.data()),
                       chunk_data.size() * sizeof(float));

            // Write vertex data
            file.write(reinterpret_cast<const char*>(vertex_data.data()),
                       vertex_data.size() * sizeof(uint32_t));

            // Write SH data
            if (!sh_data.empty()) {
                file.write(reinterpret_cast<const char*>(sh_data.data()), sh_data.size());
            }

            if (!file.good()) {
                return std::unexpected("Failed to write compressed PLY file");
            }

            LOG_INFO("Exported compressed PLY: {} splats, {} chunks to {}",
                     N, chunk_count, options.output_path.string());

            return {};

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Export failed: {}", e.what()));
        }
    }

} // namespace lfs::io
