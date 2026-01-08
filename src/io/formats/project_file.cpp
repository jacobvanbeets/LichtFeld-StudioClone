/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef _WIN32
#define NOMINMAX
#endif

#include "project_file.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/exporter.hpp"
#include "io/formats/ply.hpp"
#include "visualizer/scene/scene.hpp"
#include <archive.h>
#include <archive_entry.h>
#include <chrono>
#include <format>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>
#include <unordered_map>

namespace lfs::io {

    using namespace lfs::vis;
    using json = nlohmann::json;

    namespace {

#ifdef _WIN32
        using ssize_t = std::ptrdiff_t;
#endif

        // Get current ISO 8601 timestamp
        std::string get_iso8601_timestamp() {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            std::tm tm;
#ifdef _WIN32
            localtime_s(&tm, &time_t);
#else
            localtime_r(&time_t, &tm);
#endif
            std::ostringstream oss;
            oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
            return oss.str();
        }

        // Convert glm::mat4 to JSON array (row-major)
        json mat4_to_json(const glm::mat4& m) {
            json j = json::array();
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    j.push_back(m[col][row]);
                }
            }
            return j;
        }

        // Convert JSON array to glm::mat4 (row-major)
        glm::mat4 json_to_mat4(const json& j) {
            glm::mat4 m(1.0f);
            if (j.is_array() && j.size() == 16) {
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        m[col][row] = j[row * 4 + col].get<float>();
                    }
                }
            }
            return m;
        }

        // Convert glm::vec3 to JSON array
        json vec3_to_json(const glm::vec3& v) {
            return json::array({v.x, v.y, v.z});
        }

        // Convert JSON array to glm::vec3
        glm::vec3 json_to_vec3(const json& j) {
            if (j.is_array() && j.size() == 3) {
                return glm::vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
            }
            return glm::vec3(0.0f);
        }

        // Serialize a scene node to JSON
        json node_to_json(const SceneNode& node) {
            json j;
            j["id"] = node.id;
            j["name"] = node.name;
            j["type"] = static_cast<int>(node.type);
            j["parent_id"] = node.parent_id;
            j["visible"] = node.visible.get();
            j["locked"] = node.locked.get();
            j["local_transform"] = mat4_to_json(node.local_transform.get());

            // Crop box data
            if (node.cropbox) {
                json cb;
                cb["min"] = vec3_to_json(node.cropbox->min);
                cb["max"] = vec3_to_json(node.cropbox->max);
                cb["inverse"] = node.cropbox->inverse;
                cb["enabled"] = node.cropbox->enabled;
                cb["color"] = vec3_to_json(node.cropbox->color);
                cb["line_width"] = node.cropbox->line_width;
                j["cropbox"] = cb;
            }

            return j;
        }

        // Serialize the entire scene to JSON
        json serialize_scene(const Scene& scene) {
            json project;

            // Metadata
            project["metadata"]["version"] = PROJECT_FILE_VERSION;
            project["metadata"]["created_with"] = "LichtFeld Studio";  // TODO: Get actual version
            project["metadata"]["creation_date"] = get_iso8601_timestamp();

            // Scene graph
            json nodes_array = json::array();
            std::unordered_map<NodeId, int> node_to_model_index;
            int model_index = 0;

            for (const auto* node : scene.getNodes()) {
                if (!node) continue;

                json node_json = node_to_json(*node);

                // Map splat nodes to model files
                if (node->type == NodeType::SPLAT && node->model && node->model->size() > 0) {
                    node_json["model_file"] = std::format("models/model_{}.ply", model_index);
                    node_to_model_index[node->id] = model_index++;
                }

                nodes_array.push_back(node_json);
            }

            project["nodes"] = nodes_array;
            project["model_count"] = model_index;

            return project;
        }

        // Save a single PLY file to memory buffer
        std::expected<std::vector<uint8_t>, std::string> save_ply_to_memory(
            const lfs::core::SplatData& splat_data) {

            // Create temporary file
            auto temp_path = std::filesystem::temp_directory_path() / std::format("temp_{}.ply", std::rand());

            // Save to temp file
            PlySaveOptions options;
            options.output_path = temp_path;
            options.binary = true;
            options.async = false;
            
            auto result = save_ply(splat_data, options);
            if (!result) {
                return std::unexpected(result.error().format());
            }

            // Read back into memory
            std::ifstream file;
            if (!lfs::core::open_file_for_read(temp_path, std::ios::binary, file)) {
                std::filesystem::remove(temp_path);
                return std::unexpected("Failed to read temporary PLY file");
            }

            file.seekg(0, std::ios::end);
            size_t size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<uint8_t> buffer(size);
            file.read(reinterpret_cast<char*>(buffer.data()), size);
            file.close();

            // Clean up temp file
            std::filesystem::remove(temp_path);

            return buffer;
        }

    } // anonymous namespace

    std::expected<void, std::string> save_project_file(
        const std::filesystem::path& path,
        const Scene& scene) {

        LOG_INFO("Saving project file: {}", lfs::core::path_to_utf8(path));

        try {
            // Serialize scene to JSON
            json project_json = serialize_scene(scene);
            std::string json_str = project_json.dump(2); // Pretty print with 2 spaces

            // Create archive
            struct archive* a = archive_write_new();
            archive_write_set_format_zip(a);

            // Use wide-character API on Windows for proper Unicode path handling
#ifdef _WIN32
            int result = archive_write_open_filename_w(a, path.wstring().c_str());
#else
            int result = archive_write_open_filename(a, path.c_str(), 10240);
#endif

            if (result != ARCHIVE_OK) {
                archive_write_free(a);
                return std::unexpected(std::format("Failed to create archive: {}",
                                                   archive_error_string(a)));
            }

            // Write project.json
            {
                struct archive_entry* entry = archive_entry_new();
                archive_entry_set_pathname(entry, "project.json");
                archive_entry_set_size(entry, json_str.size());
                archive_entry_set_filetype(entry, AE_IFREG);
                archive_entry_set_perm(entry, 0644);

                if (archive_write_header(a, entry) != ARCHIVE_OK) {
                    archive_entry_free(entry);
                    archive_write_free(a);
                    return std::unexpected("Failed to write project.json header");
                }

                archive_write_data(a, json_str.data(), json_str.size());
                archive_entry_free(entry);
            }

            // Write embedded PLY files
            int model_index = 0;
            for (const auto* node : scene.getNodes()) {
                if (!node || node->type != NodeType::SPLAT || !node->model || node->model->size() == 0) {
                    continue;
                }

                LOG_DEBUG("Embedding model '{}' as model_{}.ply", node->name, model_index);

                // Save PLY to memory
                auto ply_buffer = save_ply_to_memory(*node->model);
                if (!ply_buffer) {
                    archive_write_free(a);
                    return std::unexpected(std::format("Failed to save model '{}': {}",
                                                       node->name, ply_buffer.error()));
                }

                // Write to archive
                std::string filename = std::format("models/model_{}.ply", model_index);
                struct archive_entry* entry = archive_entry_new();
                archive_entry_set_pathname(entry, filename.c_str());
                archive_entry_set_size(entry, ply_buffer->size());
                archive_entry_set_filetype(entry, AE_IFREG);
                archive_entry_set_perm(entry, 0644);

                if (archive_write_header(a, entry) != ARCHIVE_OK) {
                    archive_entry_free(entry);
                    archive_write_free(a);
                    return std::unexpected(std::format("Failed to write header for {}", filename));
                }

                archive_write_data(a, ply_buffer->data(), ply_buffer->size());
                archive_entry_free(entry);

                model_index++;
            }

            archive_write_close(a);
            archive_write_free(a);

            LOG_INFO("Project saved successfully: {} nodes, {} models",
                     project_json["nodes"].size(), model_index);
            return {};

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to save project: {}", e.what()));
        }
    }

    std::expected<void, std::string> load_project_file(
        const std::filesystem::path& path,
        Scene& scene) {

        LOG_INFO("Loading project file: {}", lfs::core::path_to_utf8(path));

        try {
            // Open archive
            struct archive* a = archive_read_new();
            archive_read_support_format_zip(a);
            archive_read_support_filter_all(a);

#ifdef _WIN32
            int result = archive_read_open_filename_w(a, path.wstring().c_str(), 10240);
#else
            int result = archive_read_open_filename(a, path.c_str(), 10240);
#endif

            if (result != ARCHIVE_OK) {
                archive_read_free(a);
                return std::unexpected(std::format("Failed to open archive: {}",
                                                   archive_error_string(a)));
            }

            // Read all files from archive
            struct archive_entry* entry;
            std::string project_json_str;
            std::unordered_map<std::string, std::vector<uint8_t>> model_files;

            while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
                std::string filename = archive_entry_pathname(entry);
                size_t size = archive_entry_size(entry);

                LOG_DEBUG("Reading {} ({} bytes)", filename, size);

                std::vector<uint8_t> data(size);
                ssize_t r = archive_read_data(a, data.data(), size);

                if (r != static_cast<ssize_t>(size)) {
                    archive_read_free(a);
                    return std::unexpected(std::format("Failed to read {} from archive", filename));
                }

                if (filename == "project.json") {
                    project_json_str = std::string(data.begin(), data.end());
                } else if (filename.starts_with("models/") && filename.ends_with(".ply")) {
                    model_files[filename] = std::move(data);
                }
            }

            archive_read_free(a);

            if (project_json_str.empty()) {
                return std::unexpected("Missing project.json in archive");
            }

            // Parse project JSON
            json project_json = json::parse(project_json_str);

            // Validate version
            std::string version = project_json["metadata"]["version"];
            if (version != PROJECT_FILE_VERSION) {
                LOG_WARN("Project file version mismatch: {} (expected {})", version, PROJECT_FILE_VERSION);
            }

            // Clear scene
            scene.clear();

            // Load models from memory into temporary files, then load them
            std::unordered_map<std::string, std::unique_ptr<lfs::core::SplatData>> loaded_models;

            for (const auto& [filename, data] : model_files) {
                // Write to temp file
                auto temp_path = std::filesystem::temp_directory_path() / std::format("temp_load_{}.ply", std::rand());
                
                std::ofstream temp_file;
                if (!lfs::core::open_file_for_write(temp_path, std::ios::binary, temp_file)) {
                    return std::unexpected(std::format("Failed to create temp file for {}", filename));
                }
                
                temp_file.write(reinterpret_cast<const char*>(data.data()), data.size());
                temp_file.close();

                // Load PLY
                auto splat_result = load_ply(temp_path);
                std::filesystem::remove(temp_path);

                if (!splat_result) {
                    return std::unexpected(std::format("Failed to load {}: {}",
                                                       filename, splat_result.error()));
                }

                loaded_models[filename] = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
                LOG_DEBUG("Loaded model: {} ({} gaussians)", filename, loaded_models[filename]->size());
            }

            // Reconstruct scene graph
            std::unordered_map<NodeId, NodeId> old_to_new_id;

            for (const auto& node_json : project_json["nodes"]) {
                NodeId old_id = node_json["id"];
                std::string name = node_json["name"];
                NodeType type = static_cast<NodeType>(node_json["type"].get<int>());
                NodeId old_parent_id = node_json["parent_id"];
                
                NodeId new_id = NULL_NODE;

                // Create node based on type
                if (type == NodeType::SPLAT && node_json.contains("model_file")) {
                    std::string model_file = node_json["model_file"];
                    if (loaded_models.contains(model_file)) {
                        new_id = scene.addSplat(name, std::move(loaded_models[model_file]), NULL_NODE);
                    }
                } else if (type == NodeType::GROUP) {
                    new_id = scene.addGroup(name, NULL_NODE);
                } else if (type == NodeType::CROPBOX) {
                    // Cropbox will be created when we set its data later
                    continue;
                }

                if (new_id == NULL_NODE) {
                    continue;
                }

                old_to_new_id[old_id] = new_id;

                // Set node properties
                auto* node = scene.getMutableNode(name);
                if (node) {
                    node->visible = node_json["visible"].get<bool>();
                    node->locked = node_json["locked"].get<bool>();
                    node->local_transform = json_to_mat4(node_json["local_transform"]);
                    node->transform_dirty = true;

                    // Create and set cropbox if present
                    if (node_json.contains("cropbox") && type == NodeType::SPLAT) {
                        NodeId cropbox_id = scene.getOrCreateCropBoxForSplat(node->id);
                        if (cropbox_id != NULL_NODE) {
                            const auto& cb_json = node_json["cropbox"];
                            CropBoxData cb;
                            cb.min = json_to_vec3(cb_json["min"]);
                            cb.max = json_to_vec3(cb_json["max"]);
                            cb.inverse = cb_json["inverse"];
                            cb.enabled = cb_json["enabled"];
                            cb.color = json_to_vec3(cb_json["color"]);
                            cb.line_width = cb_json["line_width"];
                            scene.setCropBoxData(cropbox_id, cb);
                        }
                    }
                }
            }

            // Second pass: fix parent relationships
            for (const auto& node_json : project_json["nodes"]) {
                NodeId old_id = node_json["id"];
                NodeId old_parent_id = node_json["parent_id"];

                if (old_parent_id != NULL_NODE && old_to_new_id.contains(old_id) && old_to_new_id.contains(old_parent_id)) {
                    NodeId new_id = old_to_new_id[old_id];
                    NodeId new_parent_id = old_to_new_id[old_parent_id];
                    scene.reparent(new_id, new_parent_id);
                }
            }

            // Third pass: emit PLYAdded events for UI update
            using namespace lfs::core::events;
            for (const auto& node_json : project_json["nodes"]) {
                std::string name = node_json["name"];
                NodeType type = static_cast<NodeType>(node_json["type"].get<int>());
                
                const auto* node = scene.getNode(name);
                if (!node)
                    continue;

                std::string parent_name;
                if (node->parent_id != NULL_NODE) {
                    if (const auto* p = scene.getNodeById(node->parent_id)) {
                        parent_name = p->name;
                    }
                }

                state::PLYAdded{
                    .name = name,
                    .node_gaussians = node->gaussian_count,
                    .total_gaussians = scene.getTotalGaussianCount(),
                    .is_visible = node->visible.get(),
                    .parent_name = parent_name,
                    .is_group = (type == NodeType::GROUP),
                    .node_type = static_cast<int>(type)}
                    .emit();
            }

            LOG_INFO("Project loaded successfully: {} nodes", project_json["nodes"].size());
            return {};

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to load project: {}", e.what()));
        }
    }

} // namespace lfs::io
