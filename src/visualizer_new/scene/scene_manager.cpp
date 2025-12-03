/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene/scene_manager.hpp"
#include "command/command_history.hpp"
#include "command/commands/crop_command.hpp"
#include "core_new/logger.hpp"
#include "core_new/splat_data_export.hpp"
#include "core_new/splat_data_transform.hpp"
#include "geometry_new/bounding_box.hpp"
#include "geometry_new/euclidean_transform.hpp"
#include "gui/panels/gizmo_toolbar.hpp"
#include "loader_new/loader.hpp"
#include "rendering/rendering_manager.hpp"
#include "training/training_manager.hpp"
#include "training_new/training_setup.hpp"
#include <algorithm>
#include <format>
#include <glm/gtc/quaternion.hpp>
#include <set>
#include <stdexcept>

namespace lfs::vis {

    using namespace lfs::core::events;

    SceneManager::SceneManager() {
        setupEventHandlers();
        LOG_DEBUG("SceneManager initialized");
    }

    SceneManager::~SceneManager() = default;

    void SceneManager::setupEventHandlers() {

        // Handle PLY commands
        cmd::AddPLY::when([this](const auto& cmd) {
            addSplatFile(cmd.path, cmd.name);
        });

        cmd::RemovePLY::when([this](const auto& cmd) {
            removePLY(cmd.name, cmd.keep_children);
        });

        cmd::SetPLYVisibility::when([this](const auto& cmd) {
            setPLYVisibility(cmd.name, cmd.visible);
        });

        cmd::ClearScene::when([this](const auto&) {
            clear();
        });

        // Handle PLY cycling with proper event emission for UI updates
        cmd::CyclePLY::when([this](const auto&) {
            // Check if rendering manager has split view enabled (in PLY comparison mode)
            if (rendering_manager_) {
                auto settings = rendering_manager_->getSettings();
                if (settings.split_view_mode == lfs::vis::SplitViewMode::PLYComparison) {
                    // In split mode: advance the offset
                    rendering_manager_->advanceSplitOffset();
                    LOG_DEBUG("Advanced split view offset");
                    return; // Don't cycle visibility when in split view
                }
            }

            // Normal mode: existing cycle code
            if (content_type_ == ContentType::SplatFiles) {
                auto [hidden, shown] = scene_.cycleVisibilityWithNames();

                if (!hidden.empty()) {
                    cmd::SetPLYVisibility{.name = hidden, .visible = false}.emit();
                }
                if (!shown.empty()) {
                    cmd::SetPLYVisibility{.name = shown, .visible = true}.emit();
                    LOG_DEBUG("Cycled to: {}", shown);
                }

                emitSceneChanged();
            }
        });

        cmd::CropPLY::when([this](const auto& cmd) {
            handleCropActivePly(cmd.crop_box, cmd.inverse);
        });

        cmd::FitCropBoxToScene::when([this](const auto& cmd) {
            updateCropBoxToFitScene(cmd.use_percentile);
        });

        cmd::RenamePLY::when([this](const auto& cmd) {
            handleRenamePly(cmd);
        });

        cmd::ReparentNode::when([this](const auto& cmd) {
            handleReparentNode(cmd.node_name, cmd.new_parent_name);
        });

        cmd::AddGroup::when([this](const auto& cmd) {
            handleAddGroup(cmd.name, cmd.parent_name);
        });

        cmd::DuplicateNode::when([this](const auto& cmd) {
            handleDuplicateNode(cmd.name);
        });

        cmd::MergeGroup::when([this](const auto& cmd) {
            handleMergeGroup(cmd.name);
        });

        cmd::SetNodeLocked::when([this](const auto& cmd) {
            scene_.setNodeLocked(cmd.name, cmd.locked);
        });

        // Handle node selection from scene panel (both PLYs and Groups)
        ui::NodeSelected::when([this](const auto& event) {
            if (event.type == "PLY" || event.type == "Group") {
                // Skip if this is a multi-select event (already handled by selectNodes)
                auto it = event.metadata.find("multi_select");
                if (it != event.metadata.end() && it->second != "1") {
                    // Multi-select already set selected_nodes_, just sync cropbox
                    syncCropBoxToRenderSettings();
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    selected_nodes_.clear();
                    selected_nodes_.insert(event.path);
                }
                // Sync selected node's cropbox to render settings
                syncCropBoxToRenderSettings();
            }
        });

        // Handle node deselection
        ui::NodeDeselected::when([this](const auto&) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            selected_nodes_.clear();
        });
    }

    void SceneManager::changeContentType(const ContentType& type) {
        std::lock_guard<std::mutex> lock(state_mutex_);

        const char* type_str = (type == ContentType::Empty) ? "Empty" : (type == ContentType::SplatFiles) ? "SplatFiles"
                                                                                                          : "Dataset";
        LOG_DEBUG("Changing content type to: {}", type_str);

        content_type_ = type;
    }

    void SceneManager::loadSplatFile(const std::filesystem::path& path) {
        LOG_TIMER("SceneManager::loadSplatFile");

        try {
            LOG_INFO("Loading splat file: {}", path.string());

            // Clear existing scene
            clear();

            // Load the file
            LOG_DEBUG("Creating loader for splat file");
            auto loader = lfs::loader::Loader::create();
            lfs::loader::LoadOptions options{
                .resize_factor = -1,
                .max_width = 3840,
                .images_folder = "images",
                .validate_only = false};

            LOG_TRACE("Loading splat file with loader");
            auto load_result = loader->load(path, options);
            if (!load_result) {
                LOG_ERROR("Failed to load splat file: {}", load_result.error());
                throw std::runtime_error(load_result.error());
            }

            auto* splat_data = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&load_result->data);
            if (!splat_data || !*splat_data) {
                LOG_ERROR("Expected splat file but got different data type from: {}", path.string());
                throw std::runtime_error("Expected splat file but got different data type");
            }

            // Add to scene
            std::string name = path.stem().string();
            size_t gaussian_count = (*splat_data)->size();
            LOG_DEBUG("Adding '{}' to scene with {} gaussians", name, gaussian_count);

            scene_.addNode(name, std::make_unique<lfs::core::SplatData>(std::move(**splat_data)));

            // Create cropbox as child of this splat
            const auto* splat_node = scene_.getNode(name);
            if (splat_node) {
                const NodeId cropbox_id = scene_.getOrCreateCropBoxForSplat(splat_node->id);
                if (cropbox_id != NULL_NODE) {
                    LOG_DEBUG("Created cropbox for '{}'", name);
                }
            }

            // Update content state
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                content_type_ = ContentType::SplatFiles;
                splat_paths_.clear();
                splat_paths_[name] = path;
            }

            // Determine file type for event
            auto ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            auto file_type = (ext == ".sog") ? state::SceneLoaded::Type::SOG : state::SceneLoaded::Type::PLY;

            // Emit events
            state::SceneLoaded{
                .scene = nullptr,
                .path = path,
                .type = file_type,
                .num_gaussians = scene_.getTotalGaussianCount()}
                .emit();

            state::PLYAdded{
                .name = name,
                .node_gaussians = gaussian_count,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = true,
                .parent_name = "",
                .is_group = false,
                .node_type = 0}  // SPLAT
                .emit();

            // Emit PLYAdded for the cropbox (re-lookup splat as vector may have reallocated)
            const auto* splat_for_cropbox = scene_.getNode(name);
            if (splat_for_cropbox) {
                const NodeId cropbox_id = scene_.getCropBoxForSplat(splat_for_cropbox->id);
                if (cropbox_id != NULL_NODE) {
                    const auto* cropbox_node = scene_.getNodeById(cropbox_id);
                    if (cropbox_node) {
                        LOG_DEBUG("Emitting PLYAdded for cropbox '{}'", cropbox_node->name);
                        state::PLYAdded{
                            .name = cropbox_node->name,
                            .node_gaussians = 0,
                            .total_gaussians = scene_.getTotalGaussianCount(),
                            .is_visible = true,
                            .parent_name = name,
                            .is_group = false,
                            .node_type = 2}  // CROPBOX
                            .emit();
                    }
                }
            }

            emitSceneChanged();
            updateCropBoxToFitScene(true);
            selectNode(name);
            tools::SetToolbarTool{.tool_mode = static_cast<int>(gui::panels::ToolMode::CropBox)}.emit();

            LOG_INFO("Loaded '{}' with {} gaussians", name, gaussian_count);

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load splat file: {} (path: {})", e.what(), path.string());
            throw;
        }
    }

    void SceneManager::addSplatFile(const std::filesystem::path& path, const std::string& name_hint,
                                    bool is_visible) {
        LOG_TIMER_TRACE("SceneManager::addSplatFile");

        try {
            // If not in splat mode, switch to it
            if (content_type_ != ContentType::SplatFiles) {
                LOG_DEBUG("Not in splat mode, switching to splat mode and loading");
                loadSplatFile(path);
                return;
            }

            LOG_INFO("Adding splat file to scene: {}", path.string());

            // Load the file
            auto loader = lfs::loader::Loader::create();
            lfs::loader::LoadOptions options{
                .resize_factor = -1,
                .max_width = 3840,
                .images_folder = "images",
                .validate_only = false};

            LOG_TRACE("Loading splat data");
            auto load_result = loader->load(path, options);
            if (!load_result) {
                LOG_ERROR("Failed to load splat file: {}", load_result.error());
                throw std::runtime_error(load_result.error());
            }

            auto* splat_data = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&load_result->data);
            if (!splat_data || !*splat_data) {
                LOG_ERROR("Expected splat file from: {}", path.string());
                throw std::runtime_error("Expected splat file");
            }

            // Generate unique name
            std::string base_name = name_hint.empty() ? path.stem().string() : name_hint;
            std::string name = base_name;
            int counter = 1;

            while (scene_.getNode(name) != nullptr) {
                name = std::format("{}_{}", base_name, counter++);
                LOG_TRACE("Name '{}' already exists, trying '{}'", base_name, name);
            }

            size_t gaussian_count = (*splat_data)->size();
            LOG_DEBUG("Adding node '{}' with {} gaussians", name, gaussian_count);

            scene_.addNode(name, std::make_unique<lfs::core::SplatData>(std::move(**splat_data)));

            // Create cropbox as child of this splat
            const auto* splat_node = scene_.getNode(name);
            if (splat_node) {
                [[maybe_unused]] const NodeId cropbox_id = scene_.getOrCreateCropBoxForSplat(splat_node->id);
            }

            // Update paths
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                splat_paths_[name] = path;
            }

            state::PLYAdded{
                .name = name,
                .node_gaussians = gaussian_count,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = is_visible,
                .parent_name = "",
                .is_group = false,
                .node_type = 0}  // SPLAT
                .emit();

            // Emit PLYAdded for the cropbox (re-lookup splat as vector may have reallocated)
            const auto* splat_for_cropbox = scene_.getNode(name);
            if (splat_for_cropbox) {
                const NodeId cropbox_id = scene_.getCropBoxForSplat(splat_for_cropbox->id);
                if (cropbox_id != NULL_NODE) {
                    const auto* cropbox_node = scene_.getNodeById(cropbox_id);
                    if (cropbox_node) {
                        LOG_DEBUG("Emitting PLYAdded for cropbox '{}'", cropbox_node->name);
                        state::PLYAdded{
                            .name = cropbox_node->name,
                            .node_gaussians = 0,
                            .total_gaussians = scene_.getTotalGaussianCount(),
                            .is_visible = true,
                            .parent_name = name,
                            .is_group = false,
                            .node_type = 2}  // CROPBOX
                            .emit();
                    }
                }
            }

            emitSceneChanged();
            updateCropBoxToFitScene(true);
            selectNode(name);

            LOG_INFO("Added '{}' ({} gaussians)", name, gaussian_count);

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to add splat file: {} (path: {})", e.what(), path.string());
            throw;
        }
    }

    void SceneManager::removePLY(const std::string& name, const bool keep_children) {
        std::string parent_name;
        if (const auto* node = scene_.getNode(name)) {
            if (node->parent_id != NULL_NODE) {
                if (const auto* p = scene_.getNodeById(node->parent_id)) {
                    parent_name = p->name;
                }
            }
        }

        scene_.removeNode(name, keep_children);
        {
            std::lock_guard lock(state_mutex_);
            splat_paths_.erase(name);
        }

        if (lfs_project_) lfs_project_->removePly(name);

        if (scene_.getNodeCount() == 0) {
            std::lock_guard lock(state_mutex_);
            content_type_ = ContentType::Empty;
        }

        state::PLYRemoved{.name = name, .children_kept = keep_children, .parent_of_removed = parent_name}.emit();
        emitSceneChanged();
    }

    void SceneManager::setPLYVisibility(const std::string& name, bool visible) {
        LOG_TRACE("Setting '{}' visibility to: {}", name, visible);
        scene_.setNodeVisibility(name, visible);
        emitSceneChanged();
    }

    // ========== Node Selection ==========

    void SceneManager::selectNode(const std::string& name) {
        const auto* node = scene_.getNode(name);
        if (node != nullptr) {
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                selected_nodes_.clear();
                selected_nodes_.insert(name);
            }
            ui::NodeSelected{
                .path = name,
                .type = "PLY",
                .metadata = {
                    {"name", name},
                    {"gaussians", std::to_string(node->model ? node->model->size() : 0)},
                    {"visible", node->visible ? "true" : "false"}
                }
            }.emit();
        }
    }

    void SceneManager::selectNodes(const std::vector<std::string>& names) {
        std::string first_name;
        size_t num_selected = 0;
        size_t gaussians = 0;
        bool visible = false;

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            selected_nodes_.clear();
            for (const auto& name : names) {
                if (scene_.getNode(name) != nullptr) {
                    selected_nodes_.insert(name);
                }
            }
            if (!selected_nodes_.empty()) {
                first_name = *selected_nodes_.begin();
                num_selected = selected_nodes_.size();
                const auto* node = scene_.getNode(first_name);
                if (node) {
                    gaussians = node->model ? node->model->size() : 0;
                    visible = node->visible;
                }
            }
        }

        if (!first_name.empty()) {
            ui::NodeSelected{
                .path = first_name,
                .type = "PLY",
                .metadata = {
                    {"name", first_name},
                    {"gaussians", std::to_string(gaussians)},
                    {"visible", visible ? "true" : "false"},
                    {"multi_select", std::to_string(num_selected)}
                }
            }.emit();
        }
    }

    void SceneManager::addToSelection(const std::string& name) {
        const auto* node = scene_.getNode(name);
        if (node != nullptr) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            selected_nodes_.insert(name);
        }
    }

    void SceneManager::clearSelection() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        selected_nodes_.clear();
        LOG_TRACE("Cleared node selection");
    }

    std::string SceneManager::getSelectedNodeName() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty()) return "";
        return *selected_nodes_.begin();
    }

    std::vector<std::string> SceneManager::getSelectedNodeNames() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return std::vector<std::string>(selected_nodes_.begin(), selected_nodes_.end());
    }

    bool SceneManager::hasSelectedNode() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty()) return false;
        // Check if at least one selected node still exists
        for (const auto& name : selected_nodes_) {
            if (scene_.getNode(name) != nullptr) return true;
        }
        return false;
    }

    bool SceneManager::isSelectedNodeLocked() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty()) return false;
        // Return true if any selected node is locked
        for (const auto& name : selected_nodes_) {
            if (scene_.isNodeLocked(name)) return true;
        }
        return false;
    }

    int SceneManager::getSelectedNodeIndex() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty()) { return -1; }
        return scene_.getVisibleNodeIndex(*selected_nodes_.begin());
    }

    std::vector<bool> SceneManager::getSelectedNodeMask() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return scene_.getSelectedNodeMask(std::vector<std::string>(selected_nodes_.begin(), selected_nodes_.end()));
    }

    std::string SceneManager::pickNodeAtWorldPosition(const glm::vec3& world_pos) const {
        constexpr float INVALID_COORD = -1e9f;
        if (world_pos.x <= INVALID_COORD) {
            return "";
        }

        for (const auto* node : scene_.getVisibleNodes()) {
            if (node->type != NodeType::SPLAT) continue;

            glm::vec3 local_min, local_max;
            if (!scene_.getNodeBounds(node->id, local_min, local_max)) continue;

            const glm::mat4 world_to_local = glm::inverse(scene_.getWorldTransform(node->id));
            const glm::vec3 local_pos = glm::vec3(world_to_local * glm::vec4(world_pos, 1.0f));

            if (local_pos.x >= local_min.x && local_pos.x <= local_max.x &&
                local_pos.y >= local_min.y && local_pos.y <= local_max.y &&
                local_pos.z >= local_min.z && local_pos.z <= local_max.z) {
                return node->name;
            }
        }
        return "";
    }

    std::vector<std::string> SceneManager::pickNodesInScreenRect(
        const glm::vec2& rect_min, const glm::vec2& rect_max,
        const glm::mat4& view, const glm::mat4& proj,
        const glm::ivec2& viewport_size) const {

        constexpr float BEHIND_CAMERA = -1e10f;
        constexpr int BBOX_CORNERS = 8;

        std::vector<std::string> result;

        const auto projectToScreen = [&](const glm::vec3& world_pos) -> glm::vec2 {
            const glm::vec4 clip = proj * view * glm::vec4(world_pos, 1.0f);
            if (clip.w <= 0.0f) return glm::vec2(BEHIND_CAMERA);
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            return glm::vec2(
                (ndc.x * 0.5f + 0.5f) * static_cast<float>(viewport_size.x),
                (1.0f - (ndc.y * 0.5f + 0.5f)) * static_cast<float>(viewport_size.y));
        };

        const auto rectsOverlap = [](const glm::vec2& a_min, const glm::vec2& a_max,
                                     const glm::vec2& b_min, const glm::vec2& b_max) {
            return !(a_max.x < b_min.x || b_max.x < a_min.x ||
                     a_max.y < b_min.y || b_max.y < a_min.y);
        };

        for (const auto* node : scene_.getVisibleNodes()) {
            if (node->type != NodeType::SPLAT) continue;

            glm::vec3 local_min, local_max;
            if (!scene_.getNodeBounds(node->id, local_min, local_max)) continue;

            const glm::mat4 world_transform = scene_.getWorldTransform(node->id);

            glm::vec2 screen_min(1e10f);
            glm::vec2 screen_max(-1e10f);
            bool any_visible = false;

            for (int i = 0; i < BBOX_CORNERS; ++i) {
                const glm::vec3 corner(
                    (i & 1) ? local_max.x : local_min.x,
                    (i & 2) ? local_max.y : local_min.y,
                    (i & 4) ? local_max.z : local_min.z);
                const glm::vec3 world_corner = glm::vec3(world_transform * glm::vec4(corner, 1.0f));
                const glm::vec2 screen_pos = projectToScreen(world_corner);

                if (screen_pos.x > BEHIND_CAMERA + 1e5f) {
                    screen_min = glm::min(screen_min, screen_pos);
                    screen_max = glm::max(screen_max, screen_pos);
                    any_visible = true;
                }
            }

            if (!any_visible) continue;

            // Check if the screen-space bbox overlaps with the selection rectangle
            if (rectsOverlap(rect_min, rect_max, screen_min, screen_max)) {
                result.push_back(node->name);
            }
        }

        return result;
    }

    void SceneManager::ensureCropBoxForSelectedNode() {
        std::string node_name;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (selected_nodes_.empty()) return;
            node_name = *selected_nodes_.begin();
        }
        if (node_name.empty()) return;

        const auto* node = scene_.getNode(node_name);
        if (!node) return;

        // For CROPBOX nodes, use parent SPLAT
        NodeId target_id = node->id;
        if (node->type == NodeType::CROPBOX && node->parent_id != NULL_NODE) {
            target_id = node->parent_id;
        } else if (node->type == NodeType::GROUP) {
            // For groups, find first child SPLAT
            for (const NodeId child_id : node->children) {
                if (const auto* child = scene_.getNodeById(child_id)) {
                    if (child->type == NodeType::SPLAT) {
                        target_id = child_id;
                        break;
                    }
                }
            }
        }

        const auto* target = scene_.getNodeById(target_id);
        if (!target || target->type != NodeType::SPLAT) return;

        // Check if cropbox already exists
        const NodeId existing = scene_.getCropBoxForSplat(target_id);
        if (existing != NULL_NODE) return;

        // Create cropbox and fit to bounds
        const NodeId cropbox_id = scene_.getOrCreateCropBoxForSplat(target_id);
        if (cropbox_id == NULL_NODE) return;

        // Fit cropbox to node bounds
        glm::vec3 min_bounds, max_bounds;
        if (scene_.getNodeBounds(target_id, min_bounds, max_bounds)) {
            CropBoxData data;
            data.min = min_bounds;
            data.max = max_bounds;
            data.enabled = true;
            scene_.setCropBoxData(cropbox_id, data);
        }

        // Emit PLYAdded for the new cropbox
        if (const auto* cropbox = scene_.getNodeById(cropbox_id)) {
            state::PLYAdded{
                .name = cropbox->name,
                .node_gaussians = 0,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = cropbox->visible,
                .parent_name = target->name,
                .is_group = false,
                .node_type = static_cast<int>(NodeType::CROPBOX)
            }.emit();
        }

        LOG_DEBUG("Created cropbox for node '{}'", target->name);
    }

    // ========== Node Transforms ==========

    void SceneManager::setNodeTransform(const std::string& name, const glm::mat4& transform) {
        scene_.setNodeTransform(name, transform);
        emitSceneChanged();
    }

    glm::mat4 SceneManager::getNodeTransform(const std::string& name) const {
        return scene_.getNodeTransform(name);
    }

    void SceneManager::setSelectedNodeTranslation(const glm::vec3& translation) {
        std::string node_name;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (selected_nodes_.empty()) {
                LOG_TRACE("No node selected for translation");
                return;
            }
            node_name = *selected_nodes_.begin();
        }

        if (node_name.empty()) {
            LOG_TRACE("No node selected for translation");
            return;
        }

        // Create translation matrix
        glm::mat4 transform = glm::mat4(1.0f);
        transform[3][0] = translation.x;
        transform[3][1] = translation.y;
        transform[3][2] = translation.z;

        scene_.setNodeTransform(node_name, transform);
        emitSceneChanged();
    }

    glm::vec3 SceneManager::getSelectedNodeTranslation() const {
        std::string node_name;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (selected_nodes_.empty()) return glm::vec3(0.0f);
            node_name = *selected_nodes_.begin();
        }

        if (node_name.empty()) {
            return glm::vec3(0.0f);
        }

        glm::mat4 transform = scene_.getNodeTransform(node_name);
        return glm::vec3(transform[3][0], transform[3][1], transform[3][2]);
    }

    glm::vec3 SceneManager::getSelectedNodeCentroid() const {
        std::string node_name;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (selected_nodes_.empty()) return glm::vec3(0.0f);
            node_name = *selected_nodes_.begin();
        }

        const auto* node = scene_.getNode(node_name);
        if (!node || !node->model) return glm::vec3(0.0f);
        return node->centroid;
    }

    glm::vec3 SceneManager::getSelectedNodeCenter() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty()) return glm::vec3(0.0f);

        const std::string& node_name = *selected_nodes_.begin();
        const auto* const node = scene_.getNode(node_name);
        if (!node) return glm::vec3(0.0f);

        return scene_.getNodeBoundsCenter(node->id);
    }

    void SceneManager::setSelectedNodeTransform(const glm::mat4& transform) {
        std::string node_name;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (selected_nodes_.empty()) {
                LOG_TRACE("No node selected for transform");
                return;
            }
            node_name = *selected_nodes_.begin();
        }

        LOG_DEBUG("setSelectedNodeTransform '{}': pos=[{:.2f}, {:.2f}, {:.2f}]",
                  node_name, transform[3][0], transform[3][1], transform[3][2]);
        scene_.setNodeTransform(node_name, transform);
        emitSceneChanged();
    }

    glm::mat4 SceneManager::getSelectedNodeTransform() const {
        std::string node_name;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (selected_nodes_.empty()) return glm::mat4(1.0f);
            node_name = *selected_nodes_.begin();
        }

        return scene_.getNodeTransform(node_name);
    }

    glm::vec3 SceneManager::getSelectionCenter() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty()) return glm::vec3(0.0f);

        // Single selection - use node's local bounds center
        if (selected_nodes_.size() == 1) {
            const auto* node = scene_.getNode(*selected_nodes_.begin());
            if (!node) return glm::vec3(0.0f);
            return scene_.getNodeBoundsCenter(node->id);
        }

        // Multi-selection - compute combined bounds center in local space
        glm::vec3 total_min(std::numeric_limits<float>::max());
        glm::vec3 total_max(std::numeric_limits<float>::lowest());
        bool has_bounds = false;

        for (const auto& name : selected_nodes_) {
            const auto* node = scene_.getNode(name);
            if (!node) continue;

            glm::vec3 node_min, node_max;
            if (scene_.getNodeBounds(node->id, node_min, node_max)) {
                total_min = glm::min(total_min, node_min);
                total_max = glm::max(total_max, node_max);
                has_bounds = true;
            }
        }

        return has_bounds ? (total_min + total_max) * 0.5f : glm::vec3(0.0f);
    }

    glm::vec3 SceneManager::getSelectionWorldCenter() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty()) return glm::vec3(0.0f);

        glm::vec3 total_min(std::numeric_limits<float>::max());
        glm::vec3 total_max(std::numeric_limits<float>::lowest());
        bool has_bounds = false;

        for (const auto& name : selected_nodes_) {
            const auto* node = scene_.getNode(name);
            if (!node) continue;

            glm::vec3 local_min, local_max;
            if (!scene_.getNodeBounds(node->id, local_min, local_max)) continue;

            const glm::mat4 world_transform = scene_.getWorldTransform(node->id);
            const glm::vec3 corners[8] = {
                {local_min.x, local_min.y, local_min.z},
                {local_max.x, local_min.y, local_min.z},
                {local_min.x, local_max.y, local_min.z},
                {local_max.x, local_max.y, local_min.z},
                {local_min.x, local_min.y, local_max.z},
                {local_max.x, local_min.y, local_max.z},
                {local_min.x, local_max.y, local_max.z},
                {local_max.x, local_max.y, local_max.z}
            };

            for (const auto& corner : corners) {
                const glm::vec3 world_corner = glm::vec3(world_transform * glm::vec4(corner, 1.0f));
                total_min = glm::min(total_min, world_corner);
                total_max = glm::max(total_max, world_corner);
            }
            has_bounds = true;
        }

        return has_bounds ? (total_min + total_max) * 0.5f : glm::vec3(0.0f);
    }

    // ========== Cropbox Operations ==========

    NodeId SceneManager::getSelectedNodeCropBoxId() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty()) return NULL_NODE;

        const auto* node = scene_.getNode(*selected_nodes_.begin());
        if (!node) return NULL_NODE;

        // If selected node is a cropbox, return its ID
        if (node->type == NodeType::CROPBOX) {
            return node->id;
        }

        // If selected node is a splat, return its cropbox child
        if (node->type == NodeType::SPLAT) {
            return scene_.getCropBoxForSplat(node->id);
        }

        // For groups, no cropbox
        return NULL_NODE;
    }

    CropBoxData* SceneManager::getSelectedNodeCropBox() {
        const NodeId cropbox_id = getSelectedNodeCropBoxId();
        if (cropbox_id == NULL_NODE) return nullptr;
        return scene_.getCropBoxData(cropbox_id);
    }

    const CropBoxData* SceneManager::getSelectedNodeCropBox() const {
        const NodeId cropbox_id = getSelectedNodeCropBoxId();
        if (cropbox_id == NULL_NODE) return nullptr;
        return scene_.getCropBoxData(cropbox_id);
    }

    void SceneManager::syncCropBoxToRenderSettings() {
        // Scene graph is single source of truth - just trigger re-render
        if (rendering_manager_) {
            rendering_manager_->markDirty();
        }
    }

    void SceneManager::loadDataset(const std::filesystem::path& path,
                                   const lfs::core::param::TrainingParameters& params) {
        LOG_TIMER("SceneManager::loadDataset");

        try {
            LOG_INFO("Loading dataset: {}", path.string());

            // Stop any existing training
            if (trainer_manager_) {
                LOG_DEBUG("Clearing existing trainer");
                trainer_manager_->clearTrainer();
            }

            // Clear scene
            clear();

            // Setup training
            auto dataset_params = params;
            dataset_params.dataset.data_path = path;
            cached_params_ = dataset_params;

            LOG_DEBUG("Setting up training with parameters");
            LOG_TRACE("Dataset path: {}", path.string());
            LOG_TRACE("Iterations: {}", dataset_params.optimization.iterations);

            auto setup_result = lfs::training::setupTraining(dataset_params);
            if (!setup_result) {
                LOG_ERROR("Failed to setup training: {}", setup_result.error());
                throw std::runtime_error(setup_result.error());
            }

            // Pass trainer to manager
            if (trainer_manager_) {
                LOG_DEBUG("Setting trainer in manager");
                trainer_manager_->setTrainer(std::move(setup_result->trainer));
            } else {
                LOG_ERROR("No trainer manager available");
                throw std::runtime_error("No trainer manager available");
            }

            // Update content state
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                content_type_ = ContentType::Dataset;
                dataset_path_ = path;
            }

            // Emit events
            const size_t num_gaussians = trainer_manager_->getTrainer()
                                             ->get_strategy()
                                             .get_model()
                                             .size();

            LOG_INFO("Dataset loaded successfully - {} images, {} initial gaussians",
                     setup_result->dataset->size(), num_gaussians);

            state::SceneLoaded{
                .scene = nullptr,
                .path = path,
                .type = state::SceneLoaded::Type::Dataset,
                .num_gaussians = num_gaussians}
                .emit();

            state::DatasetLoadCompleted{
                .path = path,
                .success = true,
                .error = std::nullopt,
                .num_images = setup_result->dataset->size(),
                .num_points = num_gaussians}
                .emit();

            emitSceneChanged();

            // Switch to point cloud rendering mode by default for datasets
            // Re-enabled with debug logging to investigate dimension mismatch
            if (num_gaussians > 0 && trainer_manager_ && trainer_manager_->getTrainer()) {
                ui::PointCloudModeChanged{
                    .enabled = true,
                    .voxel_size = 0.01f}
                    .emit();
                LOG_INFO("Switched to point cloud rendering mode for dataset ({} gaussians)", num_gaussians);
            }

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load dataset: {} (path: {})", e.what(), path.string());
            throw;
        }
    }

    void SceneManager::clear() {
        LOG_DEBUG("Clearing scene");

        // Stop training if active
        if (trainer_manager_ && content_type_ == ContentType::Dataset) {
            LOG_DEBUG("Stopping training before clearing");
            cmd::StopTraining{}.emit();
            trainer_manager_->clearTrainer();
        }

        scene_.clear();

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            content_type_ = ContentType::Empty;
            splat_paths_.clear();
            dataset_path_.clear();
        }

        state::SceneCleared{}.emit();
        emitSceneChanged();

        LOG_INFO("Scene cleared");
    }

    const lfs::core::SplatData* SceneManager::getModelForRendering() const {
        std::lock_guard<std::mutex> lock(state_mutex_);

        if (content_type_ == ContentType::SplatFiles) {
            return scene_.getCombinedModel();
        } else if (content_type_ == ContentType::Dataset) {
            if (trainer_manager_ && trainer_manager_->getTrainer()) {
                return &trainer_manager_->getTrainer()->get_strategy().get_model();
            }
        }

        return nullptr;
    }

    SceneRenderState SceneManager::buildRenderState() const {
        std::lock_guard<std::mutex> lock(state_mutex_);

        SceneRenderState state;

        // Get combined model
        if (content_type_ == ContentType::SplatFiles) {
            state.combined_model = scene_.getCombinedModel();
        } else if (content_type_ == ContentType::Dataset) {
            if (trainer_manager_ && trainer_manager_->getTrainer()) {
                state.combined_model = &trainer_manager_->getTrainer()->get_strategy().get_model();
            }
        }

        // Get transforms and indices
        state.model_transforms = scene_.getVisibleNodeTransforms();
        state.transform_indices = scene_.getTransformIndices();
        state.visible_splat_count = state.model_transforms.size();


        // Get selection mask
        state.selection_mask = scene_.getSelectionMask();
        state.has_selection = scene_.hasSelection();

        // Get selected node info for desaturation
        // Use mask-based approach: when a group is selected, all descendant SPLAT nodes are marked
        state.selected_node_name = selected_nodes_.empty() ? "" : *selected_nodes_.begin();
        state.selected_node_mask = scene_.getSelectedNodeMask(std::vector<std::string>(selected_nodes_.begin(), selected_nodes_.end()));

        // Get cropboxes
        state.cropboxes = scene_.getVisibleCropBoxes();

        // Find selected cropbox index
        if (!selected_nodes_.empty()) {
            const auto* selected = scene_.getNode(*selected_nodes_.begin());
            if (selected) {
                NodeId cropbox_id = NULL_NODE;
                if (selected->type == NodeType::CROPBOX) {
                    cropbox_id = selected->id;
                } else if (selected->type == NodeType::SPLAT) {
                    cropbox_id = scene_.getCropBoxForSplat(selected->id);
                }
                if (cropbox_id != NULL_NODE) {
                    for (size_t i = 0; i < state.cropboxes.size(); ++i) {
                        if (state.cropboxes[i].node_id == cropbox_id) {
                            state.selected_cropbox_index = static_cast<int>(i);
                            break;
                        }
                    }
                }
            }
        }

        return state;
    }

    SceneManager::SceneInfo SceneManager::getSceneInfo() const {
        std::lock_guard<std::mutex> lock(state_mutex_);

        SceneInfo info;

        switch (content_type_) {
        case ContentType::Empty:
            info.source_type = "Empty";
            break;

        case ContentType::SplatFiles:
            info.has_model = scene_.hasNodes();
            info.num_gaussians = scene_.getTotalGaussianCount();
            info.num_nodes = scene_.getNodeCount();
            info.source_type = "Splat";
            if (!splat_paths_.empty()) {
                info.source_path = splat_paths_.rbegin()->second; // get the "last" element of the splat_paths_
                // Determine specific type from extension
                auto ext = info.source_path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".sog") {
                    info.source_type = "SOG";
                } else if (ext == ".ply") {
                    info.source_type = "PLY";
                }
            }
            break;

        case ContentType::Dataset:
            info.has_model = trainer_manager_ && trainer_manager_->getTrainer();
            if (info.has_model) {
                info.num_gaussians = trainer_manager_->getTrainer()
                                         ->get_strategy()
                                         .get_model()
                                         .size();
            }
            info.num_nodes = 1;
            info.source_type = "Dataset";
            info.source_path = dataset_path_;
            break;
        }

        LOG_TRACE("Scene info - type: {}, gaussians: {}, nodes: {}",
                  info.source_type, info.num_gaussians, info.num_nodes);

        return info;
    }

    void SceneManager::emitSceneChanged() {
        state::SceneChanged{}.emit();
    }

    void SceneManager::setRenderingManager(RenderingManager* rm) {
        rendering_manager_ = rm;
    }

    void SceneManager::handleCropActivePly(const lfs::geometry::BoundingBox& crop_box, const bool inverse) {
        changeContentType(ContentType::SplatFiles);

        // Only crop the selected node, not all visible nodes
        std::vector<std::string> node_names;
        bool had_selection = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!selected_nodes_.empty()) {
                had_selection = true;
                // Add all selected SPLAT nodes
                for (const auto& node_name : selected_nodes_) {
                    const auto* selected = scene_.getNode(node_name);
                    if (selected && selected->type == NodeType::SPLAT) {
                        node_names.push_back(node_name);
                        LOG_INFO("Cropping selected SPLAT node: {}", node_name);
                    } else if (selected) {
                        LOG_INFO("Selected node '{}' is not a SPLAT (type={}), not cropping",
                                  node_name, static_cast<int>(selected->type));
                    } else {
                        LOG_WARN("Selected node '{}' not found in scene", node_name);
                    }
                }
            }
        }

        // Only fall back to all visible nodes if there was NO selection at all
        // If there was a selection but it wasn't a SPLAT, don't crop anything
        if (node_names.empty() && !had_selection) {
            LOG_INFO("No selection, falling back to all visible nodes");
            for (const auto* node : scene_.getVisibleNodes()) {
                node_names.push_back(node->name);
            }
        }

        if (node_names.empty()) {
            LOG_INFO("No nodes to crop");
            return;
        }

        for (const auto& node_name : node_names) {
            auto* node = scene_.getMutableNode(node_name);
            if (!node || !node->model) {
                continue;
            }

            try {
                const size_t original_count = node->model->size();
                const size_t original_visible = node->model->visible_count();

                // Capture old deletion mask for undo
                lfs::core::Tensor old_deleted_mask = node->model->has_deleted_mask()
                    ? node->model->deleted().clone()
                    : lfs::core::Tensor::zeros({original_count}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);

                // Transform crop box to node's local space if node has a transform
                lfs::geometry::BoundingBox local_crop_box = crop_box;
                const glm::mat4 node_world_transform = scene_.getWorldTransform(node->id);
                static const glm::mat4 IDENTITY_MATRIX(1.0f);

                if (node_world_transform != IDENTITY_MATRIX) {
                    // Combine: node_local -> world -> cropbox_local
                    // world2bbox transforms world -> cropbox_local
                    // node_world transforms node_local -> world
                    // So we need: world2bbox * node_world to go node_local -> cropbox_local
                    if (crop_box.hasFullTransform()) {
                        local_crop_box.setworld2BBox(crop_box.getworld2BBoxMat4() * node_world_transform);
                    } else {
                        const lfs::geometry::EuclideanTransform node_to_world(node_world_transform);
                        local_crop_box.setworld2BBox(crop_box.getworld2BBox() * node_to_world);
                    }
                }

                const auto applied_mask = lfs::core::soft_crop_by_cropbox(*node->model, local_crop_box, inverse);
                if (!applied_mask.is_valid()) {
                    continue;
                }

                const size_t new_visible = node->model->visible_count();
                if (new_visible == original_visible) {
                    continue;
                }

                LOG_INFO("Cropped '{}': {} -> {} visible", node_name, original_visible, new_visible);

                if (command_history_) {
                    lfs::core::Tensor new_deleted_mask = node->model->deleted().clone();
                    auto cmd = std::make_unique<command::CropCommand>(
                        this, node_name, std::move(old_deleted_mask), std::move(new_deleted_mask));
                    command_history_->execute(std::move(cmd));
                }

                state::PLYAdded{
                    .name = node_name,
                    .node_gaussians = new_visible,
                    .total_gaussians = scene_.getTotalGaussianCount(),
                    .is_visible = true,
                    .parent_name = "",
                    .is_group = false,
                    .node_type = 0  // SPLAT
                }.emit();

            } catch (const std::exception& e) {
                LOG_ERROR("Failed to crop '{}': {}", node_name, e.what());
            }
        }

        emitSceneChanged();
    }

    void SceneManager::updatePlyPath(const std::string& ply_name, const std::filesystem::path& ply_path) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto it = splat_paths_.find(ply_name);
        if (it != splat_paths_.end()) {
            it->second = ply_path;
        } else {
            LOG_WARN("ply name was not found {}", ply_name);
        }
    }

    size_t SceneManager::applyDeleted() {
        const size_t removed = scene_.applyDeleted();
        if (removed > 0 && rendering_manager_) {
            rendering_manager_->markDirty();
        }
        return removed;
    }

    bool SceneManager::renamePLY(const std::string& old_name, const std::string& new_name) {
        LOG_DEBUG("Renaming '{}' to '{}'", old_name, new_name);

        // Attempt to rename in the scene
        bool success = scene_.renameNode(old_name, new_name);

        if (success && old_name != new_name) {
            // Update the splat_paths_ map to use the new name
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                auto it = splat_paths_.find(old_name);
                if (it != splat_paths_.end()) {
                    auto path = it->second;

                    splat_paths_.erase(it);
                    std::filesystem::path new_ply_path;
                    // chaning ply name and path
                    std::filesystem::path new_path = path;
                    // resposibility of file systems changes should be on the project
                    if (lfs_project_) {
                        auto parent = path.parent_path();
                        auto extension = path.extension();
                        new_path = parent / (new_name + extension.string());
                        if (!lfs_project_->updatePlyPath(old_name, new_path)) {
                            // os rename failed - reducing to old path
                            new_path = path;
                        }
                        lfs_project_->renamePly(old_name, new_name);
                    }

                    splat_paths_[new_name] = new_path;
                }
            }

            emitSceneChanged();

            LOG_INFO("Successfully renamed '{}' to '{}'", old_name, new_name);
        } else if (!success) {
            LOG_WARN("Failed to rename '{}' to '{}' - name may already exist", old_name, new_name);
        }

        return success;
    }
    void SceneManager::handleRenamePly(const cmd::RenamePLY& event) {
        renamePLY(event.old_name, event.new_name);
    }

    void SceneManager::handleReparentNode(const std::string& node_name, const std::string& new_parent_name) {
        auto* node = scene_.getMutableNode(node_name);
        if (!node) return;

        std::string old_parent_name;
        if (node->parent_id != NULL_NODE) {
            if (const auto* p = scene_.getNodeById(node->parent_id)) {
                old_parent_name = p->name;
            }
        }

        NodeId parent_id = NULL_NODE;
        if (!new_parent_name.empty()) {
            const auto* parent = scene_.getNode(new_parent_name);
            if (!parent) return;
            parent_id = parent->id;
        }

        scene_.reparent(node->id, parent_id);
        state::NodeReparented{.name = node_name, .old_parent = old_parent_name, .new_parent = new_parent_name}.emit();
        emitSceneChanged();
    }

    void SceneManager::handleAddGroup(const std::string& name, const std::string& parent_name) {
        NodeId parent_id = NULL_NODE;
        if (!parent_name.empty()) {
            const auto* parent = scene_.getNode(parent_name);
            if (!parent) return;
            parent_id = parent->id;
        }

        std::string unique_name = name;
        for (int i = 1; scene_.getNode(unique_name); ++i) {
            unique_name = std::format("{} {}", name, i);
        }

        scene_.addGroup(unique_name, parent_id);
        state::PLYAdded{
            .name = unique_name,
            .node_gaussians = 0,
            .total_gaussians = scene_.getTotalGaussianCount(),
            .is_visible = true,
            .parent_name = parent_name,
            .is_group = true,
            .node_type = 1  // GROUP
        }.emit();
    }

    void SceneManager::handleDuplicateNode(const std::string& name) {
        const auto* src = scene_.getNode(name);
        if (!src) return;

        std::string parent_name;
        if (src->parent_id != NULL_NODE) {
            if (const auto* p = scene_.getNodeById(src->parent_id)) {
                parent_name = p->name;
            }
        }

        const std::string new_name = scene_.duplicateNode(name);
        if (new_name.empty()) return;

        // Emit PLYAdded for duplicated node tree
        std::function<void(const std::string&, const std::string&)> emit_added =
            [&](const std::string& n, const std::string& pn) {
            const auto* node = scene_.getNode(n);
            if (!node) return;

            state::PLYAdded{
                .name = node->name,
                .node_gaussians = node->gaussian_count,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = node->visible,
                .parent_name = pn,
                .is_group = node->type == NodeType::GROUP,
                .node_type = static_cast<int>(node->type)
            }.emit();

            for (const NodeId cid : node->children) {
                if (const auto* c = scene_.getNodeById(cid)) {
                    emit_added(c->name, node->name);
                }
            }
        };

        emit_added(new_name, parent_name);
        emitSceneChanged();
    }

    void SceneManager::handleMergeGroup(const std::string& name) {
        const auto* group = scene_.getNode(name);
        if (!group || group->type != NodeType::GROUP) {
            return;
        }

        std::string parent_name;
        if (group->parent_id != NULL_NODE) {
            if (const auto* p = scene_.getNodeById(group->parent_id)) {
                parent_name = p->name;
            }
        }

        // Check if the group being merged is currently selected
        bool was_selected = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            was_selected = (selected_nodes_.count(name) > 0);
            if (was_selected) {
                selected_nodes_.erase(name);
            }
        }

        // Collect children to emit PLYRemoved events
        std::vector<std::string> children_to_remove;
        std::function<void(const SceneNode*)> collect_children = [&](const SceneNode* n) {
            for (const NodeId cid : n->children) {
                if (const auto* c = scene_.getNodeById(cid)) {
                    children_to_remove.push_back(c->name);
                    collect_children(c);
                }
            }
        };
        collect_children(group);

        const std::string merged_name = scene_.mergeGroup(name);
        if (merged_name.empty()) {
            LOG_WARN("Failed to merge group '{}'", name);
            return;
        }

        // Emit PLYRemoved for all original children and the group
        for (const auto& child_name : children_to_remove) {
            state::PLYRemoved{.name = child_name}.emit();
        }
        state::PLYRemoved{.name = name}.emit();

        // Emit PLYAdded for merged node
        const auto* merged = scene_.getNode(merged_name);
        if (merged) {
            state::PLYAdded{
                .name = merged->name,
                .node_gaussians = merged->gaussian_count,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = merged->visible,
                .parent_name = parent_name,
                .is_group = false,
                .node_type = static_cast<int>(merged->type)
            }.emit();

            // Re-select the merged node if the group was selected
            if (was_selected) {
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    selected_nodes_.insert(merged_name);
                }
                ui::NodeSelected{
                    .path = merged_name,
                    .type = "PLY",
                    .metadata = {{"name", merged_name}}
                }.emit();
            }
        }

        emitSceneChanged();
        LOG_INFO("Merged group '{}' -> '{}'", name, merged_name);
    }

    void SceneManager::updateCropBoxToFitScene(const bool use_percentile) {
        if (!rendering_manager_) {
            return;
        }

        const auto* combined_model = scene_.getCombinedModel();
        if (!combined_model || combined_model->size() == 0) {
            return;
        }

        glm::vec3 min_bounds, max_bounds;
        if (!lfs::core::compute_bounds(*combined_model, min_bounds, max_bounds, 0.0f, use_percentile)) {
            return;
        }

        const glm::vec3 center = (min_bounds + max_bounds) * 0.5f;
        const glm::vec3 size = max_bounds - min_bounds;
        const glm::vec3 half_size = size * 0.5f;

        // Update all scene graph cropboxes (single source of truth)
        for (const auto* node : scene_.getNodes()) {
            if (node->type == NodeType::CROPBOX && node->cropbox) {
                auto* mutable_node = scene_.getMutableNode(node->name);
                if (mutable_node && mutable_node->cropbox) {
                    mutable_node->cropbox->min = -half_size;
                    mutable_node->cropbox->max = half_size;
                    mutable_node->local_transform = glm::translate(glm::mat4(1.0f), center);
                    mutable_node->transform_dirty = true;
                }
            }
        }

        if (rendering_manager_) {
            rendering_manager_->markDirty();
        }

        LOG_INFO("Cropbox reset: center({:.2f},{:.2f},{:.2f}) size({:.2f},{:.2f},{:.2f})",
                 center.x, center.y, center.z, size.x, size.y, size.z);
    }

    SceneManager::ClipboardEntry::HierarchyNode SceneManager::copyNodeHierarchy(const SceneNode* node) {
        ClipboardEntry::HierarchyNode result;
        result.type = node->type;
        result.local_transform = node->local_transform.get();

        if (node->cropbox) {
            result.cropbox = std::make_unique<CropBoxData>(*node->cropbox);
        }

        for (const NodeId child_id : node->children) {
            if (const auto* child = scene_.getNodeById(child_id)) {
                result.children.push_back(copyNodeHierarchy(child));
            }
        }

        return result;
    }

    void SceneManager::pasteNodeHierarchy(const ClipboardEntry::HierarchyNode& src, const NodeId parent_id) {
        for (const auto& child : src.children) {
            if (child.type == NodeType::CROPBOX && child.cropbox) {
                const NodeId cropbox_id = scene_.getOrCreateCropBoxForSplat(parent_id);
                if (cropbox_id == NULL_NODE) continue;

                const auto* cropbox_info = scene_.getNodeById(cropbox_id);
                if (!cropbox_info) continue;

                auto* cropbox_node = scene_.getMutableNode(cropbox_info->name);
                if (cropbox_node && cropbox_node->cropbox) {
                    *cropbox_node->cropbox = *child.cropbox;
                    cropbox_node->local_transform = child.local_transform;
                    cropbox_node->transform_dirty = true;
                }
            }
        }
    }

    bool SceneManager::copySelection() {
        static constexpr glm::mat4 IDENTITY{1.0f};

        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty()) {
            clipboard_.clear();
            return false;
        }

        clipboard_.clear();
        clipboard_.reserve(selected_nodes_.size());

        for (const auto& node_name : selected_nodes_) {
            const auto* node = scene_.getNode(node_name);
            if (!node || !node->model || node->model->size() == 0) continue;

            const auto& src = *node->model;
            auto cloned = std::make_unique<lfs::core::SplatData>(
                src.get_max_sh_degree(),
                src.means_raw().clone(), src.sh0_raw().clone(), src.shN_raw().clone(),
                src.scaling_raw().clone(), src.rotation_raw().clone(), src.opacity_raw().clone(),
                src.get_scene_scale());
            cloned->set_active_sh_degree(src.get_active_sh_degree());

            ClipboardEntry entry;
            entry.data = std::move(cloned);
            entry.transform = node->local_transform.get();
            entry.hierarchy = copyNodeHierarchy(node);

            clipboard_.push_back(std::move(entry));
        }

        LOG_DEBUG("Copied {} nodes to clipboard", clipboard_.size());
        return !clipboard_.empty();
    }

    std::vector<std::string> SceneManager::pasteSelection() {
        std::vector<std::string> pasted_names;
        if (clipboard_.empty()) {
            return pasted_names;
        }

        pasted_names.reserve(clipboard_.size());

        for (const auto& entry : clipboard_) {
            if (!entry.data || entry.data->size() == 0) continue;

            auto paste_data = std::make_unique<lfs::core::SplatData>(
                entry.data->get_max_sh_degree(),
                entry.data->means_raw().clone(), entry.data->sh0_raw().clone(), entry.data->shN_raw().clone(),
                entry.data->scaling_raw().clone(), entry.data->rotation_raw().clone(), entry.data->opacity_raw().clone(),
                entry.data->get_scene_scale());
            paste_data->set_active_sh_degree(entry.data->get_active_sh_degree());

            ++clipboard_counter_;
            const std::string name = std::format("Pasted_{}", clipboard_counter_);
            const size_t count = entry.data->size();
            scene_.addNode(name, std::move(paste_data));

            // Apply original transform
            static constexpr glm::mat4 IDENTITY{1.0f};
            if (entry.transform != IDENTITY) {
                scene_.setNodeTransform(name, entry.transform);
            }

            // Paste node hierarchy (cropbox)
            const auto* splat_node = scene_.getNode(name);
            if (splat_node && entry.hierarchy) {
                pasteNodeHierarchy(*entry.hierarchy, splat_node->id);
            }

            state::PLYAdded{
                .name = name,
                .node_gaussians = count,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = true,
                .parent_name = "",
                .is_group = false,
                .node_type = 0
            }.emit();

            // Emit PLYAdded for cropbox
            const auto* pasted_splat = scene_.getNode(name);
            if (pasted_splat) {
                const NodeId cropbox_id = scene_.getCropBoxForSplat(pasted_splat->id);
                if (cropbox_id != NULL_NODE) {
                    if (const auto* cropbox_node = scene_.getNodeById(cropbox_id)) {
                        state::PLYAdded{
                            .name = cropbox_node->name,
                            .node_gaussians = 0,
                            .total_gaussians = scene_.getTotalGaussianCount(),
                            .is_visible = true,
                            .parent_name = name,
                            .is_group = false,
                            .node_type = 2
                        }.emit();
                    }
                }
            }

            pasted_names.push_back(name);
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (content_type_ == ContentType::Empty && !pasted_names.empty()) {
                content_type_ = ContentType::SplatFiles;
            }
        }

        scene_.invalidateCache();
        emitSceneChanged();

        if (rendering_manager_) {
            rendering_manager_->markDirty();
        }

        LOG_DEBUG("Pasted {} nodes", pasted_names.size());
        return pasted_names;
    }

} // namespace lfs::vis