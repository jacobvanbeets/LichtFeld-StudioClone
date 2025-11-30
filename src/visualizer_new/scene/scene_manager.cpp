/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene/scene_manager.hpp"
#include "command/command_history.hpp"
#include "command/commands/crop_command.hpp"
#include "core_new/logger.hpp"
#include "core_new/splat_data_export.hpp"
#include "core_new/splat_data_transform.hpp"
#include "geometry_new/euclidean_transform.hpp"
#include "gui/panels/gizmo_toolbar.hpp"
#include "loader_new/loader.hpp"
#include "rendering/rendering_manager.hpp"
#include "training/training_manager.hpp"
#include "training_new/training_setup.hpp"
#include <glm/gtc/quaternion.hpp>
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
            removePLY(cmd.name);
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

        // Handle node selection from scene panel
        ui::NodeSelected::when([this](const auto& event) {
            if (event.type == "PLY") {
                std::lock_guard<std::mutex> lock(state_mutex_);
                if (selected_node_ != event.path) {
                    selected_node_ = event.path;
                }
            }
        });

        // Handle node deselection
        ui::NodeDeselected::when([this](const auto&) {
            clearSelection();
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
                .is_visible = true}
                .emit();

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

            // Update paths
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                splat_paths_[name] = path;
            }

            state::PLYAdded{
                .name = name,
                .node_gaussians = gaussian_count,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = is_visible}
                .emit();

            emitSceneChanged();
            updateCropBoxToFitScene(true);

            LOG_INFO("Added '{}' ({} gaussians)", name, gaussian_count);

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to add splat file: {} (path: {})", e.what(), path.string());
            throw;
        }
    }

    void SceneManager::removePLY(const std::string& name) {
        LOG_DEBUG("Removing '{}' from scene", name);

        scene_.removeNode(name);
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            splat_paths_.erase(name);
        }

        if (lfs_project_) {
            lfs_project_->removePly(name);
        }

        // If no nodes left, transition to empty
        if (scene_.getNodeCount() == 0) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            content_type_ = ContentType::Empty;
            LOG_DEBUG("No nodes remaining, transitioning to empty state");
        }

        state::PLYRemoved{.name = name}.emit();
        emitSceneChanged();

        LOG_INFO("Removed '{}' from scene", name);
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
                selected_node_ = name;
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

    void SceneManager::clearSelection() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        selected_node_.clear();
        LOG_TRACE("Cleared node selection");
    }

    std::string SceneManager::getSelectedNodeName() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return selected_node_;
    }

    bool SceneManager::hasSelectedNode() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return !selected_node_.empty() && scene_.getNode(selected_node_) != nullptr;
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
            node_name = selected_node_;
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
            node_name = selected_node_;
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
            node_name = selected_node_;
        }

        if (node_name.empty()) {
            LOG_INFO("getSelectedNodeCentroid: no node selected");
            return glm::vec3(0.0f);
        }

        // Get the node - centroid is pre-cached when model was loaded
        const auto* node = scene_.getNode(node_name);
        if (!node || !node->model) {
            LOG_INFO("getSelectedNodeCentroid: node '{}' not found or no model", node_name);
            return glm::vec3(0.0f);
        }

        return node->centroid;
    }

    void SceneManager::setSelectedNodeTransform(const glm::mat4& transform) {
        std::string node_name;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            node_name = selected_node_;
        }

        if (node_name.empty()) {
            LOG_TRACE("No node selected for transform");
            return;
        }

        scene_.setNodeTransform(node_name, transform);
        emitSceneChanged();
    }

    glm::mat4 SceneManager::getSelectedNodeTransform() const {
        std::string node_name;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            node_name = selected_node_;
        }

        if (node_name.empty()) {
            return glm::mat4(1.0f);
        }

        return scene_.getNodeTransform(node_name);
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

        std::vector<std::string> node_names;
        for (const auto* node : scene_.getVisibleNodes()) {
            node_names.push_back(node->name);
        }

        if (node_names.empty()) {
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
                static const glm::mat4 IDENTITY_MATRIX(1.0f);

                if (node->transform != IDENTITY_MATRIX) {
                    // Combine: local -> world -> box = world2bbox * node_to_world
                    const auto& world2bbox = crop_box.getworld2BBox();
                    const lfs::geometry::EuclideanTransform node_to_world(node->transform);
                    local_crop_box.setworld2BBox(world2bbox * node_to_world);
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
                    .is_visible = true
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

        auto settings = rendering_manager_->getSettings();
        const glm::vec3 center = (min_bounds + max_bounds) * 0.5f;
        const glm::vec3 size = max_bounds - min_bounds;

        settings.crop_min = -size * 0.5f;
        settings.crop_max = size * 0.5f;
        settings.crop_transform = lfs::geometry::EuclideanTransform(
            glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
            center
        );

        rendering_manager_->updateSettings(settings);

        LOG_INFO("Cropbox: center({:.2f}, {:.2f}, {:.2f}), size({:.2f}, {:.2f}, {:.2f})",
                 center.x, center.y, center.z, size.x, size.y, size.z);
    }

} // namespace lfs::vis