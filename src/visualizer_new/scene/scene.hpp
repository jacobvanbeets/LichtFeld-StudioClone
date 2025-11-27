/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lfs::vis {

    class Scene {
    public:
        struct Node {
            std::string name;
            std::unique_ptr<lfs::core::SplatData> model;
            glm::mat4 transform{1.0f};
            bool visible = true;
            size_t gaussian_count = 0;
            glm::vec3 centroid{0.0f};  // Cached centroid, computed once when model is loaded
        };

        Scene() = default;
        ~Scene() = default;

        // Delete copy operations
        Scene(const Scene&) = delete;
        Scene& operator=(const Scene&) = delete;

        // Allow move operations
        Scene(Scene&&) = default;
        Scene& operator=(Scene&&) = default;

        // Node management
        void addNode(const std::string& name, std::unique_ptr<lfs::core::SplatData> model);
        void removeNode(const std::string& name);
        void replaceNodeModel(const std::string& name, std::unique_ptr<lfs::core::SplatData> model);
        void setNodeVisibility(const std::string& name, bool visible);
        void setNodeTransform(const std::string& name, const glm::mat4& transform);
        glm::mat4 getNodeTransform(const std::string& name) const;
        bool renameNode(const std::string& old_name, const std::string& new_name);
        void clear();
        std::pair<std::string, std::string> cycleVisibilityWithNames();

        // Get combined model for rendering
        const lfs::core::SplatData* getCombinedModel() const;

        // Get transforms for visible nodes (for kernel-based transform)
        std::vector<glm::mat4> getVisibleNodeTransforms() const;

        // Get per-Gaussian transform indices tensor (for kernel-based transform)
        // Returns nullptr if no transforms needed (single node with identity transform)
        std::shared_ptr<lfs::core::Tensor> getTransformIndices() const;

        // Selection mask for highlighting selected Gaussians
        // Returns nullptr if no selection (all zeros = no selection)
        std::shared_ptr<lfs::core::Tensor> getSelectionMask() const;

        // Set selection for Gaussians (indices into combined model)
        void setSelection(const std::vector<size_t>& selected_indices);

        // Set selection mask directly from GPU tensor (for GPU-based brush selection)
        void setSelectionMask(std::shared_ptr<lfs::core::Tensor> mask);

        // Clear all selection
        void clearSelection();

        // Check if any Gaussians are selected
        bool hasSelection() const;

        // Direct queries
        size_t getNodeCount() const { return nodes_.size(); }
        size_t getTotalGaussianCount() const;
        std::vector<const Node*> getNodes() const;
        const Node* getNode(const std::string& name) const;
        Node* getMutableNode(const std::string& name);
        bool hasNodes() const { return !nodes_.empty(); }

        // Get visible nodes for split view
        std::vector<const Node*> getVisibleNodes() const;

    private:
        std::vector<Node> nodes_;

        // Caching for combined model
        mutable std::unique_ptr<lfs::core::SplatData> cached_combined_;
        mutable std::shared_ptr<lfs::core::Tensor> cached_transform_indices_;
        mutable std::vector<glm::mat4> cached_transforms_;
        mutable bool cache_valid_ = false;

        // Selection mask (per-Gaussian, uint8, 1=selected)
        mutable std::shared_ptr<lfs::core::Tensor> selection_mask_;
        mutable bool has_selection_ = false;

        void invalidateCache() { cache_valid_ = false; }
        void rebuildCacheIfNeeded() const;
    };

} // namespace lfs::vis