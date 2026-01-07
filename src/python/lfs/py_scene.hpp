/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "py_splat_data.hpp"
#include "py_tensor.hpp"
#include "visualizer/scene/scene.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <optional>

namespace nb = nanobind;

namespace lfs::python {

    // Forward declarations
    class PyScene;
    class PyCameraDataset;

    // Selection group view (read-only)
    struct PySelectionGroup {
        uint8_t id;
        std::string name;
        std::tuple<float, float, float> color;
        size_t count;
        bool locked;
    };

    // CropBox wrapper
    class PyCropBox {
    public:
        explicit PyCropBox(vis::CropBoxData* data) : data_(data) {
            assert(data_ != nullptr);
        }

        std::tuple<float, float, float> min() const {
            return {data_->min.x, data_->min.y, data_->min.z};
        }
        void set_min(std::tuple<float, float, float> v) {
            data_->min = {std::get<0>(v), std::get<1>(v), std::get<2>(v)};
        }

        std::tuple<float, float, float> max() const {
            return {data_->max.x, data_->max.y, data_->max.z};
        }
        void set_max(std::tuple<float, float, float> v) {
            data_->max = {std::get<0>(v), std::get<1>(v), std::get<2>(v)};
        }

        bool inverse() const { return data_->inverse; }
        void set_inverse(bool v) { data_->inverse = v; }

        bool enabled() const { return data_->enabled; }
        void set_enabled(bool v) { data_->enabled = v; }

        std::tuple<float, float, float> color() const {
            return {data_->color.x, data_->color.y, data_->color.z};
        }
        void set_color(std::tuple<float, float, float> c) {
            data_->color = {std::get<0>(c), std::get<1>(c), std::get<2>(c)};
        }

        float line_width() const { return data_->line_width; }
        void set_line_width(float w) { data_->line_width = w; }

    private:
        vis::CropBoxData* data_;
    };

    // PointCloud wrapper
    class PyPointCloud {
    public:
        explicit PyPointCloud(core::PointCloud* pc, bool owns = false)
            : pc_(pc), owns_(owns) {
            assert(pc_ != nullptr);
        }

        PyTensor means() const { return PyTensor(pc_->means, false); }
        PyTensor colors() const { return PyTensor(pc_->colors, false); }

        std::optional<PyTensor> normals() const {
            if (!pc_->normals.is_valid()) return std::nullopt;
            return PyTensor(pc_->normals, false);
        }
        std::optional<PyTensor> sh0() const {
            if (!pc_->sh0.is_valid()) return std::nullopt;
            return PyTensor(pc_->sh0, false);
        }
        std::optional<PyTensor> shN() const {
            if (!pc_->shN.is_valid()) return std::nullopt;
            return PyTensor(pc_->shN, false);
        }
        std::optional<PyTensor> opacity() const {
            if (!pc_->opacity.is_valid()) return std::nullopt;
            return PyTensor(pc_->opacity, false);
        }
        std::optional<PyTensor> scaling() const {
            if (!pc_->scaling.is_valid()) return std::nullopt;
            return PyTensor(pc_->scaling, false);
        }
        std::optional<PyTensor> rotation() const {
            if (!pc_->rotation.is_valid()) return std::nullopt;
            return PyTensor(pc_->rotation, false);
        }

        int64_t size() const { return pc_->size(); }
        bool is_gaussian() const { return pc_->is_gaussian(); }
        std::vector<std::string> attribute_names() const { return pc_->attribute_names; }

        void normalize_colors() { pc_->normalize_colors(); }

        core::PointCloud* data() { return pc_; }
        const core::PointCloud* data() const { return pc_; }

    private:
        core::PointCloud* pc_;
        bool owns_;
    };

    // Scene node wrapper
    class PySceneNode {
    public:
        PySceneNode(vis::SceneNode* node, vis::Scene* scene)
            : node_(node), scene_(scene) {
            assert(node_ != nullptr);
            assert(scene_ != nullptr);
        }

        // Identity (read-only)
        int32_t id() const { return node_->id; }
        int32_t parent_id() const { return node_->parent_id; }
        std::vector<int32_t> children() const { return node_->children; }
        uint8_t type() const { return static_cast<uint8_t>(node_->type); }
        std::string name() const { return node_->name; }

        // Transform
        nb::tuple local_transform() const;
        void set_local_transform(nb::ndarray<float, nb::shape<4, 4>> transform);
        nb::tuple world_transform() const;

        // Visibility
        bool visible() const { return node_->visible.get(); }
        void set_visible(bool v) { node_->visible = v; }

        // Locked
        bool locked() const { return node_->locked.get(); }
        void set_locked(bool l) { node_->locked = l; }

        // Metadata
        size_t gaussian_count() const { return node_->gaussian_count; }
        std::tuple<float, float, float> centroid() const {
            return {node_->centroid.x, node_->centroid.y, node_->centroid.z};
        }

        // Data accessors
        std::optional<PySplatData> splat_data();
        std::optional<PyPointCloud> point_cloud();
        std::optional<PyCropBox> cropbox();

        // Camera node specific
        int camera_index() const { return node_->camera_index; }
        int camera_uid() const { return node_->camera_uid; }
        std::string image_path() const { return node_->image_path; }
        std::string mask_path() const { return node_->mask_path; }

    private:
        vis::SceneNode* node_;
        vis::Scene* scene_;
    };

    // Main scene wrapper
    class PyScene {
    public:
        explicit PyScene(vis::Scene* scene) : scene_(scene) {
            assert(scene_ != nullptr);
        }

        // Node CRUD
        int32_t add_group(const std::string& name, int32_t parent = vis::NULL_NODE);
        void remove_node(const std::string& name, bool keep_children = false);
        bool rename_node(const std::string& old_name, const std::string& new_name);
        void clear() { scene_->clear(); }

        // Hierarchy
        void reparent(int32_t node_id, int32_t new_parent_id);
        std::vector<int32_t> root_nodes() const { return scene_->getRootNodes(); }

        // Queries
        std::optional<PySceneNode> get_node_by_id(int32_t id);
        std::optional<PySceneNode> get_node(const std::string& name);
        std::vector<PySceneNode> get_nodes();
        std::vector<PySceneNode> get_visible_nodes();
        bool is_node_effectively_visible(int32_t id) const {
            return scene_->isNodeEffectivelyVisible(id);
        }

        // Transforms
        nb::tuple get_world_transform(int32_t node_id) const;
        void set_node_transform(const std::string& name, nb::ndarray<float, nb::shape<4, 4>> transform);

        // Combined model
        std::optional<PySplatData> combined_model();
        std::optional<PySplatData> training_model();
        void set_training_model_node(const std::string& name) {
            scene_->setTrainingModelNode(name);
        }
        std::string training_model_node_name() const {
            return scene_->getTrainingModelNodeName();
        }

        // Bounds
        std::optional<std::tuple<std::tuple<float,float,float>, std::tuple<float,float,float>>>
            get_node_bounds(int32_t id) const;
        std::tuple<float, float, float> get_node_bounds_center(int32_t id) const;

        // CropBox operations
        int32_t get_cropbox_for_splat(int32_t splat_id) const {
            return scene_->getCropBoxForSplat(splat_id);
        }
        int32_t get_or_create_cropbox_for_splat(int32_t splat_id) {
            return scene_->getOrCreateCropBoxForSplat(splat_id);
        }
        std::optional<PyCropBox> get_cropbox_data(int32_t cropbox_id);
        void set_cropbox_data(int32_t cropbox_id, const PyCropBox& data);

        // Selection
        std::optional<PyTensor> selection_mask() const;
        void set_selection(const std::vector<size_t>& indices) {
            scene_->setSelection(indices);
        }
        void set_selection_mask(const PyTensor& mask) {
            scene_->setSelectionMask(std::make_shared<core::Tensor>(mask.tensor()));
        }
        void clear_selection() { scene_->clearSelection(); }
        bool has_selection() const { return scene_->hasSelection(); }

        // Selection groups
        uint8_t add_selection_group(const std::string& name, std::tuple<float,float,float> color);
        void remove_selection_group(uint8_t id) { scene_->removeSelectionGroup(id); }
        void rename_selection_group(uint8_t id, const std::string& name) {
            scene_->renameSelectionGroup(id, name);
        }
        void set_selection_group_color(uint8_t id, std::tuple<float,float,float> color);
        void set_selection_group_locked(uint8_t id, bool locked) {
            scene_->setSelectionGroupLocked(id, locked);
        }
        bool is_selection_group_locked(uint8_t id) const {
            return scene_->isSelectionGroupLocked(id);
        }
        void set_active_selection_group(uint8_t id) {
            scene_->setActiveSelectionGroup(id);
        }
        uint8_t active_selection_group() const {
            return scene_->getActiveSelectionGroup();
        }
        std::vector<PySelectionGroup> selection_groups() const;
        void update_selection_group_counts() { scene_->updateSelectionGroupCounts(); }
        void clear_selection_group(uint8_t id) { scene_->clearSelectionGroup(id); }
        void reset_selection_state() { scene_->resetSelectionState(); }

        // Training data
        bool has_training_data() const { return scene_->hasTrainingData(); }
        PyTensor scene_center() const;

        // Counts
        size_t node_count() const { return scene_->getNodeCount(); }
        size_t total_gaussian_count() const { return scene_->getTotalGaussianCount(); }
        bool has_nodes() const { return scene_->hasNodes(); }

        // Operations
        size_t apply_deleted() { return scene_->applyDeleted(); }
        void invalidate_cache() { scene_->invalidateCache(); }
        std::string duplicate_node(const std::string& name) {
            return scene_->duplicateNode(name);
        }
        std::string merge_group(const std::string& group_name) {
            return scene_->mergeGroup(group_name);
        }

        // Access underlying scene (for internal use)
        vis::Scene* scene() { return scene_; }
        const vis::Scene* scene() const { return scene_; }

    private:
        vis::Scene* scene_;
    };

    // Register scene classes with nanobind module
    void register_scene(nb::module_& m);

} // namespace lfs::python
