/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_scene.hpp"
#include <nanobind/ndarray.h>

namespace lfs::python {

    // Helper to convert glm::mat4 to nb::tuple (row-major for NumPy compatibility)
    static nb::tuple mat4_to_tuple(const glm::mat4& m) {
        nb::list rows;
        for (int i = 0; i < 4; ++i) {
            nb::list row;
            for (int j = 0; j < 4; ++j) {
                row.append(m[j][i]); // glm is column-major, so m[col][row]
            }
            rows.append(nb::tuple(row));
        }
        return nb::tuple(rows);
    }

    // Helper to convert ndarray to glm::mat4
    static glm::mat4 ndarray_to_mat4(nb::ndarray<float, nb::shape<4, 4>> arr) {
        glm::mat4 m;
        auto view = arr.view();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                m[j][i] = view(i, j); // glm is column-major
            }
        }
        return m;
    }

    // PySceneNode implementation
    nb::tuple PySceneNode::local_transform() const {
        return mat4_to_tuple(node_->local_transform.get());
    }

    void PySceneNode::set_local_transform(nb::ndarray<float, nb::shape<4, 4>> transform) {
        node_->local_transform = ndarray_to_mat4(transform);
    }

    nb::tuple PySceneNode::world_transform() const {
        return mat4_to_tuple(scene_->getWorldTransform(node_->id));
    }

    std::optional<PySplatData> PySceneNode::splat_data() {
        if (node_->type != vis::NodeType::SPLAT || !node_->model) {
            return std::nullopt;
        }
        return PySplatData(node_->model.get());
    }

    std::optional<PyPointCloud> PySceneNode::point_cloud() {
        if (node_->type != vis::NodeType::POINTCLOUD || !node_->point_cloud) {
            return std::nullopt;
        }
        return PyPointCloud(node_->point_cloud.get(), false);
    }

    std::optional<PyCropBox> PySceneNode::cropbox() {
        if (node_->type != vis::NodeType::CROPBOX || !node_->cropbox) {
            return std::nullopt;
        }
        return PyCropBox(node_->cropbox.get());
    }

    // PyScene implementation
    int32_t PyScene::add_group(const std::string& name, int32_t parent) {
        return scene_->addGroup(name, parent);
    }

    void PyScene::remove_node(const std::string& name, bool keep_children) {
        scene_->removeNode(name, keep_children);
    }

    bool PyScene::rename_node(const std::string& old_name, const std::string& new_name) {
        return scene_->renameNode(old_name, new_name);
    }

    void PyScene::reparent(int32_t node_id, int32_t new_parent_id) {
        scene_->reparent(node_id, new_parent_id);
    }

    std::optional<PySceneNode> PyScene::get_node_by_id(int32_t id) {
        auto* node = scene_->getNodeById(id);
        if (!node)
            return std::nullopt;
        return PySceneNode(node, scene_);
    }

    std::optional<PySceneNode> PyScene::get_node(const std::string& name) {
        auto* node = scene_->getMutableNode(name);
        if (!node)
            return std::nullopt;
        return PySceneNode(node, scene_);
    }

    std::vector<PySceneNode> PyScene::get_nodes() {
        std::vector<PySceneNode> result;
        for (const auto* node : scene_->getNodes()) {
            result.emplace_back(const_cast<vis::SceneNode*>(node), scene_);
        }
        return result;
    }

    std::vector<PySceneNode> PyScene::get_visible_nodes() {
        std::vector<PySceneNode> result;
        for (const auto* node : scene_->getVisibleNodes()) {
            result.emplace_back(const_cast<vis::SceneNode*>(node), scene_);
        }
        return result;
    }

    nb::tuple PyScene::get_world_transform(int32_t node_id) const {
        return mat4_to_tuple(scene_->getWorldTransform(node_id));
    }

    void PyScene::set_node_transform(const std::string& name, nb::ndarray<float, nb::shape<4, 4>> transform) {
        scene_->setNodeTransform(name, ndarray_to_mat4(transform));
    }

    std::optional<PySplatData> PyScene::combined_model() {
        auto* model = const_cast<core::SplatData*>(scene_->getCombinedModel());
        if (!model)
            return std::nullopt;
        return PySplatData(model);
    }

    std::optional<PySplatData> PyScene::training_model() {
        auto* model = scene_->getTrainingModel();
        if (!model)
            return std::nullopt;
        return PySplatData(model);
    }

    std::optional<std::tuple<std::tuple<float, float, float>, std::tuple<float, float, float>>>
    PyScene::get_node_bounds(int32_t id) const {
        glm::vec3 min_bound, max_bound;
        if (!scene_->getNodeBounds(id, min_bound, max_bound)) {
            return std::nullopt;
        }
        return std::make_tuple(
            std::make_tuple(min_bound.x, min_bound.y, min_bound.z),
            std::make_tuple(max_bound.x, max_bound.y, max_bound.z));
    }

    std::tuple<float, float, float> PyScene::get_node_bounds_center(int32_t id) const {
        auto center = scene_->getNodeBoundsCenter(id);
        return {center.x, center.y, center.z};
    }

    std::optional<PyCropBox> PyScene::get_cropbox_data(int32_t cropbox_id) {
        auto* data = scene_->getCropBoxData(cropbox_id);
        if (!data)
            return std::nullopt;
        return PyCropBox(data);
    }

    void PyScene::set_cropbox_data(int32_t cropbox_id, const PyCropBox& data) {
        // Copy the data from Python wrapper to scene
        auto* scene_data = scene_->getCropBoxData(cropbox_id);
        if (scene_data) {
            auto [min_x, min_y, min_z] = data.min();
            auto [max_x, max_y, max_z] = data.max();
            auto [r, g, b] = data.color();
            scene_data->min = {min_x, min_y, min_z};
            scene_data->max = {max_x, max_y, max_z};
            scene_data->inverse = data.inverse();
            scene_data->enabled = data.enabled();
            scene_data->color = {r, g, b};
            scene_data->line_width = data.line_width();
        }
    }

    std::optional<PyTensor> PyScene::selection_mask() const {
        auto mask = scene_->getSelectionMask();
        if (!mask || !mask->is_valid())
            return std::nullopt;
        return PyTensor(*mask, false);
    }

    uint8_t PyScene::add_selection_group(const std::string& name, std::tuple<float, float, float> color) {
        return scene_->addSelectionGroup(name, {std::get<0>(color), std::get<1>(color), std::get<2>(color)});
    }

    void PyScene::set_selection_group_color(uint8_t id, std::tuple<float, float, float> color) {
        scene_->setSelectionGroupColor(id, {std::get<0>(color), std::get<1>(color), std::get<2>(color)});
    }

    std::vector<PySelectionGroup> PyScene::selection_groups() const {
        std::vector<PySelectionGroup> result;
        for (const auto& group : scene_->getSelectionGroups()) {
            result.push_back({group.id,
                              group.name,
                              {group.color.x, group.color.y, group.color.z},
                              group.count,
                              group.locked});
        }
        return result;
    }

    PyTensor PyScene::scene_center() const {
        return PyTensor(scene_->getSceneCenter(), false);
    }

    void register_scene(nb::module_& m) {
        // NodeType enum
        nb::enum_<vis::NodeType>(m, "NodeType")
            .value("SPLAT", vis::NodeType::SPLAT)
            .value("POINTCLOUD", vis::NodeType::POINTCLOUD)
            .value("GROUP", vis::NodeType::GROUP)
            .value("CROPBOX", vis::NodeType::CROPBOX)
            .value("DATASET", vis::NodeType::DATASET)
            .value("CAMERA_GROUP", vis::NodeType::CAMERA_GROUP)
            .value("CAMERA", vis::NodeType::CAMERA)
            .value("IMAGE_GROUP", vis::NodeType::IMAGE_GROUP)
            .value("IMAGE", vis::NodeType::IMAGE);

        // SelectionGroup struct
        nb::class_<PySelectionGroup>(m, "SelectionGroup")
            .def_ro("id", &PySelectionGroup::id)
            .def_ro("name", &PySelectionGroup::name)
            .def_ro("color", &PySelectionGroup::color)
            .def_ro("count", &PySelectionGroup::count)
            .def_ro("locked", &PySelectionGroup::locked);

        // CropBox class
        nb::class_<PyCropBox>(m, "CropBox")
            .def_prop_rw("min", &PyCropBox::min, &PyCropBox::set_min)
            .def_prop_rw("max", &PyCropBox::max, &PyCropBox::set_max)
            .def_prop_rw("inverse", &PyCropBox::inverse, &PyCropBox::set_inverse)
            .def_prop_rw("enabled", &PyCropBox::enabled, &PyCropBox::set_enabled)
            .def_prop_rw("color", &PyCropBox::color, &PyCropBox::set_color)
            .def_prop_rw("line_width", &PyCropBox::line_width, &PyCropBox::set_line_width);

        // PointCloud class
        nb::class_<PyPointCloud>(m, "PointCloud")
            .def_prop_ro("means", &PyPointCloud::means)
            .def_prop_ro("colors", &PyPointCloud::colors)
            .def_prop_ro("normals", &PyPointCloud::normals)
            .def_prop_ro("sh0", &PyPointCloud::sh0)
            .def_prop_ro("shN", &PyPointCloud::shN)
            .def_prop_ro("opacity", &PyPointCloud::opacity)
            .def_prop_ro("scaling", &PyPointCloud::scaling)
            .def_prop_ro("rotation", &PyPointCloud::rotation)
            .def_prop_ro("size", &PyPointCloud::size)
            .def("is_gaussian", &PyPointCloud::is_gaussian)
            .def_prop_ro("attribute_names", &PyPointCloud::attribute_names)
            .def("normalize_colors", &PyPointCloud::normalize_colors);

        // SceneNode class
        nb::class_<PySceneNode>(m, "SceneNode")
            // Identity
            .def_prop_ro("id", &PySceneNode::id)
            .def_prop_ro("parent_id", &PySceneNode::parent_id)
            .def_prop_ro("children", &PySceneNode::children)
            .def_prop_ro("type", &PySceneNode::type)
            .def_prop_ro("name", &PySceneNode::name)
            // Transform
            .def_prop_ro("local_transform", &PySceneNode::local_transform)
            .def("set_local_transform", &PySceneNode::set_local_transform)
            .def_prop_ro("world_transform", &PySceneNode::world_transform)
            // Visibility
            .def_prop_rw("visible", &PySceneNode::visible, &PySceneNode::set_visible)
            .def_prop_rw("locked", &PySceneNode::locked, &PySceneNode::set_locked)
            // Metadata
            .def_prop_ro("gaussian_count", &PySceneNode::gaussian_count)
            .def_prop_ro("centroid", &PySceneNode::centroid)
            // Data accessors
            .def("splat_data", &PySceneNode::splat_data)
            .def("point_cloud", &PySceneNode::point_cloud)
            .def("cropbox", &PySceneNode::cropbox)
            // Camera specific
            .def_prop_ro("camera_index", &PySceneNode::camera_index)
            .def_prop_ro("camera_uid", &PySceneNode::camera_uid)
            .def_prop_ro("image_path", &PySceneNode::image_path)
            .def_prop_ro("mask_path", &PySceneNode::mask_path);

        // Scene class
        nb::class_<PyScene>(m, "Scene")
            // Node CRUD
            .def("add_group", &PyScene::add_group,
                 nb::arg("name"), nb::arg("parent") = vis::NULL_NODE)
            .def("remove_node", &PyScene::remove_node,
                 nb::arg("name"), nb::arg("keep_children") = false)
            .def("rename_node", &PyScene::rename_node,
                 nb::arg("old_name"), nb::arg("new_name"))
            .def("clear", &PyScene::clear)
            // Hierarchy
            .def("reparent", &PyScene::reparent,
                 nb::arg("node_id"), nb::arg("new_parent_id"))
            .def("root_nodes", &PyScene::root_nodes)
            // Queries
            .def("get_node_by_id", &PyScene::get_node_by_id, nb::arg("id"))
            .def("get_node", &PyScene::get_node, nb::arg("name"))
            .def("get_nodes", &PyScene::get_nodes)
            .def("get_visible_nodes", &PyScene::get_visible_nodes)
            .def("is_node_effectively_visible", &PyScene::is_node_effectively_visible,
                 nb::arg("id"))
            // Transforms
            .def("get_world_transform", &PyScene::get_world_transform, nb::arg("node_id"))
            .def("set_node_transform", &PyScene::set_node_transform,
                 nb::arg("name"), nb::arg("transform"))
            // Combined/training model
            .def("combined_model", &PyScene::combined_model)
            .def("training_model", &PyScene::training_model)
            .def("set_training_model_node", &PyScene::set_training_model_node, nb::arg("name"))
            .def_prop_ro("training_model_node_name", &PyScene::training_model_node_name)
            // Bounds
            .def("get_node_bounds", &PyScene::get_node_bounds, nb::arg("id"))
            .def("get_node_bounds_center", &PyScene::get_node_bounds_center, nb::arg("id"))
            // CropBox
            .def("get_cropbox_for_splat", &PyScene::get_cropbox_for_splat, nb::arg("splat_id"))
            .def("get_or_create_cropbox_for_splat", &PyScene::get_or_create_cropbox_for_splat,
                 nb::arg("splat_id"))
            .def("get_cropbox_data", &PyScene::get_cropbox_data, nb::arg("cropbox_id"))
            .def("set_cropbox_data", &PyScene::set_cropbox_data,
                 nb::arg("cropbox_id"), nb::arg("data"))
            // Selection
            .def_prop_ro("selection_mask", &PyScene::selection_mask)
            .def("set_selection", &PyScene::set_selection, nb::arg("indices"))
            .def("set_selection_mask", &PyScene::set_selection_mask, nb::arg("mask"))
            .def("clear_selection", &PyScene::clear_selection)
            .def("has_selection", &PyScene::has_selection)
            // Selection groups
            .def("add_selection_group", &PyScene::add_selection_group,
                 nb::arg("name"), nb::arg("color"))
            .def("remove_selection_group", &PyScene::remove_selection_group, nb::arg("id"))
            .def("rename_selection_group", &PyScene::rename_selection_group,
                 nb::arg("id"), nb::arg("name"))
            .def("set_selection_group_color", &PyScene::set_selection_group_color,
                 nb::arg("id"), nb::arg("color"))
            .def("set_selection_group_locked", &PyScene::set_selection_group_locked,
                 nb::arg("id"), nb::arg("locked"))
            .def("is_selection_group_locked", &PyScene::is_selection_group_locked, nb::arg("id"))
            .def_prop_rw("active_selection_group",
                         &PyScene::active_selection_group, &PyScene::set_active_selection_group)
            .def("selection_groups", &PyScene::selection_groups)
            .def("update_selection_group_counts", &PyScene::update_selection_group_counts)
            .def("clear_selection_group", &PyScene::clear_selection_group, nb::arg("id"))
            .def("reset_selection_state", &PyScene::reset_selection_state)
            // Training data
            .def("has_training_data", &PyScene::has_training_data)
            .def_prop_ro("scene_center", &PyScene::scene_center)
            // Counts
            .def_prop_ro("node_count", &PyScene::node_count)
            .def_prop_ro("total_gaussian_count", &PyScene::total_gaussian_count)
            .def("has_nodes", &PyScene::has_nodes)
            // Operations
            .def("apply_deleted", &PyScene::apply_deleted)
            .def("invalidate_cache", &PyScene::invalidate_cache)
            .def("duplicate_node", &PyScene::duplicate_node, nb::arg("name"))
            .def("merge_group", &PyScene::merge_group, nb::arg("group_name"));
    }

} // namespace lfs::python
