/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_cameras.hpp"

namespace lfs::python {

    void register_cameras(nb::module_& m) {
        // Camera class
        nb::class_<PyCamera>(m, "Camera")
            // Intrinsics
            .def_prop_ro("focal_x", &PyCamera::focal_x)
            .def_prop_ro("focal_y", &PyCamera::focal_y)
            .def_prop_ro("center_x", &PyCamera::center_x)
            .def_prop_ro("center_y", &PyCamera::center_y)
            .def_prop_ro("fov_x", &PyCamera::fov_x)
            .def_prop_ro("fov_y", &PyCamera::fov_y)
            // Image info
            .def_prop_ro("image_width", &PyCamera::image_width)
            .def_prop_ro("image_height", &PyCamera::image_height)
            .def_prop_ro("camera_width", &PyCamera::camera_width)
            .def_prop_ro("camera_height", &PyCamera::camera_height)
            .def_prop_ro("image_name", &PyCamera::image_name)
            .def_prop_ro("image_path", &PyCamera::image_path)
            .def_prop_ro("mask_path", &PyCamera::mask_path)
            .def_prop_ro("has_mask", &PyCamera::has_mask)
            .def_prop_ro("uid", &PyCamera::uid)
            // Transforms
            .def_prop_ro("R", &PyCamera::R, "Rotation matrix [3, 3]")
            .def_prop_ro("T", &PyCamera::T, "Translation vector [3]")
            .def_prop_ro("K", &PyCamera::K, "Intrinsic matrix [3, 3]")
            .def_prop_ro("world_view_transform", &PyCamera::world_view_transform,
                         "World-to-view transform [4, 4]")
            .def_prop_ro("cam_position", &PyCamera::cam_position,
                         "Camera position in world space [3]")
            // Load methods
            .def("load_image", &PyCamera::load_image,
                 nb::arg("resize_factor") = 1, nb::arg("max_width") = 3840,
                 "Load image as tensor [C, H, W] on CUDA")
            .def("load_mask", &PyCamera::load_mask,
                 nb::arg("resize_factor") = 1, nb::arg("max_width") = 3840,
                 nb::arg("invert") = false, nb::arg("threshold") = 0.5f,
                 "Load mask as tensor [1, H, W] on CUDA");

        // CameraDataset class
        nb::class_<PyCameraDataset>(m, "CameraDataset")
            .def("__len__", &PyCameraDataset::size)
            .def("__getitem__", &PyCameraDataset::get, nb::arg("index"))
            .def("get_camera_by_filename", &PyCameraDataset::get_camera_by_filename,
                 nb::arg("filename"))
            .def("cameras", &PyCameraDataset::cameras,
                 "Get all cameras as a list")
            .def("set_resize_factor", &PyCameraDataset::set_resize_factor,
                 nb::arg("factor"))
            .def("set_max_width", &PyCameraDataset::set_max_width,
                 nb::arg("width"));
    }

} // namespace lfs::python
