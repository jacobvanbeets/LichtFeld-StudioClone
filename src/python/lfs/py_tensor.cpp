/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_tensor.hpp"
#include "core/logger.hpp"

#include <cuda_runtime.h>
#include <nanobind/stl/optional.h>
#include <sstream>

namespace nb = nanobind;

namespace lfs::python {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;

    PyTensor::PyTensor(Tensor tensor, bool owns_data)
        : tensor_(std::move(tensor)),
          owns_data_(owns_data) {}

    nb::tuple PyTensor::shape() const {
        const auto& dims = tensor_.shape().dims();
        nb::list shape_list;
        for (size_t d : dims) {
            shape_list.append(static_cast<int64_t>(d));
        }
        return nb::tuple(shape_list);
    }

    size_t PyTensor::ndim() const {
        return tensor_.shape().rank();
    }

    size_t PyTensor::numel() const {
        return tensor_.numel();
    }

    std::string PyTensor::device() const {
        return tensor_.device() == Device::CUDA ? "cuda" : "cpu";
    }

    std::string PyTensor::dtype() const {
        switch (tensor_.dtype()) {
        case DataType::Float32: return "float32";
        case DataType::Float16: return "float16";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::UInt8: return "uint8";
        case DataType::Bool: return "bool";
        default: return "unknown";
        }
    }

    bool PyTensor::is_contiguous() const {
        return tensor_.is_contiguous();
    }

    bool PyTensor::is_cuda() const {
        return tensor_.device() == Device::CUDA;
    }

    size_t PyTensor::size(int dim) const {
        const int resolved = dim < 0 ? static_cast<int>(tensor_.shape().rank()) + dim : dim;
        if (resolved < 0 || resolved >= static_cast<int>(tensor_.shape().rank())) {
            throw std::out_of_range("Dimension out of range");
        }
        return tensor_.shape()[static_cast<size_t>(resolved)];
    }

    PyTensor PyTensor::clone() const {
        return PyTensor(tensor_.clone());
    }

    PyTensor PyTensor::cpu() const {
        if (tensor_.device() == Device::CPU) {
            return PyTensor(tensor_);
        }
        return PyTensor(tensor_.cpu());
    }

    PyTensor PyTensor::cuda() const {
        if (tensor_.device() == Device::CUDA) {
            return PyTensor(tensor_);
        }
        return PyTensor(tensor_.cuda());
    }

    PyTensor PyTensor::contiguous() const {
        if (tensor_.is_contiguous()) {
            return PyTensor(tensor_);
        }
        return PyTensor(tensor_.contiguous());
    }

    void PyTensor::sync() const {
        if (tensor_.device() == Device::CUDA) {
            cudaDeviceSynchronize();
        }
    }

    float PyTensor::item() const {
        if (tensor_.numel() != 1) {
            throw std::runtime_error("item() requires a tensor with exactly 1 element");
        }
        return tensor_.item<float>();
    }

    int64_t PyTensor::item_int() const {
        if (tensor_.numel() != 1) {
            throw std::runtime_error("item() requires a tensor with exactly 1 element");
        }
        if (tensor_.dtype() == DataType::Int32) {
            return static_cast<int64_t>(tensor_.item<int>());
        } else if (tensor_.dtype() == DataType::Int64) {
            return tensor_.item<int64_t>();
        }
        return static_cast<int64_t>(tensor_.item<float>());
    }

    bool PyTensor::item_bool() const {
        if (tensor_.numel() != 1) {
            throw std::runtime_error("item() requires a tensor with exactly 1 element");
        }
        if (tensor_.dtype() == DataType::Bool) {
            return tensor_.item<unsigned char>() != 0;
        }
        return tensor_.item<float>() != 0.0f;
    }

    nb::object PyTensor::numpy(bool copy) const {
        Tensor cpu_tensor = tensor_.device() == Device::CUDA ? tensor_.cpu() : tensor_;

        // Ensure contiguous
        if (!cpu_tensor.is_contiguous()) {
            cpu_tensor = cpu_tensor.contiguous();
        }

        const auto& dims = cpu_tensor.shape().dims();
        const std::vector<size_t> shape_vec(dims.begin(), dims.end());

        // Compute strides in bytes
        std::vector<int64_t> strides(dims.size());
        size_t elem_size = 4;
        switch (cpu_tensor.dtype()) {
        case DataType::Float32: elem_size = 4; break;
        case DataType::Float16: elem_size = 2; break;
        case DataType::Int32: elem_size = 4; break;
        case DataType::Int64: elem_size = 8; break;
        case DataType::UInt8:
        case DataType::Bool: elem_size = 1; break;
        }

        if (!strides.empty()) {
            strides.back() = static_cast<int64_t>(elem_size);
            for (int i = static_cast<int>(strides.size()) - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * static_cast<int64_t>(dims[i + 1]);
            }
        }

        if (copy) {
            // Copy mode: allocate new buffer and copy data
            const size_t total_bytes = cpu_tensor.numel() * elem_size;
            void* const buffer = std::malloc(total_bytes);
            if (!buffer) {
                throw std::bad_alloc();
            }
            std::memcpy(buffer, cpu_tensor.data_ptr(), total_bytes);

            // Create owner capsule for memory management
            nb::capsule owner(buffer, [](void* p) noexcept { std::free(p); });

            switch (cpu_tensor.dtype()) {
            case DataType::Float32:
                return nb::cast(nb::ndarray<nb::numpy, float>(
                    buffer, shape_vec.size(), shape_vec.data(), owner,
                    strides.data()));
            case DataType::Int32:
                return nb::cast(nb::ndarray<nb::numpy, int32_t>(
                    buffer, shape_vec.size(), shape_vec.data(), owner,
                    strides.data()));
            case DataType::Int64:
                return nb::cast(nb::ndarray<nb::numpy, int64_t>(
                    buffer, shape_vec.size(), shape_vec.data(), owner,
                    strides.data()));
            case DataType::UInt8:
            case DataType::Bool:
                return nb::cast(nb::ndarray<nb::numpy, uint8_t>(
                    buffer, shape_vec.size(), shape_vec.data(), owner,
                    strides.data()));
            default:
                std::free(buffer);
                throw std::runtime_error("Unsupported dtype for numpy conversion");
            }
        } else {
            // Zero-copy mode: use capsule to hold tensor reference
            auto* const tensor_copy = new Tensor(cpu_tensor);
            const nb::capsule owner(tensor_copy, [](void* p) noexcept { delete static_cast<Tensor*>(p); });

            void* const data = tensor_copy->data_ptr();

            switch (cpu_tensor.dtype()) {
            case DataType::Float32:
                return nb::cast(nb::ndarray<nb::numpy, float>(
                    data, shape_vec.size(), shape_vec.data(), owner,
                    strides.data()));
            case DataType::Int32:
                return nb::cast(nb::ndarray<nb::numpy, int32_t>(
                    data, shape_vec.size(), shape_vec.data(), owner,
                    strides.data()));
            case DataType::Int64:
                return nb::cast(nb::ndarray<nb::numpy, int64_t>(
                    data, shape_vec.size(), shape_vec.data(), owner,
                    strides.data()));
            case DataType::UInt8:
            case DataType::Bool:
                return nb::cast(nb::ndarray<nb::numpy, uint8_t>(
                    data, shape_vec.size(), shape_vec.data(), owner,
                    strides.data()));
            default:
                throw std::runtime_error("Unsupported dtype for numpy conversion");
            }
        }
    }

    PyTensor PyTensor::from_numpy(nb::ndarray<> arr, bool copy) {
        // Get shape
        std::vector<size_t> shape_vec;
        for (size_t i = 0; i < arr.ndim(); ++i) {
            shape_vec.push_back(arr.shape(i));
        }
        const TensorShape shape(shape_vec);

        // Determine dtype
        DataType dtype = DataType::Float32;
        const auto nb_dtype = arr.dtype();
        if (nb_dtype == nb::dtype<float>()) {
            dtype = DataType::Float32;
        } else if (nb_dtype == nb::dtype<int32_t>()) {
            dtype = DataType::Int32;
        } else if (nb_dtype == nb::dtype<int64_t>()) {
            dtype = DataType::Int64;
        } else if (nb_dtype == nb::dtype<uint8_t>()) {
            dtype = DataType::UInt8;
        } else if (nb_dtype == nb::dtype<bool>()) {
            dtype = DataType::Bool;
        } else {
            throw std::runtime_error("Unsupported numpy dtype");
        }

        // Create CPU tensor and copy data
        Tensor tensor = Tensor::empty(shape, Device::CPU, dtype, false);

        size_t elem_size = 4;
        switch (dtype) {
        case DataType::Float32: elem_size = 4; break;
        case DataType::Int32: elem_size = 4; break;
        case DataType::Int64: elem_size = 8; break;
        case DataType::UInt8:
        case DataType::Bool: elem_size = 1; break;
        default: break;
        }

        std::memcpy(tensor.data_ptr(), arr.data(), shape.elements() * elem_size);

        return PyTensor(std::move(tensor));
    }

    // Slicing
    PyTensor::SliceInfo PyTensor::parse_slice(const nb::slice& sl, size_t dim_size) const {
        SliceInfo info;
        auto [start, stop, step, count] = sl.compute(dim_size);
        info.start = start;
        info.stop = stop;
        info.step = step;
        return info;
    }

    PyTensor PyTensor::getitem(const nb::object& key) const {
        // Single integer index
        if (nb::isinstance<nb::int_>(key)) {
            int64_t idx = nb::cast<int64_t>(key);
            if (idx < 0) {
                idx += static_cast<int64_t>(tensor_.shape()[0]);
            }
            if (idx < 0 || idx >= static_cast<int64_t>(tensor_.shape()[0])) {
                throw std::out_of_range("Index out of range");
            }
            return PyTensor(tensor_.slice(0, static_cast<size_t>(idx), static_cast<size_t>(idx + 1)).squeeze(0));
        }

        // Single slice
        if (nb::isinstance<nb::slice>(key)) {
            auto sl = nb::cast<nb::slice>(key);
            SliceInfo info = parse_slice(sl, tensor_.shape()[0]);

            if (info.step != 1) {
                throw std::runtime_error("Step != 1 not yet supported");
            }

            return PyTensor(tensor_.slice(0, static_cast<size_t>(info.start), static_cast<size_t>(info.stop)));
        }

        // Tuple of indices/slices
        if (nb::isinstance<nb::tuple>(key)) {
            auto tup = nb::cast<nb::tuple>(key);
            Tensor result = tensor_;

            // Track dimension offset due to squeezed dimensions
            int dim_offset = 0;

            for (size_t i = 0; i < tup.size(); ++i) {
                int current_dim = static_cast<int>(i) - dim_offset;
                nb::object item = tup[i];

                if (nb::isinstance<nb::int_>(item)) {
                    int64_t idx = nb::cast<int64_t>(item);
                    if (idx < 0) {
                        idx += static_cast<int64_t>(result.shape()[current_dim]);
                    }
                    result = result.slice(current_dim, static_cast<size_t>(idx), static_cast<size_t>(idx + 1)).squeeze(current_dim);
                    dim_offset++;
                } else if (nb::isinstance<nb::slice>(item)) {
                    auto sl = nb::cast<nb::slice>(item);
                    SliceInfo info = parse_slice(sl, result.shape()[current_dim]);

                    if (info.step != 1) {
                        throw std::runtime_error("Step != 1 not yet supported");
                    }

                    result = result.slice(current_dim, static_cast<size_t>(info.start), static_cast<size_t>(info.stop));
                }
            }

            return PyTensor(result);
        }

        // Boolean mask
        if (nb::isinstance<PyTensor>(key)) {
            auto mask_tensor = nb::cast<PyTensor>(key);
            if (mask_tensor.tensor().dtype() != DataType::Bool) {
                throw std::runtime_error("Mask must be a boolean tensor");
            }
            return PyTensor(tensor_.masked_select(mask_tensor.tensor()));
        }

        throw std::runtime_error("Unsupported index type");
    }

    void PyTensor::setitem(const nb::object& key, const nb::object& value) {
        // Get the value tensor
        Tensor val_tensor;
        if (nb::isinstance<PyTensor>(value)) {
            val_tensor = nb::cast<PyTensor>(value).tensor();
        } else if (nb::isinstance<nb::float_>(value) || nb::isinstance<nb::int_>(value)) {
            float scalar = nb::cast<float>(value);
            // Create scalar tensor
            val_tensor = Tensor::full({1}, scalar, tensor_.device(), tensor_.dtype());
        } else {
            throw std::runtime_error("Unsupported value type for setitem");
        }

        // Single integer index
        if (nb::isinstance<nb::int_>(key)) {
            int64_t idx = nb::cast<int64_t>(key);
            if (idx < 0) {
                idx += static_cast<int64_t>(tensor_.shape()[0]);
            }
            if (idx < 0 || idx >= static_cast<int64_t>(tensor_.shape()[0])) {
                throw std::out_of_range("Index out of range");
            }
            tensor_.slice(0, static_cast<size_t>(idx), static_cast<size_t>(idx + 1)).copy_from(val_tensor);
            return;
        }

        // Single slice
        if (nb::isinstance<nb::slice>(key)) {
            auto sl = nb::cast<nb::slice>(key);
            SliceInfo info = parse_slice(sl, tensor_.shape()[0]);

            if (info.step != 1) {
                throw std::runtime_error("Step != 1 not yet supported");
            }

            tensor_.slice(0, static_cast<size_t>(info.start), static_cast<size_t>(info.stop)).copy_from(val_tensor);
            return;
        }

        throw std::runtime_error("Unsupported index type for setitem");
    }

    // Arithmetic operators
    PyTensor PyTensor::add(const PyTensor& other) const {
        return PyTensor(tensor_.add(other.tensor_));
    }

    PyTensor PyTensor::add_scalar(float scalar) const {
        return PyTensor(tensor_.add(scalar));
    }

    PyTensor PyTensor::sub(const PyTensor& other) const {
        return PyTensor(tensor_.sub(other.tensor_));
    }

    PyTensor PyTensor::sub_scalar(float scalar) const {
        return PyTensor(tensor_.sub(scalar));
    }

    PyTensor PyTensor::rsub_scalar(float scalar) const {
        return PyTensor(Tensor::full(tensor_.shape(), scalar, tensor_.device(), tensor_.dtype()).sub(tensor_));
    }

    PyTensor PyTensor::mul(const PyTensor& other) const {
        return PyTensor(tensor_.mul(other.tensor_));
    }

    PyTensor PyTensor::mul_scalar(float scalar) const {
        return PyTensor(tensor_.mul(scalar));
    }

    PyTensor PyTensor::div(const PyTensor& other) const {
        return PyTensor(tensor_.div(other.tensor_));
    }

    PyTensor PyTensor::div_scalar(float scalar) const {
        return PyTensor(tensor_.div(scalar));
    }

    PyTensor PyTensor::rdiv_scalar(float scalar) const {
        return PyTensor(Tensor::full(tensor_.shape(), scalar, tensor_.device(), tensor_.dtype()).div(tensor_));
    }

    PyTensor PyTensor::neg() const {
        return PyTensor(tensor_.neg());
    }

    PyTensor PyTensor::abs() const {
        return PyTensor(tensor_.abs());
    }

    // In-place arithmetic
    PyTensor& PyTensor::iadd(const PyTensor& other) {
        tensor_.add_(other.tensor_);
        return *this;
    }

    PyTensor& PyTensor::iadd_scalar(float scalar) {
        tensor_.add_(scalar);
        return *this;
    }

    PyTensor& PyTensor::isub(const PyTensor& other) {
        tensor_.sub_(other.tensor_);
        return *this;
    }

    PyTensor& PyTensor::isub_scalar(float scalar) {
        tensor_.sub_(scalar);
        return *this;
    }

    PyTensor& PyTensor::imul(const PyTensor& other) {
        tensor_.mul_(other.tensor_);
        return *this;
    }

    PyTensor& PyTensor::imul_scalar(float scalar) {
        tensor_.mul_(scalar);
        return *this;
    }

    PyTensor& PyTensor::idiv(const PyTensor& other) {
        tensor_.div_(other.tensor_);
        return *this;
    }

    PyTensor& PyTensor::idiv_scalar(float scalar) {
        tensor_.div_(scalar);
        return *this;
    }

    // Comparison operators
    PyTensor PyTensor::eq(const PyTensor& other) const {
        return PyTensor(tensor_.eq(other.tensor_));
    }

    PyTensor PyTensor::eq_scalar(float scalar) const {
        return PyTensor(tensor_.eq(scalar));
    }

    PyTensor PyTensor::ne(const PyTensor& other) const {
        return PyTensor(tensor_.ne(other.tensor_));
    }

    PyTensor PyTensor::ne_scalar(float scalar) const {
        return PyTensor(tensor_.ne(scalar));
    }

    PyTensor PyTensor::lt(const PyTensor& other) const {
        return PyTensor(tensor_.lt(other.tensor_));
    }

    PyTensor PyTensor::lt_scalar(float scalar) const {
        return PyTensor(tensor_.lt(scalar));
    }

    PyTensor PyTensor::le(const PyTensor& other) const {
        return PyTensor(tensor_.le(other.tensor_));
    }

    PyTensor PyTensor::le_scalar(float scalar) const {
        return PyTensor(tensor_.le(scalar));
    }

    PyTensor PyTensor::gt(const PyTensor& other) const {
        return PyTensor(tensor_.gt(other.tensor_));
    }

    PyTensor PyTensor::gt_scalar(float scalar) const {
        return PyTensor(tensor_.gt(scalar));
    }

    PyTensor PyTensor::ge(const PyTensor& other) const {
        return PyTensor(tensor_.ge(other.tensor_));
    }

    PyTensor PyTensor::ge_scalar(float scalar) const {
        return PyTensor(tensor_.ge(scalar));
    }

    // Logical operators
    PyTensor PyTensor::logical_and(const PyTensor& other) const {
        return PyTensor(tensor_.logical_and(other.tensor_));
    }

    PyTensor PyTensor::logical_or(const PyTensor& other) const {
        return PyTensor(tensor_.logical_or(other.tensor_));
    }

    PyTensor PyTensor::logical_not() const {
        return PyTensor(tensor_.logical_not());
    }

    // Reduction operations
    PyTensor PyTensor::sum(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.sum(*dim, keepdim));
        }
        return PyTensor(tensor_.sum());
    }

    PyTensor PyTensor::mean(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.mean(*dim, keepdim));
        }
        return PyTensor(tensor_.mean());
    }

    PyTensor PyTensor::max(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.max(*dim, keepdim));
        }
        return PyTensor(tensor_.max());
    }

    PyTensor PyTensor::min(std::optional<int> dim, bool keepdim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.min(*dim, keepdim));
        }
        return PyTensor(tensor_.min());
    }

    float PyTensor::sum_scalar() const {
        return tensor_.sum_scalar();
    }

    float PyTensor::mean_scalar() const {
        return tensor_.mean_scalar();
    }

    float PyTensor::max_scalar() const {
        return tensor_.max_scalar();
    }

    float PyTensor::min_scalar() const {
        return tensor_.min_scalar();
    }

    // Shape operations
    PyTensor PyTensor::reshape(const std::vector<int64_t>& new_shape) const {
        std::vector<int> shape_vec;
        shape_vec.reserve(new_shape.size());

        // Handle -1 for inferred dimension
        int64_t infer_idx = -1;
        int known_size = 1;
        for (size_t i = 0; i < new_shape.size(); ++i) {
            if (new_shape[i] == -1) {
                if (infer_idx != -1) {
                    throw std::runtime_error("Only one dimension can be inferred");
                }
                infer_idx = static_cast<int64_t>(i);
                shape_vec.push_back(0); // Placeholder
            } else {
                shape_vec.push_back(static_cast<int>(new_shape[i]));
                known_size *= static_cast<int>(new_shape[i]);
            }
        }

        if (infer_idx != -1) {
            shape_vec[static_cast<size_t>(infer_idx)] = static_cast<int>(tensor_.numel()) / known_size;
        }

        return PyTensor(tensor_.reshape(shape_vec));
    }

    PyTensor PyTensor::view(const std::vector<int64_t>& new_shape) const {
        return reshape(new_shape);
    }

    PyTensor PyTensor::squeeze(std::optional<int> dim) const {
        if (dim.has_value()) {
            return PyTensor(tensor_.squeeze(*dim));
        }
        return PyTensor(tensor_.squeeze());
    }

    PyTensor PyTensor::unsqueeze(int dim) const {
        return PyTensor(tensor_.unsqueeze(dim));
    }

    PyTensor PyTensor::transpose(int dim0, int dim1) const {
        return PyTensor(tensor_.transpose(dim0, dim1));
    }

    PyTensor PyTensor::permute(const std::vector<int>& dims) const {
        return PyTensor(tensor_.permute(dims));
    }

    PyTensor PyTensor::flatten(int start_dim, int end_dim) const {
        return PyTensor(tensor_.flatten(start_dim, end_dim));
    }

    std::string PyTensor::repr() const {
        std::ostringstream oss;
        oss << "Tensor(shape=" << tensor_.shape().str()
            << ", dtype=" << dtype()
            << ", device=" << device() << ")";
        return oss.str();
    }

    void register_tensor(nb::module_& m) {
        nb::class_<PyTensor>(m, "Tensor")
            .def(nb::init<>())

            // Properties
            .def_prop_ro("shape", &PyTensor::shape, "Tensor shape as tuple")
            .def_prop_ro("ndim", &PyTensor::ndim, "Number of dimensions")
            .def_prop_ro("numel", &PyTensor::numel, "Total number of elements")
            .def_prop_ro("device", &PyTensor::device, "Device: 'cpu' or 'cuda'")
            .def_prop_ro("dtype", &PyTensor::dtype, "Data type")
            .def_prop_ro("is_contiguous", &PyTensor::is_contiguous, "Whether memory is contiguous")
            .def_prop_ro("is_cuda", &PyTensor::is_cuda, "Whether tensor is on CUDA")

            // Memory operations
            .def("clone", &PyTensor::clone, "Deep copy of tensor")
            .def("cpu", &PyTensor::cpu, "Move tensor to CPU")
            .def("cuda", &PyTensor::cuda, "Move tensor to CUDA")
            .def("contiguous", &PyTensor::contiguous, "Make tensor contiguous")
            .def("sync", &PyTensor::sync, "Synchronize CUDA stream")
            .def("size", &PyTensor::size, nb::arg("dim"), "Size of dimension")

            // Scalar extraction
            .def("item", &PyTensor::item, "Extract scalar value")
            .def("float_", &PyTensor::item_float, "Extract as float")
            .def("int_", &PyTensor::item_int, "Extract as int")
            .def("bool_", &PyTensor::item_bool, "Extract as bool")

            // NumPy conversion
            .def("numpy", &PyTensor::numpy, nb::arg("copy") = true,
                 "Convert to NumPy array")
            .def_static("from_numpy", &PyTensor::from_numpy,
                        nb::arg("arr"), nb::arg("copy") = true,
                        "Create tensor from NumPy array")

            // Indexing
            .def("__getitem__", &PyTensor::getitem, "Get item/slice")
            .def("__setitem__", &PyTensor::setitem, "Set item/slice")

            // Arithmetic operators
            .def("__add__", &PyTensor::add, "Add tensor")
            .def("__add__", &PyTensor::add_scalar, "Add scalar")
            .def("__radd__", &PyTensor::add_scalar, "Reverse add scalar")
            .def(
                "__iadd__", [](PyTensor& self, const PyTensor& other) -> PyTensor& {
                    return self.iadd(other);
                },
                nb::rv_policy::reference, "In-place add tensor")
            .def(
                "__iadd__", [](PyTensor& self, float scalar) -> PyTensor& {
                    return self.iadd_scalar(scalar);
                },
                nb::rv_policy::reference, "In-place add scalar")

            .def("__sub__", &PyTensor::sub, "Subtract tensor")
            .def("__sub__", &PyTensor::sub_scalar, "Subtract scalar")
            .def("__rsub__", &PyTensor::rsub_scalar, "Reverse subtract scalar")
            .def(
                "__isub__", [](PyTensor& self, const PyTensor& other) -> PyTensor& {
                    return self.isub(other);
                },
                nb::rv_policy::reference, "In-place subtract tensor")
            .def(
                "__isub__", [](PyTensor& self, float scalar) -> PyTensor& {
                    return self.isub_scalar(scalar);
                },
                nb::rv_policy::reference, "In-place subtract scalar")

            .def("__mul__", &PyTensor::mul, "Multiply tensor")
            .def("__mul__", &PyTensor::mul_scalar, "Multiply scalar")
            .def("__rmul__", &PyTensor::mul_scalar, "Reverse multiply scalar")
            .def(
                "__imul__", [](PyTensor& self, const PyTensor& other) -> PyTensor& {
                    return self.imul(other);
                },
                nb::rv_policy::reference, "In-place multiply tensor")
            .def(
                "__imul__", [](PyTensor& self, float scalar) -> PyTensor& {
                    return self.imul_scalar(scalar);
                },
                nb::rv_policy::reference, "In-place multiply scalar")

            .def("__truediv__", &PyTensor::div, "Divide tensor")
            .def("__truediv__", &PyTensor::div_scalar, "Divide scalar")
            .def("__rtruediv__", &PyTensor::rdiv_scalar, "Reverse divide scalar")
            .def(
                "__itruediv__", [](PyTensor& self, const PyTensor& other) -> PyTensor& {
                    return self.idiv(other);
                },
                nb::rv_policy::reference, "In-place divide tensor")
            .def(
                "__itruediv__", [](PyTensor& self, float scalar) -> PyTensor& {
                    return self.idiv_scalar(scalar);
                },
                nb::rv_policy::reference, "In-place divide scalar")

            .def("__neg__", &PyTensor::neg, "Negate")
            .def("__abs__", &PyTensor::abs, "Absolute value")

            // Comparison operators
            .def("__eq__", &PyTensor::eq, "Equal tensor")
            .def("__eq__", &PyTensor::eq_scalar, "Equal scalar")
            .def("__ne__", &PyTensor::ne, "Not equal tensor")
            .def("__ne__", &PyTensor::ne_scalar, "Not equal scalar")
            .def("__lt__", &PyTensor::lt, "Less than tensor")
            .def("__lt__", &PyTensor::lt_scalar, "Less than scalar")
            .def("__le__", &PyTensor::le, "Less equal tensor")
            .def("__le__", &PyTensor::le_scalar, "Less equal scalar")
            .def("__gt__", &PyTensor::gt, "Greater than tensor")
            .def("__gt__", &PyTensor::gt_scalar, "Greater than scalar")
            .def("__ge__", &PyTensor::ge, "Greater equal tensor")
            .def("__ge__", &PyTensor::ge_scalar, "Greater equal scalar")

            // Logical operators
            .def("__and__", &PyTensor::logical_and, "Logical AND")
            .def("__or__", &PyTensor::logical_or, "Logical OR")
            .def("__invert__", &PyTensor::logical_not, "Logical NOT")

            // Reduction methods
            .def("sum", &PyTensor::sum, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Sum reduction")
            .def("mean", &PyTensor::mean, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Mean reduction")
            .def("max", &PyTensor::max, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Max reduction")
            .def("min", &PyTensor::min, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, "Min reduction")

            // Shape operations
            .def("reshape", &PyTensor::reshape, nb::arg("shape"), "Reshape tensor")
            .def("view", &PyTensor::view, nb::arg("shape"), "View tensor with new shape")
            .def("squeeze", &PyTensor::squeeze, nb::arg("dim") = nb::none(), "Remove size-1 dimensions")
            .def("unsqueeze", &PyTensor::unsqueeze, nb::arg("dim"), "Add size-1 dimension")
            .def("transpose", &PyTensor::transpose, nb::arg("dim0"), nb::arg("dim1"), "Transpose dimensions")
            .def("permute", &PyTensor::permute, nb::arg("dims"), "Permute dimensions")
            .def("flatten", &PyTensor::flatten, nb::arg("start_dim") = 0, nb::arg("end_dim") = -1, "Flatten dimensions")

            // String representation
            .def("__repr__", &PyTensor::repr);
    }

} // namespace lfs::python
