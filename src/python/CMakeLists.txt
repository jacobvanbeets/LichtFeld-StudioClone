#SPDX - FileCopyrightText : 2025 LichtFeld Studio Authors
#
#SPDX - License - Identifier : GPL - 3.0 - or -later

if (NOT BUILD_PYTHON_BINDINGS)
    return()
endif()

message(STATUS "Configuring Python components (runner + nanobind module)")

#Python for interpreter + embedding + nanobind
find_package(Python REQUIRED COMPONENTS Interpreter Development Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

add_library(lfs_python_utils STATIC
    runner.cpp
    package_manager.cpp
    subprocess.cpp
    uv_runner.cpp
    py_panel_registry.cpp
    ui_hooks.cpp
)

target_include_directories(lfs_python_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src # for includes like "python/runner.hpp"
)

target_link_libraries(lfs_python_utils
    PUBLIC
        Python::Python
        lfs_core
)

target_compile_definitions(lfs_python_utils
    PUBLIC
        LFS_BUILD_PYTHON_BINDINGS
        LFS_PYTHON_EXECUTABLE="${Python_EXECUTABLE}"
)

set_target_properties(lfs_python_utils PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

# nanobind module (Python import name: lichtfeld)
nanobind_add_module(
    lfs_py
    STABLE_ABI
    NB_STATIC
    lfs/module.cpp
    lfs/py_tensor.cpp
    lfs/py_splat_data.cpp
    lfs/py_scene.cpp
    lfs/py_cameras.cpp
    lfs/py_io.cpp
    lfs/py_packages.cpp
    lfs/py_params.cpp
    lfs/py_ui.cpp
)

set_target_properties(lfs_py PROPERTIES
    CUDA_ARCHITECTURES ""
    CUDA_SEPARABLE_COMPILATION OFF
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME lichtfeld
)

#Need to explicitly link tensor kernels since they're linked privately in lfs_tensor
#When building a shared library(nanobind module), all symbols must be resolved at link time
target_link_libraries(lfs_py PRIVATE
    lfs_core
    lfs_tensor
    lfs_tensor_kernels
    lfs_training
    lfs_geometry
    lfs_io
    lfs_visualizer
    OpenImageIO::OpenImageIO
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

target_include_directories(lfs_py PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/visualizer
)

# Generate type stubs for IDE support
nanobind_add_stub(lichtfeld_stub
    MODULE lichtfeld
    OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}
    PYTHON_PATH ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS lfs_py
    RECURSIVE
    MARKER_FILE ${CMAKE_CURRENT_BINARY_DIR}/lichtfeld/py.typed
)

# UV package manager for Python package installation support
include(${CMAKE_SOURCE_DIR}/cmake/FetchUV.cmake)
fetch_uv()
copy_uv_to_build()
install_uv()
