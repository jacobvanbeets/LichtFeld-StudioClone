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

# Shared library for panel registry - must be shared to avoid duplicate statics
# when linked into both the main executable and the Python module
add_library(lfs_panel_registry SHARED
    py_panel_registry.cpp
)

target_include_directories(lfs_panel_registry
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(lfs_panel_registry
    PUBLIC
        Python::Python
)

target_compile_definitions(lfs_panel_registry
    PRIVATE
        LFS_PANEL_REGISTRY_EXPORTS
)

set_target_properties(lfs_panel_registry PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
)

# Static library for other Python utilities
add_library(lfs_python_utils STATIC
    runner.cpp
    package_manager.cpp
    subprocess.cpp
    uv_runner.cpp
    ui_hooks.cpp
)

target_include_directories(lfs_python_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(lfs_python_utils
    PUBLIC
        Python::Python
        lfs_core
        lfs_panel_registry
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
    lfs/py_plugins.cpp
    lfs/py_ui.cpp
    lfs/py_mcp.cpp
    lfs/py_rendering.cpp
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
    lfs_event_bridge    # Explicit: shared event bus for Python<->C++ events
    lfs_tensor
    lfs_tensor_kernels
    lfs_training
    lfs_geometry
    lfs_io
    lfs_visualizer
    lfs_rendering       # For render_view, compute_screen_positions
    lfs_mcp
    lfs_panel_registry
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

# Multi-config generators (MSVC) output to Release/Debug subdirs - copy to parent for Python to find
if(CMAKE_CONFIGURATION_TYPES)
    add_custom_command(TARGET lfs_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:lfs_py>"
            "${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:lfs_py>"
        COMMENT "Copying $<TARGET_FILE_NAME:lfs_py> to ${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

# Type stubs for IDE support (output to typings/ to avoid runtime conflicts)
# Disabled by default on Windows CI where CUDA DLLs aren't in Python's DLL search path
option(BUILD_PYTHON_STUBS "Generate Python type stubs (requires loading the module)" ON)
if(BUILD_PYTHON_STUBS)
    nanobind_add_stub(lichtfeld_stub
        MODULE lichtfeld
        OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/typings
        PYTHON_PATH ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS lfs_py
        RECURSIVE
        MARKER_FILE ${CMAKE_CURRENT_BINARY_DIR}/typings/lichtfeld/py.typed
    )
endif()

# Copy lfs_plugins Python module to build directory
file(GLOB LFS_PLUGINS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/lfs_plugins/*.py")
add_custom_target(lfs_plugins_copy ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/lfs_plugins
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LFS_PLUGINS_FILES} ${CMAKE_CURRENT_BINARY_DIR}/lfs_plugins/
    COMMENT "Copying lfs_plugins Python module"
)
add_dependencies(lfs_py lfs_plugins_copy)

# UV package manager for Python package installation support
include(${CMAKE_SOURCE_DIR}/cmake/FetchUV.cmake)
fetch_uv()
copy_uv_to_build()
install_uv()

# =============================================================================
# INSTALLATION - Python SDK components for plugin development
# =============================================================================
set(PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/python")

# Install lichtfeld Python module (.so/.pyd)
install(TARGETS lfs_py
    LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}"
    RUNTIME DESTINATION "${PYTHON_INSTALL_DIR}"  # Windows DLLs
)

# Install lfs_panel_registry shared library (required by lichtfeld module)
install(TARGETS lfs_panel_registry
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"  # Windows DLLs
)

# Install lfs_plugins Python package
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lfs_plugins"
    DESTINATION "${PYTHON_INSTALL_DIR}"
    FILES_MATCHING PATTERN "*.py"
)

# Install type stubs for IDE support (if built)
if(BUILD_PYTHON_STUBS)
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/typings/"
        DESTINATION "${PYTHON_INSTALL_DIR}"
        OPTIONAL  # May not exist if stub generation failed
    )
endif()
