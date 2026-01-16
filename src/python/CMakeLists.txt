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
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}
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
option(BUILD_PYTHON_STUBS "Generate Python type stubs (requires loading the module)" ON)
if(BUILD_PYTHON_STUBS)
    if(WIN32)
        # On Windows, we need to set PATH for DLL loading before running stubgen
        # Find the stubgen script (NB_DIR is set by find_package(nanobind))
        if(EXISTS "${NB_DIR}/src/stubgen.py")
            set(NB_STUBGEN "${NB_DIR}/src/stubgen.py")
        elseif(EXISTS "${NB_DIR}/stubgen.py")
            set(NB_STUBGEN "${NB_DIR}/stubgen.py")
        else()
            message(WARNING "nanobind stubgen.py not found at ${NB_DIR}, disabling stub generation")
            set(BUILD_PYTHON_STUBS OFF)
        endif()

        if(BUILD_PYTHON_STUBS)
            # Build PATH with all required DLL directories
            set(STUBGEN_DLL_PATHS
                "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin"
                "${CMAKE_BINARY_DIR}"
            )
            # Add CUDA bin if available
            if(DEFINED ENV{CUDA_PATH})
                list(APPEND STUBGEN_DLL_PATHS "$ENV{CUDA_PATH}/bin")
            elseif(CUDAToolkit_BIN_DIR)
                list(APPEND STUBGEN_DLL_PATHS "${CUDAToolkit_BIN_DIR}")
            endif()
            list(JOIN STUBGEN_DLL_PATHS "\\;" STUBGEN_PATH_STR)

            add_custom_command(
                OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/typings/lichtfeld/py.typed
                COMMAND ${CMAKE_COMMAND} -E env
                        "PATH=${STUBGEN_PATH_STR}\\;$ENV{PATH}"
                        "PYTHONPATH=$<TARGET_FILE_DIR:lfs_py>"
                        "${Python_EXECUTABLE}" "${NB_STUBGEN}"
                        -m lichtfeld
                        -M ${CMAKE_CURRENT_BINARY_DIR}/typings/lichtfeld/py.typed
                        -O ${CMAKE_CURRENT_BINARY_DIR}/typings
                        -r
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                DEPENDS lfs_py "${NB_STUBGEN}"
                COMMENT "Generating Python stubs for lichtfeld (Windows)"
            )
            add_custom_target(lichtfeld_stub ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/typings/lichtfeld/py.typed)
        endif()
    else()
        nanobind_add_stub(lichtfeld_stub
            MODULE lichtfeld
            OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/typings
            PYTHON_PATH ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS lfs_py
            RECURSIVE
            MARKER_FILE ${CMAKE_CURRENT_BINARY_DIR}/typings/lichtfeld/py.typed
        )
    endif()
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

# =============================================================================
# PORTABLE BUILD - Bundle vcpkg Python interpreter and standard library
# =============================================================================
if(BUILD_PORTABLE AND BUILD_PYTHON_BINDINGS)
    message(STATUS "Portable build: bundling Python ${Python_VERSION}")

    # Get vcpkg Python directories
    get_filename_component(VCPKG_PYTHON_TOOLS_DIR "${Python_EXECUTABLE}" DIRECTORY)

    if(WIN32)
        # Windows: Python layout is tools/python3/{python.exe, Lib/, DLLs/, python312.dll}
        # We install to lib/python3.12/ to match Linux layout for simpler getPythonHome()
        set(VCPKG_PYTHON_STDLIB "${VCPKG_PYTHON_TOOLS_DIR}/Lib")
        set(VCPKG_PYTHON_DLLS "${VCPKG_PYTHON_TOOLS_DIR}/DLLs")
        set(PYTHON_INSTALL_STDLIB "${CMAKE_INSTALL_LIBDIR}/python3.12")

        # Install Python executable to bin/
        install(PROGRAMS "${Python_EXECUTABLE}"
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
            COMPONENT runtime
        )

        # Install python3.dll and python312.dll to bin/ (where exe is)
        file(GLOB PYTHON_DLLS "${VCPKG_PYTHON_TOOLS_DIR}/python*.dll")
        if(PYTHON_DLLS)
            install(FILES ${PYTHON_DLLS}
                DESTINATION "${CMAKE_INSTALL_BINDIR}"
                COMPONENT runtime
            )
        endif()

        # Also check parent dir for python DLLs (vcpkg sometimes puts them there)
        file(GLOB PYTHON_DLLS_PARENT "${VCPKG_PYTHON_TOOLS_DIR}/../bin/python*.dll")
        if(PYTHON_DLLS_PARENT)
            install(FILES ${PYTHON_DLLS_PARENT}
                DESTINATION "${CMAKE_INSTALL_BINDIR}"
                COMPONENT runtime
            )
        endif()

        # Install Python standard library
        if(EXISTS "${VCPKG_PYTHON_STDLIB}")
            install(DIRECTORY "${VCPKG_PYTHON_STDLIB}/"
                DESTINATION "${PYTHON_INSTALL_STDLIB}"
                COMPONENT runtime
                PATTERN "__pycache__" EXCLUDE
                PATTERN "*.pyc" EXCLUDE
                PATTERN "test" EXCLUDE
                PATTERN "tests" EXCLUDE
                PATTERN "idle_test" EXCLUDE
                PATTERN "tkinter" EXCLUDE
                PATTERN "turtledemo" EXCLUDE
            )
            message(STATUS "  Will bundle Python stdlib from: ${VCPKG_PYTHON_STDLIB}")
        else()
            message(WARNING "Python stdlib not found at ${VCPKG_PYTHON_STDLIB}")
        endif()

        # Install DLLs directory (compiled extensions like _socket.pyd, etc.)
        if(EXISTS "${VCPKG_PYTHON_DLLS}")
            install(DIRECTORY "${VCPKG_PYTHON_DLLS}/"
                DESTINATION "${PYTHON_INSTALL_STDLIB}/lib-dynload"
                COMPONENT runtime
                PATTERN "__pycache__" EXCLUDE
            )
        endif()

    else()
        # Linux: Python layout is tools/python3/python3.12, lib/python3.12/
        set(VCPKG_PYTHON_STDLIB "${VCPKG_PYTHON_TOOLS_DIR}/../../lib/python3.12")
        get_filename_component(VCPKG_PYTHON_STDLIB "${VCPKG_PYTHON_STDLIB}" ABSOLUTE)
        set(PYTHON_INSTALL_STDLIB "${CMAKE_INSTALL_LIBDIR}/python3.12")

        # Install Python executable
        install(PROGRAMS "${Python_EXECUTABLE}"
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
            RENAME "python3"
            COMPONENT runtime
        )

        # Create python3.12 symlink
        install(CODE "
            execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink
                python3 \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/python3.12\")
        " COMPONENT runtime)

        # Install libpython if it exists (shared library builds)
        file(GLOB PYTHON_SHARED_LIBS "${VCPKG_PYTHON_TOOLS_DIR}/../../lib/libpython3.12.so*")
        if(PYTHON_SHARED_LIBS)
            install(FILES ${PYTHON_SHARED_LIBS}
                DESTINATION "${CMAKE_INSTALL_LIBDIR}"
                COMPONENT runtime
            )
        endif()

        # Install Python standard library
        if(EXISTS "${VCPKG_PYTHON_STDLIB}")
            install(DIRECTORY "${VCPKG_PYTHON_STDLIB}/"
                DESTINATION "${PYTHON_INSTALL_STDLIB}"
                COMPONENT runtime
                PATTERN "__pycache__" EXCLUDE
                PATTERN "*.pyc" EXCLUDE
                PATTERN "test" EXCLUDE
                PATTERN "tests" EXCLUDE
                PATTERN "idle_test" EXCLUDE
                PATTERN "tkinter" EXCLUDE
                PATTERN "turtledemo" EXCLUDE
                PATTERN "lib-dynload" EXCLUDE
            )

            # Install lib-dynload separately (compiled extensions)
            if(EXISTS "${VCPKG_PYTHON_STDLIB}/lib-dynload")
                install(DIRECTORY "${VCPKG_PYTHON_STDLIB}/lib-dynload"
                    DESTINATION "${PYTHON_INSTALL_STDLIB}"
                    COMPONENT runtime
                )
            endif()
        endif()
    endif()

    message(STATUS "  Python executable: ${Python_EXECUTABLE}")
    message(STATUS "  Python stdlib: ${VCPKG_PYTHON_STDLIB}")
endif()
