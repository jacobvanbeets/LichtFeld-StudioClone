# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# External dependencies for new LichtFeld-Studio build (LibTorch-free)
# These are NOT used by the legacy build

# ImGuizmo - 3D manipulation gizmo library
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/imguizmo/CMakeLists.txt")
    message(STATUS "Adding ImGuizmo from external/imguizmo")
    add_subdirectory(imguizmo)
else()
    message(WARNING "ImGuizmo not found at external/imguizmo - Gizmo controls will not be available")
endif()

# ImGuiColorTextEdit - Syntax highlighting text editor for ImGui
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ImGuiColorTextEdit/TextEditor.cpp")
    message(STATUS "Adding ImGuiColorTextEdit from external/ImGuiColorTextEdit")
    add_library(imgui_textedit STATIC
        ImGuiColorTextEdit/TextEditor.cpp
    )
    target_include_directories(imgui_textedit PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/ImGuiColorTextEdit
    )
    target_link_libraries(imgui_textedit PUBLIC imgui::imgui)
    set_target_properties(imgui_textedit PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
else()
    message(WARNING "ImGuiColorTextEdit not found at external/ImGuiColorTextEdit")
endif()

# nvImageCodec - GPU-accelerated image codec library
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nvImageCodec/CMakeLists.txt")
    message(STATUS "Adding nvImageCodec from external/nvImageCodec")

    # Match main project CUDA arch (e.g., "8.6" -> "86")
    if(DEFINED DETECTED_COMPUTE_CAP)
        string(REPLACE "." "" _arch "${DETECTED_COMPUTE_CAP}")
        set(CUDA_TARGET_ARCHS "${_arch}" CACHE STRING "" FORCE)
    endif()

    set(BUILD_NVJPEG_EXT ON CACHE BOOL "" FORCE)
    set(BUILD_NVJPEG2K_EXT ON CACHE BOOL "" FORCE)
    set(WITH_DYNAMIC_NVJPEG OFF CACHE BOOL "" FORCE)
    set(WITH_DYNAMIC_NVJPEG2K OFF CACHE BOOL "" FORCE)

    add_subdirectory(nvImageCodec)
else()
    message(WARNING "nvImageCodec not found at external/nvImageCodec - GPU image loading will not be available")
endif()
