# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Event bridge SHARED library - solves singleton duplication between
# main executable and Python module

add_library(lfs_event_bridge SHARED
    event_bridge.cpp
)

target_include_directories(lfs_event_bridge
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(lfs_event_bridge
    PRIVATE
        LFS_EVENT_BRIDGE_EXPORTS
)

set_target_properties(lfs_event_bridge PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
