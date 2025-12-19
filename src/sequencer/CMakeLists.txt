# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_library(lfs_sequencer STATIC
        interpolation.cpp
        timeline.cpp
)

target_include_directories(lfs_sequencer
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(lfs_sequencer
        PUBLIC
        glm::glm
        PRIVATE
        lfs_core
        nlohmann_json::nlohmann_json
)

target_compile_features(lfs_sequencer PUBLIC cxx_std_23)
