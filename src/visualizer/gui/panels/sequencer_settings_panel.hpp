/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include "io/video/video_export_options.hpp"

namespace lfs::vis::gui::panels {

    struct SequencerUIState {
        bool show_camera_path = true;
        bool snap_to_grid = false;
        float snap_interval = 0.5f;
        float playback_speed = 1.0f;

        // Playback preview
        bool follow_playback = false;   // Main camera follows playhead (default: off)
        float pip_preview_scale = 1.0f; // Preview window scale (0.5 to 2.0)

        // Video export
        lfs::io::video::VideoPreset preset = lfs::io::video::VideoPreset::YOUTUBE_1080P;
        int custom_width = 1920;
        int custom_height = 1080;
        int framerate = 30;
        int quality = 18;
    };

    void DrawSequencerSection(const UIContext& ctx, SequencerUIState& state);

} // namespace lfs::vis::gui::panels
