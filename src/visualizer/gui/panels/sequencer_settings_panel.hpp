/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include "io/video/video_export_options.hpp"

namespace lfs::vis::gui::panels {

    struct SequencerUIState {
        bool show_camera_path = true;
        lfs::io::video::VideoResolution resolution = lfs::io::video::VideoResolution::FHD_1080P;
        int framerate = 30;
        int quality = 18; // CRF
    };

    void DrawSequencerSection(const UIContext& ctx, SequencerUIState& state);

} // namespace lfs::vis::gui::panels
