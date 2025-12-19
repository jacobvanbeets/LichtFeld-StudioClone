/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "sequencer_controller.hpp"
#include <imgui.h>

namespace lfs::vis {

    namespace panel_config {
        inline constexpr float HEIGHT = 72.0f;
        inline constexpr float PADDING_H = 16.0f;
        inline constexpr float PADDING_BOTTOM = 8.0f;
        inline constexpr float INNER_PADDING = 8.0f;
        inline constexpr float RULER_HEIGHT = 16.0f;
        inline constexpr float TIMELINE_HEIGHT = 24.0f;
        inline constexpr float KEYFRAME_RADIUS = 6.0f;
        inline constexpr float PLAYHEAD_WIDTH = 2.0f;
        inline constexpr float BUTTON_SIZE = 20.0f;
        inline constexpr float BUTTON_SPACING = 4.0f;
        inline constexpr float TRANSPORT_WIDTH = 152.0f;
        inline constexpr float TIME_DISPLAY_WIDTH = 100.0f;
    }

    class SequencerPanel {
    public:
        explicit SequencerPanel(SequencerController& controller);
        void render(float viewport_x, float viewport_width, float viewport_y_bottom);

    private:
        void renderTransportControls(const ImVec2& pos, float height);
        void renderTimeline(const ImVec2& pos, float width, float height);
        void renderTimeRuler(ImDrawList* dl, const ImVec2& pos, float width);
        void renderTimeDisplay(const ImVec2& pos, float height);

        void drawKeyframeMarker(ImDrawList* dl, const ImVec2& pos, bool selected, bool hovered, float time) const;
        void drawPlayhead(ImDrawList* dl, const ImVec2& top, const ImVec2& bottom) const;

        [[nodiscard]] float getDisplayEndTime() const;
        [[nodiscard]] float timeToX(float time, float timeline_x, float timeline_width) const;
        [[nodiscard]] float xToTime(float x, float timeline_x, float timeline_width) const;

        bool dragging_playhead_ = false;
        bool dragging_keyframe_ = false;
        size_t dragged_keyframe_index_ = 0;
        std::optional<size_t> hovered_keyframe_;

        // Context menu state
        bool context_menu_open_ = false;
        float context_menu_time_ = 0.0f;
        std::optional<size_t> context_menu_keyframe_;

        SequencerController& controller_;
    };

} // namespace lfs::vis
