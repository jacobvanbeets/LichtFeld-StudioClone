/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sequencer_controller.hpp"
#include <algorithm>

namespace lfs::vis {

    void SequencerController::play() {
        if (timeline_.empty()) return;
        if (state_ == PlaybackState::STOPPED) {
            playhead_ = timeline_.startTime();
            reverse_direction_ = false;
        }
        state_ = PlaybackState::PLAYING;
    }

    void SequencerController::pause() {
        if (state_ == PlaybackState::PLAYING) {
            state_ = PlaybackState::PAUSED;
        }
    }

    void SequencerController::stop() {
        state_ = PlaybackState::STOPPED;
        playhead_ = timeline_.startTime();
        reverse_direction_ = false;
    }

    void SequencerController::togglePlayPause() {
        isPlaying() ? pause() : play();
    }

    void SequencerController::seek(const float time) {
        playhead_ = timeline_.empty() ? 0.0f : std::clamp(time, timeline_.startTime(), timeline_.endTime());
    }

    void SequencerController::seekToFirstKeyframe() {
        if (!timeline_.empty()) {
            playhead_ = timeline_.startTime();
            if (state_ == PlaybackState::PLAYING) {
                state_ = PlaybackState::PAUSED;
            }
        }
    }

    void SequencerController::seekToLastKeyframe() {
        if (!timeline_.empty()) {
            playhead_ = timeline_.endTime();
            if (state_ == PlaybackState::PLAYING) {
                state_ = PlaybackState::PAUSED;
            }
        }
    }

    void SequencerController::toggleLoop() {
        loop_mode_ = (loop_mode_ == LoopMode::ONCE) ? LoopMode::LOOP : LoopMode::ONCE;
    }

    void SequencerController::beginScrub() {
        state_ = PlaybackState::SCRUBBING;
    }

    void SequencerController::scrub(const float time) {
        playhead_ = std::clamp(time, timeline_.startTime(), timeline_.endTime());
    }

    void SequencerController::endScrub() {
        state_ = PlaybackState::PAUSED;
    }

    bool SequencerController::update(const float delta_seconds) {
        if (state_ != PlaybackState::PLAYING || timeline_.empty()) {
            return false;
        }

        const float start = timeline_.startTime();
        const float end = timeline_.endTime();
        const float delta = delta_seconds * playback_speed_ * (reverse_direction_ ? -1.0f : 1.0f);

        playhead_ += delta;

        switch (loop_mode_) {
            case LoopMode::ONCE:
                if (playhead_ >= end) {
                    playhead_ = end;
                    state_ = PlaybackState::STOPPED;
                } else if (playhead_ < start) {
                    playhead_ = start;
                    state_ = PlaybackState::STOPPED;
                }
                break;

            case LoopMode::LOOP:
                if (playhead_ >= end) {
                    playhead_ = start + (playhead_ - end);
                } else if (playhead_ < start) {
                    playhead_ = end - (start - playhead_);
                }
                break;

            case LoopMode::PING_PONG:
                if (playhead_ >= end) {
                    playhead_ = end - (playhead_ - end);
                    reverse_direction_ = true;
                } else if (playhead_ < start) {
                    playhead_ = start + (start - playhead_);
                    reverse_direction_ = false;
                }
                break;
        }
        return true;
    }

    void SequencerController::addKeyframe(const sequencer::Keyframe& keyframe) {
        timeline_.addKeyframe(keyframe);
    }

    void SequencerController::updateSelectedKeyframe(const glm::vec3& position, const glm::quat& rotation, const float fov) {
        if (!selected_keyframe_ || *selected_keyframe_ >= timeline_.size()) return;
        timeline_.updateKeyframe(*selected_keyframe_, position, rotation, fov);
    }

    void SequencerController::removeSelectedKeyframe() {
        if (!selected_keyframe_) return;
        timeline_.removeKeyframe(*selected_keyframe_);
        deselectKeyframe();
    }

    void SequencerController::selectKeyframe(const size_t index) {
        if (index < timeline_.size()) {
            selected_keyframe_ = index;
        }
    }

    void SequencerController::deselectKeyframe() {
        selected_keyframe_ = std::nullopt;
    }

    sequencer::CameraState SequencerController::currentCameraState() const {
        return timeline_.evaluate(playhead_);
    }

} // namespace lfs::vis
